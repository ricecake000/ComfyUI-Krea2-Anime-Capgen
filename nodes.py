"""ComfyUI node Krea2AnimeCapgen and the autocomplete route GET /krea2capgen/suggest?q=&k=."""
import asyncio
import os

from . import capgen_bridge as B

TAGS_TOOLTIP = ('Danbooru tags, comma-separated; "-tag" excludes. Captions lean to white background and rating s: '
                'add e.g. "-white background, -simple background".')


def setup_folders():
    """Register the "capgen" model folder (<models>/capgen, plus extra_model_paths.yaml entries) for the data."""
    import folder_paths
    folder_paths.add_model_folder_path('capgen', os.path.join(folder_paths.models_dir, 'capgen'), is_default=True)
    B.SEARCH_DIRS[:] = folder_paths.get_folder_paths('capgen')


class Krea2AnimeCapgen:
    """One caption in the style of the Krea 2 Anime-tuned training captions, built from Danbooru tags (no LLM)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'tags': ('STRING', {'multiline': True, 'default': 'cat ears, school uniform, cherry blossoms',
                                'tooltip': TAGS_TOOLTIP}),
            'length': (list(B.LENGTHS), {'default': 'long',
                                         'tooltip': 'Caption length: long ~128 tokens, medium ~74, short ~32.'}),
            # the frontend's numbers are doubles: seeds above 2^53 - 1 would not round-trip
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 2 ** 53 - 1, 'control_after_generate': True,
                             'tooltip': 'The same tags, options and seed give the same caption.'}),
            'wild': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                               'tooltip': '1 = surprising combinations of phrase donors, 0 = plausible ones.'}),
            'overlay_text': ('BOOLEAN', {'default': False, 'tooltip': 'Keep marks on the picture (watermark, artist '
                                         'signature, corner text, logo). Off: dropped unless a tag asks for one.'}),
            'scene_text': ('BOOLEAN', {'default': True, 'tooltip': 'Keep text on scene objects (shirt prints, signs, '
                                       'book covers, speech bubbles).'}),
            'mix_look': ('BOOLEAN', {'default': True, 'tooltip': 'Take hair, eyes and body traits from three '
                                     'different source captions (off: one appearance source).'}),
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('caption', 'info')
    OUTPUT_TOOLTIPS = ('The caption: feed it to the text encoder.',
                       'How each tag was understood, dropped / rejected tags and warnings.')
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = 'Krea2 Anime'
    DESCRIPTION = ('Turns Danbooru tags into a caption in the style of the Krea 2 Anime-tuned training captions. '
                   'Runs on the CPU; downloads its data (~450 MB) on first use.')
    SEARCH_ALIASES = ['capgen', 'caption', 'tags to prompt', 'danbooru', 'krea']

    def run(self, tags, length, seed, wild, overlay_text, scene_text, mix_look):
        cap, info = B.caption(tags, length, seed, wild, overlay_text, scene_text, mix_look)
        return {'ui': {'text': (cap,)}, 'result': (cap, info)}


def add_routes(server):
    from aiohttp import web

    @server.routes.get('/krea2capgen/suggest')
    async def suggest(request):
        q = request.query.get('q', '')
        try:
            k = max(1, min(50, int(request.query.get('k', 12))))
        except ValueError:
            k = 12
        st = B.status()
        if st != 'ready':
            return web.json_response({'status': st, 'items': []})
        if not q.strip(' -_'):
            return web.json_response({'status': st, 'items': []})
        items = await asyncio.get_running_loop().run_in_executor(None, B.suggest, q[:200], k)
        return web.json_response({'status': st, 'items': items})


NODE_CLASS_MAPPINGS = {'Krea2AnimeCapgen': Krea2AnimeCapgen}
NODE_DISPLAY_NAME_MAPPINGS = {'Krea2AnimeCapgen': 'Krea 2 Anime Caption (capgen)'}
