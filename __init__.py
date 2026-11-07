"""ComfyUI-Krea2-Anime-Capgen: Danbooru tags -> captions in the style of the Krea 2 Anime-tuned training captions."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, add_routes, setup_folders

WEB_DIRECTORY = './web'
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

setup_folders()
try:
    from server import PromptServer
except ImportError:         # imported outside a ComfyUI server
    PromptServer = None
if PromptServer is not None and getattr(PromptServer, 'instance', None) is not None:
    add_routes(PromptServer.instance)
