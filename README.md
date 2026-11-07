# ComfyUI-Krea2-Anime-Capgen

A ComfyUI node that turns a few Danbooru tags into a caption in the style of the training captions of
**Krea 2 Anime-tuned** ("anime illustration of ..., comma phrases, ..."). The model was fine-tuned on such
captions, so prompts written this way match what it learned.

It uses no LLM. Each caption is assembled from whole phrase groups copied out of real training captions,
picked so that every tag you give is expressed. It runs on the CPU in well under a second per caption.

Links:

- Models: [ricecake/Krea-2-Anime-tuned-Turbo](https://huggingface.co/ricecake/Krea-2-Anime-tuned-Turbo) and
  [ricecake/Krea-2-Anime-tuned-Raw](https://huggingface.co/ricecake/Krea-2-Anime-tuned-Raw)
- capgen data (downloaded by the node): [ricecake/krea2-anime-capgen](https://huggingface.co/ricecake/krea2-anime-capgen)
- Try it in the browser: [demo Space](https://huggingface.co/spaces/ricecake/krea2-anime-capgen)
- This node: [github.com/ricecake000/ComfyUI-Krea2-Anime-Capgen](https://github.com/ricecake000/ComfyUI-Krea2-Anime-Capgen)

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ricecake000/ComfyUI-Krea2-Anime-Capgen
```

or, in ComfyUI-Manager, "Install via Git URL" with `https://github.com/ricecake000/ComfyUI-Krea2-Anime-Capgen`.
Then restart ComfyUI. The node is not on the Comfy Registry yet. It needs nothing beyond what ComfyUI already
installs (Python standard library and numpy). Tested with ComfyUI v0.44.0 (frontend 1.57.6).

### Data: first-run download (~450 MB)

On first use (the first run of the node, or the first autocomplete request) the node downloads its data from
the Hugging Face repo `ricecake/krea2-anime-capgen`, at the exact revision pinned in `vendor/PIN.json`, into

```
ComfyUI/models/capgen/<version>/
    RUNTIME.json  char_series.json  data/...  model/...
```

`<version>` is `runtime_version` in `vendor/PIN.json`. The download goes to a temporary folder, each file is
checked against the pinned size and sha256, and the folder is renamed into place only when everything is there.
An interrupted download leaves nothing behind. If `<version>/` exists but is incomplete, it is left
alone (a warning says it is not used and may be deleted) and the download goes to a new `<version>.repair-.../`
next to it, which is found on later starts. Later starts read the folder without touching the network. Loading
takes about 5 s and about 1 GB of RAM, once per ComfyUI process.

**Offline / manual install.** Download `RUNTIME.json` and every file listed under `files` in `vendor/PIN.json`
from the pinned revision of `ricecake/krea2-anime-capgen`, and put them in `ComfyUI/models/capgen/<version>/`,
keeping the sub-folders (`model/`, `data/`). The node then never goes online. Other options:

| setting | effect |
|---|---|
| `capgen:` entry in `extra_model_paths.yaml` | extra folders searched for `<version>/` (downloads go to the first `capgen` folder) |
| `KREA2_CAPGEN_RUNTIME=/path/to/dir` | use this capgen runtime dir directly (it must be the pinned version) |
| `KREA2_CAPGEN_OFFLINE=1` or `HF_HUB_OFFLINE=1` | never download; a missing folder is an error with these instructions |
| `HF_ENDPOINT` | Hugging Face mirror to download from |

## The node

**Krea 2 Anime Caption (capgen)**, category `Krea2 Anime`.

| input | default | |
|---|---|---|
| `tags` | | Danbooru tags, separated by commas or line breaks. `_` and spaces are the same. |
| `length` | long | long (~128 tokens), medium (~74), short (~32) |
| `seed` | 0 | the same tags, options and seed always give the same caption; use *control after generate* = randomize for a new caption on every run |
| `wild` | 1.0 | 0 to 1. 1 = surprising combinations of the phrase sources, 0 = plausible ones (sources that often go together) |
| `overlay_text` | off | on keeps marks on the picture (watermark, artist signature, corner text, logo); off drops them unless a tag asks for one |
| `scene_text` | on | off also drops text on objects in the scene (shirt prints, signs, book covers, speech bubbles) |
| `mix_look` | on | hair, eyes and body traits come from three different source captions; off = one appearance source |

| output | |
|---|---|
| `caption` | the caption; connect it to the `text` of CLIPTextEncode |
| `info` | how each tag was understood (tag, character, series, free phrase, exclusion), dropped and rejected tags with the reason, tags that could not be expressed, warnings, the token estimate |

The caption is also shown on the node after each run. The node is an output node, so you can queue it alone to
read captions before rendering.

What you can write in `tags`:

- a tag: `cat ears`, `school uniform`, `night`
- a character (`hakurei reimu`, `kanami (strinova)`, or `kanami` when unique) or a series (`blue archive`: each
  caption picks one of its characters, named in `info`)
- a free phrase (`coffee cup`): mapped to a tag when it is one, otherwise inserted verbatim (probably unknown to
  the model)
- an exclusion `-x` (`-smile`, `-white background`): no caption contains x. A character, series or count tag cannot
  be excluded; an exclusion of something you also require is ignored (`info` says so)
- `1boy` gives a boy subject (weak, see Limits). Artist and meta tags (`highres`, `absurdres`) are dropped.
  Several people (`2girls`, `multiple girls`) are rejected: the data is solo only.

### Note on bias

From real use: the captions often contain "white background" and lean towards Danbooru rating s (roughly
R-12 to R-15 content). Remove what you do not want with exclusions, for example
`-white background, -simple background`, or exclude suggestive elements by tag (`-cleavage`, `-swimsuit`, ...).
There is no rating filter; this is a note only.

### Autocomplete

Typing in the node's `tags` box opens a list of matching tags (only on this node). The token is the text between the
previous comma and the caret; `_` works like a space, and a leading `-` suggests exclusions (`-sm` -> `-smile`). Each
row shows the tag, its Danbooru category, how many training images have it (`train`) and its Danbooru post count
(`posts`). Up / Down choose, Enter or Tab accept, Esc closes; accepting writes `tag, `. Candidates are the tags,
characters and series the captions know, plus Danbooru names except artist and meta tags.

The route behind it is `GET /krea2capgen/suggest?q=<token>&k=<n>` (JSON `{status, items}`; `status` is `loading`
while the data loads).

## Example workflows

`example_workflows/` holds three text-to-image workflows. ComfyUI lists them in the template browser under this
node pack (or drag a file onto the canvas). In each one the node's `caption` feeds the positive prompt and `info`
goes to a text preview.

| file | model(s) | sampler |
|---|---|---|
| `krea2_anime_tuned_turbo.json` | Turbo | 8 steps, cfg 1.0, euler / simple, ModelSamplingFlux 1.15 / 1.15 (a fixed mu of 1.15) |
| `krea2_anime_tuned_raw.json` | Raw | 28 steps, cfg 5.5, euler / simple, empty negative, ModelSamplingFlux base_shift 0.5, max_shift 0.90625 (Krea 2's resolution-dependent mu) |
| `krea2_anime_tuned_hybrid.json` | Raw + Turbo | one 8-step schedule at mu 1.15: Raw for steps 0-2 (cfg 2.5), then Turbo for steps 2-8 (cfg 1.0) |

ComfyUI's cfg is 1 + Krea's guidance (Krea 4.5 = cfg 5.5). Files to put in `ComfyUI/models/`:

| folder | file | from |
|---|---|---|
| `diffusion_models/` | `krea2_anime_tuned_turbo_fp8_scaled.safetensors` | [ricecake/Krea-2-Anime-tuned-Turbo](https://huggingface.co/ricecake/Krea-2-Anime-tuned-Turbo) |
| `diffusion_models/` | `krea2_anime_tuned_raw_fp8_scaled.safetensors` | [ricecake/Krea-2-Anime-tuned-Raw](https://huggingface.co/ricecake/Krea-2-Anime-tuned-Raw) |
| `text_encoders/` | `qwen3vl_4b_fp8_scaled.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders) |
| `vae/` | `qwen_image_vae.safetensors` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2/tree/main/vae) |

The model repos are gated (accept the Krea 2 Community License on the model page first).

## Limits

- Solo, 1girl-focused data: boys come out weak (girl outfits and bodies may leak in), and several people are not
  supported.
- Tags and phrases never seen in training are not ignored: they are inserted into the caption verbatim and
  reported in `info` as "free phrase, inserted as written". The model probably does not know them.
- A source phrase's quirks are copied as they are (odd wording, quoted text).

## Development

The capgen code in `vendor/` comes from a capgen runtime export. It is loaded as the private package
`krea2_capgen_vendor`, so the generic names `capgen`, `tagdict` and `textrules` never enter `sys.modules`. The only
change to the exported code is that imports of its sibling modules become package-relative; `vendor/PIN.json` lists
every rewritten line. To update it:

```bash
python tools/vendor_runtime.py /path/to/capgen-runtime --revision <HF commit of ricecake/krea2-anime-capgen>
KREA2_CAPGEN_RUNTIME=/path/to/capgen-runtime python -m unittest discover -s tests
```

The tests check that the node's caption equals a direct capgen call on that runtime for fixed seeds, and that
unsafe file paths in the pin are refused (the version and every path are validated before vendoring and before a
download).

## License

MIT, see `LICENSE`. The models have their own license (Krea 2 Community License).
