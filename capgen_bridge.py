"""capgen for ComfyUI without ComfyUI imports: runtime data lookup / download, the vendored code, one shared Generator.

Data: the files listed in vendor/PIN.json, in <dir>/<runtime version>/ for a dir of SEARCH_DIRS (set by nodes.py to
ComfyUI's "capgen" model folders), or in the dir named by $KREA2_CAPGEN_RUNTIME (a runtime dir with RUNTIME.json; no
lookup, no download); <dir>/<runtime version>.repair-*/ dirs are searched after <dir>/<runtime version>/. Missing
data is downloaded from the pinned Hugging Face revision into SEARCH_DIRS[0] (stdlib urllib, temp dir, then one atomic
rename to <version>, or to a fresh <version>.repair-* when an invalid <version> is in the way; a published dir is
never moved or deleted). $KREA2_CAPGEN_OFFLINE=1 or $HF_HUB_OFFLINE=1 forbids the download.

A capgen Generator is not thread-safe: every call goes through LOCK. Loading takes a few seconds (plus the download
on first use) and holds LOAD_LOCK; callers on an event loop run these functions in an executor.
"""
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, 'vendor')
PIN_PATH = os.path.join(VENDOR, 'PIN.json')
MANIFEST = 'RUNTIME.json'
PLACEHOLDER = 'UNPUBLISHED'
ENV_RUNTIME = 'KREA2_CAPGEN_RUNTIME'
ENV_OFFLINE = 'KREA2_CAPGEN_OFFLINE'
LENGTHS = ('long', 'medium', 'short')
SEARCH_DIRS = []            # <models>/capgen dirs; the first one receives downloads
LOCK = threading.Lock()     # around every Generator call
LOAD_LOCK = threading.Lock()
RETRY_AFTER = 60.0          # seconds before a background load retries after a failure

log = logging.getLogger('krea2capgen')
_gen = None
_load_error = None          # (time, message) of the last failed load
_bg = None                  # background load thread


VERSION_RX = re.compile(r'^[0-9a-f]{12}-(dirty-)?[0-9a-f]{8}$')
REPO_RX = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$')
VENDOR_PKG = 'krea2_capgen_vendor'       # the vendored code's private package name (never capgen / tagdict / ...)


def check_version(v):
    if not isinstance(v, str) or not VERSION_RX.match(v):
        raise ValueError(f'bad runtime version {v!r}')


def check_repo(r):
    if not isinstance(r, str) or not REPO_RX.match(r) or '..' in r:
        raise ValueError(f'bad Hugging Face repo id {r!r}')


def check_rel(rel):
    """A runtime file path must be relative, '/'-separated, without '.', '..', empty parts, backslashes or drives."""
    if not isinstance(rel, str) or not rel or rel.startswith('/') or '\\' in rel or ':' in rel or '\0' in rel or \
            any(p in ('', '.', '..') for p in rel.split('/')):
        raise ValueError(f'bad runtime file path {rel!r}')


def inside(base, rel, resolve=True):
    """base/rel (rel passed check_rel), which must stay inside base after resolving symlinks (resolve=True: where we
    write). resolve=False only joins: reading a user's install may follow symlinked files (e.g. into a download
    cache)."""
    check_rel(rel)
    p = os.path.join(base, *rel.split('/'))
    if not resolve:
        return p
    b = os.path.realpath(base)
    if not os.path.realpath(p).startswith(b + os.sep):
        raise ValueError(f'{rel!r} leaves {base}')
    return p


def load_pin():
    """vendor/PIN.json, validated (version, repo, every file path) before anything uses it."""
    with open(PIN_PATH, encoding='utf-8') as f:
        pin = json.load(f)
    check_version(pin['runtime_version'])
    check_repo(pin['hf_repo'])
    if not isinstance(pin['revision'], str) or not pin['revision'] or '/' in pin['revision']:
        raise ValueError(f'bad revision {pin["revision"]!r}')
    for rel, v in pin['files'].items():
        check_rel(rel)
        if not isinstance(v.get('size'), int) or v['size'] < 0:
            raise ValueError(f'{rel}: bad size')
    return pin


def check_dir(d, pin):
    """Problems of runtime dir d (empty = usable): RUNTIME.json names the pinned version, every pinned file is there
    with its pinned size."""
    try:
        with open(os.path.join(d, MANIFEST), encoding='utf-8') as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        return [f'{MANIFEST}: {e}']
    if not isinstance(man, dict):
        return [f'{MANIFEST}: not a JSON object']
    if man.get('version') != pin['runtime_version']:
        return [f'{MANIFEST}: version {man.get("version")!r}, this node needs {pin["runtime_version"]!r}']
    bad = []
    for rel, v in pin['files'].items():
        try:
            p = inside(d, rel, resolve=False)
        except ValueError as e:
            bad.append(str(e)); continue
        if not os.path.isfile(p):
            bad.append(f'{rel}: missing')
        elif os.path.getsize(p) != v['size']:
            bad.append(f'{rel}: {os.path.getsize(p)} bytes, expected {v["size"]}')
    return bad


def _offline():
    return any(os.environ.get(k, '').strip().lower() in ('1', 'true', 'yes', 'on') for k in (ENV_OFFLINE, 'HF_HUB_OFFLINE'))


def _manual_help(pin, target):
    return (f'Manual install: download {MANIFEST} and {", ".join(sorted(pin["files"]))} from '
            f'https://huggingface.co/{pin["hf_repo"]} (revision {pin["revision"]}) into {target}, keeping the '
            f'sub-folders, or set {ENV_RUNTIME} to a capgen runtime dir of version {pin["runtime_version"]}.')


def _fetch(url, dest, size=None, sha256=None, tries=3):
    """Download url to dest; check the size and sha256 (both from the pin) when given."""
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ComfyUI-Krea2-Anime-Capgen'})
            h, n, last = hashlib.sha256(), 0, time.time()
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); h.update(b); n += len(b)
                    if size and time.time() - last > 10:
                        log.info('krea2capgen: %s %.0f / %.0f MB', os.path.basename(dest), n / 1e6, size / 1e6)
                        last = time.time()
            if size is not None and n != size:
                raise IOError(f'{n} bytes, expected {size}')
            if sha256 is not None and h.hexdigest() != sha256:
                raise IOError('sha256 differs from the pinned one')
            return
        except Exception as e:
            if attempt == tries:
                raise RuntimeError(f'download {url}: {e}') from e
            log.warning('krea2capgen: download %s failed (%s), retrying', url, e)
            time.sleep(2 * attempt)


def candidates(root, version):
    """Dirs that may hold the pinned data, in lookup order: root/<version>, then root/<version>.repair-* by name."""
    try:
        reps = sorted(n for n in os.listdir(root) if n.startswith(version + '.repair-'))
    except OSError:
        reps = []
    return [os.path.join(root, version)] + [os.path.join(root, n) for n in reps]


def find_valid(root, pin):
    """The first valid candidate dir under root, else None."""
    for d in candidates(root, pin['runtime_version']):
        if os.path.isdir(d) and not check_dir(d, pin):
            return d
    return None


def publish(tmp, root, pin):
    """Publish the checked staging dir tmp under root and return the published dir. No lock (works on Windows), and an
    existing dir is never renamed, moved or deleted: tmp is renamed to root/<version> if that does not exist, else to
    a fresh root/<version>.repair-<UTC>-<pid>-<hex>. If another process published a valid copy meanwhile, it is used
    and tmp is removed."""
    final = os.path.join(root, pin['runtime_version'])
    if not os.path.lexists(final):
        try:
            os.rename(tmp, final)       # fails if final appeared meanwhile (another process): never replaces it
            return final
        except OSError as e:
            log.info('krea2capgen: publishing %s failed (%s), re-checking', final, e)
    d = find_valid(root, pin)
    if d is not None:
        shutil.rmtree(tmp, ignore_errors=True)
        return d
    rep = (f'{final}.repair-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}-'
           f'{secrets.token_hex(4)}')
    os.rename(tmp, rep)
    return rep


def download(root, pin):
    """The pinned runtime data -> root/<version>/ or root/<version>.repair-*/ (temp dir, then publish()). Returns the
    published dir."""
    final = os.path.join(root, pin['runtime_version'])
    if pin['revision'] == PLACEHOLDER:
        raise RuntimeError('the capgen data is not published yet (no pinned revision). ' + _manual_help(pin, final))
    if _offline():
        raise RuntimeError(f'capgen data missing in {final} and downloads are off ({ENV_OFFLINE} / HF_HUB_OFFLINE). '
                           + _manual_help(pin, final))
    os.makedirs(root, exist_ok=True)
    need = sum(v['size'] for v in pin['files'].values())
    free = shutil.disk_usage(root).free
    if free < need + (64 << 20):
        raise RuntimeError(f'{root}: {free / 1e6:.0f} MB free, the capgen data needs {need / 1e6:.0f} MB')
    base = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
    url = f'{base}/{pin["hf_repo"]}/resolve/{urllib.parse.quote(pin["revision"], safe="")}/'
    tmp = tempfile.mkdtemp(prefix=f'.{pin["runtime_version"]}.download-', dir=root)
    log.info('krea2capgen: downloading the capgen data (%.0f MB) from %s@%s to %s', need / 1e6, pin['hf_repo'],
             pin['revision'], final)
    try:
        _fetch(url + MANIFEST, os.path.join(tmp, MANIFEST))
        with open(os.path.join(tmp, MANIFEST), encoding='utf-8') as f:
            man = json.load(f)
        if not isinstance(man, dict) or man.get('version') != pin['runtime_version']:
            raise RuntimeError(f'{pin["hf_repo"]}@{pin["revision"]} holds capgen {man.get("version") if isinstance(man, dict) else man!r}, '
                               f'this node needs {pin["runtime_version"]!r}')
        for rel, v in sorted(pin['files'].items()):
            dest = inside(tmp, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _fetch(url + urllib.parse.quote(rel), dest, v['size'], v.get('sha256'))
        bad = check_dir(tmp, pin)
        if bad:
            raise RuntimeError('downloaded capgen data: ' + '; '.join(bad))
        final = publish(tmp, root, pin)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log.info('krea2capgen: capgen data ready in %s', final)
    return final


def runtime_dir(pin, allow_download=True):
    """The dir holding the pinned data: $KREA2_CAPGEN_RUNTIME, else <search dir>/<version>, else a download."""
    env = os.environ.get(ENV_RUNTIME)
    if env:
        bad = check_dir(env, pin)
        if bad:
            raise RuntimeError(f'{ENV_RUNTIME}={env}: ' + '; '.join(bad))
        return env
    if not SEARCH_DIRS:
        raise RuntimeError(f'no capgen model folder configured and {ENV_RUNTIME} is not set')
    for root in SEARCH_DIRS:
        main = os.path.join(root, pin['runtime_version'])
        bad = check_dir(main, pin) if os.path.lexists(main) else None
        if bad:
            log.warning('krea2capgen: %s is incomplete or invalid (%s); it is not used and you may delete it',
                        main, '; '.join(bad[:3]) + (' ...' if len(bad) > 3 else ''))
        d = find_valid(root, pin)
        if d is not None:
            return d
    if not allow_download:
        raise RuntimeError('capgen data not found. ' + _manual_help(pin, os.path.join(SEARCH_DIRS[0],
                                                                                       pin['runtime_version'])))
    return download(SEARCH_DIRS[0], pin)


def import_capgen():
    """The vendored capgen module as krea2_capgen_vendor.capgen: vendor/ is loaded as that private package (its
    imports of its siblings are package-relative, see tools/vendor_runtime.py); sys.path is not touched."""
    pkg = sys.modules.get(VENDOR_PKG)
    if pkg is None:
        spec = importlib.util.spec_from_file_location(VENDOR_PKG, os.path.join(VENDOR, '__init__.py'),
                                                      submodule_search_locations=[VENDOR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[VENDOR_PKG] = pkg
        try:
            spec.loader.exec_module(pkg)
        except BaseException:
            sys.modules.pop(VENDOR_PKG, None)
            raise
    if list(pkg.__path__) != [VENDOR]:
        raise ImportError(f'{VENDOR_PKG} is loaded from {list(pkg.__path__)}, expected {VENDOR}')
    return importlib.import_module(VENDOR_PKG + '.capgen')


def load_generator(d, pin):
    """Generator over runtime dir d: the model, the tag dictionary and char_series.json named by the pin; never the
    training metadata."""
    capgen = import_capgen()
    files = pin['files']
    tags = [r for r in files if r.startswith('data/danbooru-tags')]
    if len(tags) != 1:
        raise RuntimeError(f'expected one data/danbooru-tags* file in the pin, found {tags}')
    params = inspect.signature(capgen.Generator.load).parameters
    kw = {'tags_path': os.path.join(d, *tags[0].split('/'))}
    if 'metadata_path' in params:
        kw['metadata_path'] = None
    if 'char_series_path' in params:
        kw['char_series_path'] = os.path.join(d, 'char_series.json') if 'char_series.json' in files else None
    t0 = time.time()
    g = capgen.Generator.load(os.path.join(d, 'model'), **kw)
    log.info('krea2capgen: capgen %s loaded from %s in %.1fs', pin['runtime_version'], d, time.time() - t0)
    return g


def get_generator():
    """The shared Generator (loaded, and downloaded, on first use)."""
    global _gen, _load_error
    if _gen is not None:
        return _gen
    with LOAD_LOCK:
        if _gen is None:
            try:
                pin = load_pin()
                _gen = load_generator(runtime_dir(pin), pin)
                _load_error = None
            except Exception as e:
                _load_error = (time.time(), f'{type(e).__name__}: {e}')
                raise
    return _gen


def status():
    """'ready', 'loading' (a background load runs) or 'error: ...' (the last load failed less than RETRY_AFTER s ago);
    otherwise starts a background load and returns 'loading'."""
    global _bg
    if _gen is not None:
        return 'ready'
    if _bg is not None and _bg.is_alive():
        return 'loading'
    if _load_error is not None and time.time() - _load_error[0] < RETRY_AFTER:
        return 'error: ' + _load_error[1]

    def run():
        try:
            get_generator()
        except Exception as e:
            log.error('krea2capgen: loading capgen failed: %s', e)
    _bg = threading.Thread(target=run, name='krea2capgen-load', daemon=True)
    _bg.start()
    return 'loading'


def suggest(q, k=12):
    """Autocomplete rows (suggest_info) for the token q; needs a loaded Generator (see status())."""
    g = get_generator()
    with LOCK:
        return g.suggest_info(q, k)


def split_tags(tags):
    """The node's tags text -> constraint list: commas and line breaks separate tags."""
    return [t for t in re.split(r'[,\n]', tags) if t.strip()]


_KIND = {'implicit': 'implicit (every caption is solo 1girl)', 'count': 'count tag (the subject becomes a boy)',
         'subject': 'subject phrase', 'raw': 'free phrase, inserted as written (not seen in training: probably unknown '
         'to the model)'}


def describe_line(d, picked=None):
    """One line of the info text for a describe() row."""
    k, tag, cat = d['kind'], d.get('tag'), d.get('category')
    cat = f' [{cat}]' if cat else ''
    if k == 'tag':
        what = f'tag "{tag}"{cat}'
    elif k == 'char':
        what = f'character "{d["char"]}"{cat}'
    elif k == 'series':
        what = f'series "{tag}"{cat}: one of its characters' + (f' (this caption: {picked})' if picked else '')
    elif k == 'dropped':
        what = f'dropped: {d.get("reason")} tag, not written into captions'
    elif k == 'rejected':
        what = 'rejected: ' + ('the data is solo only' if d.get('reason') == 'solo only' else
                               f'{d.get("reason")} (the requirement wins)')
    elif k == 'exclude':
        what = f'excluded: tag "{tag}" and its words' if tag else 'excluded: these words'
    elif k == 'exclude_unsupported':
        what = f'exclusion ignored: {d.get("reason")}'
    elif k == 'tag_unsupported':
        what = f'tag "{tag}"{cat}: no phrases of its own (searched in the captions, else written as tag text)'
    elif k == 'implicit' and d.get('reason') == 'original':
        what = 'implicit (an original character: a generic subject)'
    else:
        what = _KIND.get(k, k) + cat
    return f'  {d["text"]} -> {what}'


def format_info(described, s):
    """The info output: how each tag was understood, the sample's unrealized / dropped / rejected tags, warnings."""
    out = ['tags:'] + ([describe_line(d, s.picked_character) for d in described] or ['  (none)'])
    if s.picked_character and not any(d['kind'] == 'series' for d in described):
        out.append(f'character: {s.picked_character}')
    if s.unrealized:
        out.append('not expressed (or only as tag text): ' + ', '.join(s.unrealized))
    if s.rejected:
        out.append('rejected: ' + ', '.join(s.rejected))
    if s.dropped:
        out.append('dropped: ' + ', '.join(f'{x["text"]} ({x["reason"]})' for x in s.dropped))
    if s.warnings:
        out += ['warnings:'] + [f'  - {w}' for w in s.warnings]
    out.append(f'~{s.approx_tokens} tokens')
    return '\n'.join(out)


def caption(tags, length='long', seed=0, wild=1.0, overlay_text=False, scene_text=True, mix_look=True):
    """(caption, info) for the node: one capgen sample of the tags with this seed and these options."""
    if length not in LENGTHS:
        raise ValueError(f'length must be one of {LENGTHS}')
    g = get_generator()
    cons = split_tags(tags)
    with LOCK:
        res = g.generate(cons, n=1, length=length, seed=int(seed), wild=float(wild), overlay_text=bool(overlay_text),
                         scene_text=bool(scene_text), mix_look=bool(mix_look))
        described = g.describe(cons)
    if not res:
        raise RuntimeError('capgen returned no caption for these tags')
    return res[0].text, format_info(described, res[0])
