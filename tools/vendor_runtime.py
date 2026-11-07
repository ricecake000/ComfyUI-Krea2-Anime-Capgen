"""Vendor a capgen runtime: copy its code into vendor/ and write vendor/PIN.json.

    python tools/vendor_runtime.py RUNTIME_DIR [--revision REV] [--repo ricecake/krea2-anime-capgen]

RUNTIME_DIR is a dir written by krea2-capgen's export_runtime.py (it has RUNTIME.json). The top-level .py files
listed in RUNTIME.json are checked against its sha256 and copied into vendor/ (old vendored .py files are removed
first), which the node loads as the private package krea2_capgen_vendor. The only change: imports of the vendored
modules by their plain names become package-relative (`import tagdict as TD` -> `from . import tagdict as TD`), so
no generic module name (capgen, tagdict, textrules) ever enters sys.modules; PIN.json lists every rewritten line.
The version and every path in RUNTIME.json are validated before any file is touched. Every other listed file is
data: PIN.json records its size and sha256 (from RUNTIME.json), so the node downloads exactly these
files from the Hugging Face repo at REV into <ComfyUI models>/capgen/<runtime version>/ on first use.

--revision is the HF commit that holds this runtime version. Without it the revision of the current PIN.json is
kept when the runtime version is unchanged; otherwise it becomes the placeholder (the node then refuses to download
and asks for a manual install).
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, 'vendor')
PIN = os.path.join(VENDOR, 'PIN.json')
MANIFEST = 'RUNTIME.json'
DEFAULT_REPO = 'ricecake/krea2-anime-capgen'
PLACEHOLDER = 'UNPUBLISHED'
MODULE_RX = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*\.py$')


def bridge():
    """capgen_bridge (no ComfyUI imports) for its path / version validation."""
    spec = importlib.util.spec_from_file_location('krea2capgen_bridge_tool', os.path.join(ROOT, 'capgen_bridge.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _vendored_imports(tree, mods):
    """[(line, end line, relative)] of every import statement (anywhere, nested too) that names a vendored module."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names = [a.name.split('.')[0] for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names = [(n.module or '').split('.')[0]] if n.level == 0 else []
        else:
            continue
        if any(x in mods for x in names):
            out.append((n.lineno, n.end_lineno, n))
    return out


def rewrite_imports(name, text, mods):
    """text of vendored module `name` with its imports of the other vendored modules made package-relative;
    returns (text, ['<name>:<line> <old> -> <new>', ...]). The import statements are found with ast (string contents
    are never touched); only one-line `import m [as x]` / `from m import ...` statements are rewritten, any other
    import of a vendored module name (several names, nested on one line with other code, ...) aborts vendoring, and
    so does any non-relative import of a vendored name left after the rewrite."""
    lines, log = text.split('\n'), []
    alt = '|'.join(map(re.escape, mods))
    rules = [(re.compile(rf'^(\s*)import ({alt})( as \w+)?(\s*(#.*)?)$'), r'\1from . import \2\3\4'),
             (re.compile(rf'^(\s*)from ({alt}) import (.+)$'), r'\1from .\2 import \3')]
    for ln, end, _ in _vendored_imports(ast.parse(text, name), mods):
        line = lines[ln - 1]
        rx_rp = next(((rx, rp) for rx, rp in rules if rx.match(line)), None) if ln == end else None
        if rx_rp is None:
            raise SystemExit(f'{name}:{ln}: import of a vendored module that cannot be rewritten: {line.strip()}')
        lines[ln - 1] = rx_rp[0].sub(rx_rp[1], line)
        log.append(f'{name}:{ln} {line.strip()} -> {lines[ln - 1].strip()}')
    out = '\n'.join(lines)
    left = _vendored_imports(ast.parse(out, name), mods)
    if left:
        raise SystemExit(f'{name}:{left[0][0]}: non-relative import of a vendored module left after the rewrite')
    return out, log


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('runtime_dir')
    ap.add_argument('--revision', help='HF commit of the runtime repo holding this version')
    ap.add_argument('--repo', default=DEFAULT_REPO)
    a = ap.parse_args()
    rt = os.path.abspath(a.runtime_dir)
    with open(os.path.join(rt, MANIFEST), encoding='utf-8') as f:
        man = json.load(f)
    files, version = man['files'], man['version']
    B = bridge()
    try:                                   # before any filesystem operation on these names
        B.check_version(version)
        B.check_repo(a.repo)
        for r in files:
            B.check_rel(r)
            B.inside(rt, r)
    except ValueError as e:
        sys.exit(f'{rt}/{MANIFEST}: {e}')
    code = sorted(r for r in files if '/' not in r and r.endswith('.py'))
    bad = [r for r in code if not MODULE_RX.match(r)]
    if bad:
        sys.exit(f'{rt}: not importable module names: {bad}')
    data = sorted(r for r in files if r not in code)
    if 'capgen.py' not in code or not data:
        sys.exit(f'{rt}: not a capgen runtime (no capgen.py or no data files in {MANIFEST})')
    if man.get('dirty'):
        print(f'warning: {version} was exported from a dirty tree', file=sys.stderr)
    # the code must be what RUNTIME.json lists (input pinning; the data files are only sized here)
    for r in code:
        if sha256_file(os.path.join(rt, r)) != files[r]['sha256']:
            sys.exit(f'{rt}/{r}: differs from {MANIFEST}')
    for r in data:
        if os.path.getsize(os.path.join(rt, r)) != files[r]['size']:
            sys.exit(f'{rt}/{r}: size differs from {MANIFEST}')
    old = {}
    if os.path.exists(PIN):
        with open(PIN, encoding='utf-8') as f:
            old = json.load(f)
    rev = a.revision or (old.get('revision') if old.get('runtime_version') == version and
                         old.get('hf_repo') == a.repo else None) or PLACEHOLDER
    mods = [r[:-3] for r in code]
    texts, rewrites = {}, {}
    for r in code:
        with open(os.path.join(rt, r), encoding='utf-8', newline='') as f:
            texts[r], rewrites[r] = rewrite_imports(r, f.read(), mods)
    os.makedirs(VENDOR, exist_ok=True)
    for n in os.listdir(VENDOR):
        if n.endswith('.py'):
            os.remove(os.path.join(VENDOR, n))
    shutil.rmtree(os.path.join(VENDOR, '__pycache__'), ignore_errors=True)
    for r in code:
        with open(os.path.join(VENDOR, r), 'w', encoding='utf-8', newline='') as f:
            f.write(texts[r])
    with open(os.path.join(VENDOR, '__init__.py'), 'w', encoding='utf-8') as f:
        f.write(f'"""Vendored capgen runtime {version} (tools/vendor_runtime.py; loaded as krea2_capgen_vendor)."""\n')
    pin = {'runtime_version': version, 'runtime_format': man.get('format'), 'commit': man.get('commit'),
           'hf_repo': a.repo, 'revision': rev,
           'code': {r: dict(files[r], import_rewrites=rewrites[r]) for r in code},
           'files': {r: files[r] for r in data}}
    tmp = PIN + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(pin, indent=1, sort_keys=True) + '\n')
    os.replace(tmp, PIN)
    mb = sum(files[r]['size'] for r in data) / 1e6
    print(f'vendored {version} (format {man.get("format")}): code {", ".join(code)}; '
          f'{len(data)} data files, {mb:.0f} MB; {a.repo}@{rev}')
    if rev == PLACEHOLDER:
        print(f'revision is the placeholder {PLACEHOLDER!r}: re-run with --revision <HF commit> once the runtime is '
              'uploaded', file=sys.stderr)
    print(f'next: KREA2_CAPGEN_RUNTIME={rt} python -m unittest discover -s tests')


if __name__ == '__main__':
    main()
