"""Danbooru tag dictionary (name -> category, post_count) and the autocomplete index used by Generator.suggest_info.

data/danbooru-tags.parquet: public HF dataset ricecake/danbooru-tags-10k (100,000 most used Danbooru tags), columns
name (underscore form), category (0 general, 1 artist, 3 copyright, 4 character, 5 meta), post_count.
data/danbooru-tags.json (runtime format 2, export_runtime.py): the same rows in the same order, names already in the
parse() form: {"format": 1, "names": [...], "category": [...], "post_count": [...]}; read with the stdlib only.
"""
import bisect
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, 'data', 'danbooru-tags.json')
PARQUET_PATH = os.path.join(HERE, 'data', 'danbooru-tags.parquet')
DEFAULT_PATH = JSON_PATH if os.path.exists(JSON_PATH) else PARQUET_PATH
JSON_FORMAT = 1
CATEGORY = {0: 'general', 1: 'artist', 3: 'copyright', 4: 'character', 5: 'meta'}
DROP_CATEGORIES = {1: 'artist', 5: 'meta'}       # never written into a caption


def norm(s):
    """The parse() form of a tag: lowercase, '_' -> ' ', whitespace collapsed."""
    return re.sub(r'\s+', ' ', s.strip().lower().replace('_', ' '))


def norm_prefix(s):
    """Like norm() but keeps one trailing space ('cat ' does not match 'catgirl')."""
    t = re.sub(r'\s+', ' ', s.lower().replace('_', ' ')).lstrip()
    return t


class TagDict:
    def __init__(self, names, categories, post_counts, normalized=False):
        self.d = {}
        for n, c, p in zip(names, categories, post_counts):
            k = n if normalized else norm(n)
            if k and (k not in self.d or p > self.d[k][1]):
                self.d[k] = (c, p)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        """A .json dictionary (stdlib only) or the parquet (pyarrow)."""
        return cls(*columns(path), normalized=True)

    def __contains__(self, k):
        return k in self.d

    def __len__(self):
        return len(self.d)

    def get(self, k):
        return self.d.get(k)

    def category(self, k):
        """Category name ('general', 'artist', 'copyright', 'character', 'meta') or None if not in the dictionary."""
        v = self.d.get(k)
        return CATEGORY.get(v[0]) if v else None

    def post_count(self, k):
        v = self.d.get(k)
        return v[1] if v else 0

    def drop_reason(self, k):
        v = self.d.get(k)
        return DROP_CATEGORIES.get(v[0]) if v else None


def columns(path):
    """(normalized names, categories, post counts) in row order: a .json dictionary (stdlib) or the parquet
    (pyarrow). path: str or path-like."""
    path = os.fspath(path)
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        if d.get('format') != JSON_FORMAT:
            raise ValueError(f'{path}: tag dictionary format {d.get("format")!r}, expected {JSON_FORMAT}')
        return d['names'], d['category'], d['post_count']
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=['name', 'category', 'post_count'])
    # Danbooru names have no whitespace: lower + '_' -> ' ' is norm()
    names = pc.replace_substring(pc.utf8_lower(t.column('name')), '_', ' ').to_pylist()
    return names, t.column('category').to_pylist(), t.column('post_count').to_pylist()


def columns_to_json(names, categories, post_counts):
    """The danbooru-tags.json text (deterministic)."""
    return json.dumps({'format': JSON_FORMAT, 'names': names, 'category': categories, 'post_count': post_counts},
                      ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'


_WORD_START = re.compile(r'(?<=[^a-z0-9])[a-z0-9]')


class PrefixIndex:
    """Autocomplete over names: class 0 = the name starts with the prefix, class 1 = a later word starts with it
    ('ears' -> 'cat ears', 'strin' -> 'kanami (strinova)'). Within a class: by rank (the caller's global order).
    Sorted names + sorted word suffixes, bisect for the range, argpartition for the top k."""

    def __init__(self, names, rank):
        """names must be sorted; rank[i] = global order of names[i]."""
        self.names = names
        self.rank = np.asarray(rank, dtype=np.int64)
        suf, sid = [], []
        for j, n in enumerate(names):
            if ' ' in n or not n.isalnum():
                for m in _WORD_START.finditer(n):
                    suf.append(n[m.start():]); sid.append(j)
        order = sorted(range(len(suf)), key=suf.__getitem__)
        self.suf = [suf[i] for i in order]
        self.suf_id = np.asarray(sid, dtype=np.int64)[order]

    @staticmethod
    def _range(keys, p):
        lo = bisect.bisect_left(keys, p)
        return lo, bisect.bisect_left(keys, p + '\U0010ffff', lo)

    def _top(self, js, k):
        if js.size > k:
            js = js[np.argpartition(self.rank[js], k - 1)[:k]]
        return js[np.argsort(self.rank[js], kind='stable')]

    def query(self, p, k=20):
        """[(caller index, class)] best first."""
        if not p:
            return []
        lo0, hi0 = self._range(self.names, p)
        out = [(int(j), 0) for j in self._top(np.arange(lo0, hi0, dtype=np.int64), k)]
        if len(out) < k:
            lo, hi = self._range(self.suf, p)
            js = np.unique(self.suf_id[lo:hi])
            js = js[(js < lo0) | (js >= hi0)]                  # class 0 names are not repeated
            out += [(int(j), 1) for j in self._top(js, k - len(out))]
        return out
