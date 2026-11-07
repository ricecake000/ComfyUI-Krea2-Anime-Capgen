"""krea2-capgen: tag-constrained, diverse caption generator (no LLM) in the style of the Krea 2 FT captions.

Library:
    g = Generator.load('model')                    # + data/danbooru-tags.{json,parquet} (tag dictionary) when present
    for s in g.generate(['beach', 'bikini', 'cat ears'], n=5, length='long', seed=1, explain=True):
        print(s.text); print(s.explain_text())
    g.suggest_info('ears'); g.describe(['highres', 'blue archive']); s.to_dict()

A Generator is NOT thread-safe: generate() keeps per-call scratch state on the instance (_cur_cons, caches).
Callers that share one instance across threads must hold a lock around generate() / describe().

CLI:
    python capgen.py "beach, bikini, cat ears" -n 5 --length long --seed 1 --explain
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from . import tagdict as TD
from . import textrules as T

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS_DEFAULT = TD.DEFAULT_PATH
# training metadata (character_name / series_name only): the v2 + v3 union of build_union.py
METADATA_DEFAULT = os.path.join(HERE, 'data', 'metadata_v2v3.parquet')
# the character -> series map exported from the metadata (export_runtime.py; a runtime dir without metadata)
CHAR_SERIES_DEFAULT = os.path.join(HERE, 'char_series.json')
CHAR_SERIES_FORMAT = 1
# the model's metadata: model/meta.json (runtime format 2, stdlib json) or model/meta.pkl (written by build_model.py)
META_FORMAT = 1
# meta.json forms of a dict with int keys and of a tuple: {"__int_keys__": [[key, value], ...]}, {"__tuple__": [...]}
_META_TAGS = {'__int_keys__': lambda kv: {k: x for k, x in kv}, '__tuple__': tuple}
_DEFAULT = object()        # load(): the module default, read at call time
SERIES_MIN_SHARE = 0.5     # char c belongs to series s if >= this share of c's rows carry s
BOY_SUBJECT = 'anime illustration of a boy'      # the subject of a 1boy caption without a subject / character
ORIGINAL = 'original'      # Danbooru's "not from a franchise" copyright tag: a generic subject (implicit)
GROUPS = (T.APPEAR, T.OUTFIT, T.ACTION, T.PLACE, T.TAIL)
NG = T.NG
EXCLUSIVE = {'hair', 'eyes', 'shot', 'light'}      # one source per caption for these facets
COMPAT_SCALE = 4.0      # compat (mean clipped PMI, nats) is multiplied by this before exp(beta_eff * .)
# wild in [0, 1] (default 1): beta_eff = (1 - wild) * beta; conflict rejection and ACTION/PLACE coupling only
# when wild < 0.5. wild = 1 draws donors uniformly among the constraint-satisfying candidates.
K_CAND = 256            # donor candidates scored per group
K_PATCH = 64            # patch phrases scored per missing constraint
P_CHAR_OUTFIT = 0.8     # outfit donor from the same character (costume)
# phrase order (order='template'): the facet sequence of a real donor caption is the template. ORDER_PICK chooses
# it: 'random' (any donor row), 'donor' (the sample's donor with the most phrases), 'best' (best facet-multiset
# overlap among K_TEMPLATE random rows), 'mix' (best among the sample's donors + K_TEMPLATE random rows)
ORDER_PICK = 'mix'
K_TEMPLATE = 128
DONOR_BONUS = 0.06       # 'mix': overlap bonus of the sample's own donors (their phrases keep their slots)
TAIL_KEY = len(T.PRIMARY_ORDER)     # template key of a TAIL-group phrase (the style tail is one block)
NK = TAIL_KEY + 1
GARMENT_BITS = T.FBIT['outfit'] | T.FBIT['exposure'] | T.FBIT['hairacc']
BODY_PART_RX = re.compile(r'\b(ears?|tails?|horns?|wings?|halo)\b')
# phrases that refer back to the phrase before them, anywhere in the phrase ('resting another hand on ...',
# 'holding a cup in her other hand', 'paired with a pleated skirt', 'matching socks'); chosen on the donor captions
# (rarely the first phrase of their group, more often after a phrase of the same group). Not 'both', 'its', 'it',
# 'them', 'each', 'second', 'same': self-contained ('with both hands') or an antecedent inside the phrase.
_DEPENDENT_RX = re.compile(r'\b(matching|another|(?<!each )other|paired with|opposite|also|likewise|similarly|'
                           r'the rest of)\b')
FEMALE_RX = re.compile(r'\b(breasts?|cleavage|underboob|sideboob|bra|lingerie|panties)\b')
# mixed appearance (mix_look): the APPEARANCE group of a generic caption comes from three donor rows, one per slot.
# MIXED = a phrase that describes two slots ('brown hair with droopy fox ears', 'long blonde hair and blue eyes'):
# never used when the look is mixed (another slot's donor describes that part)
HAIR, EYES, BODY, MIXED, MIXED_BODY = 0, 1, 2, 3, 4      # MIXED: hair + eye colour; MIXED_BODY: hair + body trait
SLOT_NAMES = ('hair', 'eyes', 'body')
K_TAIL_FALLBACK = 64       # rows tried for a mark-free style tail when the tail donor's style phrases were all marks
_HAIR_WORD_RX = T.FACET_RX['hair']
_BODY_TRAIT_RX = re.compile(r'\b(horns?|horned|wings?|winged|halo|antlers?)\b')
_EYE_OVER_RX = re.compile(r'\b(covering|over|between|hiding|across|above|framing|past|into|obscuring|shading|in front '
                          r'of|partially covering|falling over|reaching) (one |her |his |both |the |an )?'
                          r'(left |right )?eyes?\b')
_LINER_RX = re.compile(r'\bwinged (eye)?liner\b|\bwinged eye makeup\b|\bwinged (eye)?shadow\b')
_EYE_WORD_RX = re.compile(r'\beyes?\b|\bpupils?\b|\birises\b|\biris\b|heterochromia')
_COLOR_RX = re.compile(r'\b(' + T.COLOR_WORDS + r')\b')
# an eye colour: a colour word up to three words before the eye word ('large luminous blue eyes') or up to four after
# it ('eyes shifting from blue to lavender'; not 'eyes with red eyeliner'), heterochromia, multi-colored eyes
_EYE_COLOUR_RX = re.compile(r'\b(' + T.COLOR_WORDS + r'|hazel|dark|multi-colou?red)\b([\w-]* ){0,3}?[\w-]*\b(eyes?|'
                            r'irises|iris|pupils?)\b|\b(eyes?|irises|pupils?) ([\w-]+ ){0,4}?(' + T.COLOR_WORDS +
                            r')\b(?![- ]?(eye)?(liner|shadow|lashes|makeup|mascara|tears?)|[- ](hair|bangs|glasses|'
                            r'frames|blindfold|mask|ribbon|cloth|fabric))|heterochrom')
# eye words that do not describe the eyes ('white lattice eye mask', 'red flowers covering right eye', a mole)
_EYE_NOT_DESC_RX = re.compile(r'\beye ?(mask|patch)\w*|\b(covering|over|hiding|under|below|beside|near|between|framing|'
                              r'past|across|above|into) (her |the |one |both )?(left |right |other )?eyes?\b|\bmole\b|'
                              r'\bone eye\b|\beyes? (covered|hidden) by\b|\beye[- ](level|contact|catching)\b|'
                              r'\bbird.?s[- ]eye\b')
# hair words that do not describe the hair: an action on it ('one hand running through her hair'), an accessory
_HAIR_NOT_DESC_RX = re.compile(r'\b(running|touching|playing|adjusting|brushing|combing|fixing|tucking|holding|pulling|'
                               r'twirling|stroking|grabbing|tying|ruffling|fiddling|messing|in|on|through|from|into|to|'
                               r'behind|over|under|below|above|beside|near|around|against|at) (with )?(her |the |his )?'
                               r'(own )?(hair|bangs)\b|'
                               r'\bhair ?(ribbons?|clips?|pins?|ornaments?|bows?|bands?|ties?|flowers?|accessor\w*|'
                               r'sticks?|beads?|cuffs?|rings?|scrunchies?|bobbles?)\b|\bombre\b')     # 'ombre forearm'


def names_part(text, slot, male=False):
    """A HAIR / EYES phrase that describes the part itself (not only an action on it, an accessory, an eye mask, a
    phrase that refers back, a phrase about another figure: 'beside a dark horse with a pinkish mane'; male: the
    subject is a boy, 'his' is his)."""
    if is_dependent(text) or re.match(r'(wearing|dressed|clad)\b', text) or _OTHER_PRON_RX[male].search(text) or \
            (any(not _MALE_NOUN_RX.search(x.group(0)) for x in _OTHER_FIG_RX.finditer(text)) if male else
             _OTHER_FIG_RX.search(text)):
        return False
    if slot == HAIR:
        return bool(_HAIR_WORD_RX.search(_HAIR_NOT_DESC_RX.sub(' ', text)))
    return bool(_EYE_WORD_RX.search(_EYE_NOT_DESC_RX.sub(' ', text))) and not EYES_HIDDEN_RX.search(text)


_TRAIT_RX = {'wing': re.compile(r'\b(wings?|winged)\b'), 'horn': re.compile(r'\b(horns?|horned|antlers?)\b'),
             'halo': re.compile(r'\bhalos?\b'),
             'ear': re.compile(r'\b(pointed|pointy|elf|elven|elfin|animal|fluffy|furry)( [\w-]+)? ears?\b'),
             'tail': re.compile(r'\btails?\b')}
_NOT_TRAIT_RX = re.compile(r'\b(twin|pony|pig|side|side-up|side up|low|high|two|double|triple|quad|drill|braided) '
                           r'?tails?\b|\b(coat|ruffled|frilled|ribbon|shirt) ?tails?\b|'
                           r'\btails? of\b|\bwing(ed)? (eye)?liner\b|\bwing-shaped\b|\bwing hair\b|'
                           r'\b[\w]+-tails? (ponytails?|braids?|hair)\b')      # 'low dolphin-tail ponytail'
# a trait word that names an accessory, not the girl's own trait: 'cat ear headband', 'bunny-ear hood', 'horned helmet',
# 'headband with cat ears', 'wing-shaped hair clips' (an accessory word elsewhere, 'halo with bird motifs', does not)
_ACC = (r'(headband|hairband|hood|hoodie|hat|cap|beanie|beret|helmet|headphones|headset|earmuffs|clips?|hairclips?|'
        r'hairpins?|ornaments?|earrings?|piercings?|mask|costume|kigurumi|onesie|pajamas|plush\w*|keychain|pins?|'
        r'backpack|bag|purse|print|printed|pattern|motifs?|charm|pendant|necklace|brooch|badge|sticker|tattoo|decal|'
        r'accessor\w*|cushion|pillow|toy|doll|logo|emblem|graphic|design|slippers|gloves|mittens|socks|stockings)')
_TRAIT_WORD = r'(ears?|eared|tails?|horns?|horned|wings?|winged|halos?|antlers?)'
_ACCESSORY_RX = re.compile(r'\b' + _TRAIT_WORD + r'[- ]([\w-]+ )?' + _ACC + r'\b|\b' + _ACC + r' (with|featuring|'
                           r'bearing) ([\w-]+ ){0,3}' + _TRAIT_WORD + r'\b|-shaped\b|\b(shaped|patterned|printed|'
                           r'embroidered) (like )?([\w-]+ ){0,2}' + _TRAIT_WORD + r'\b')


_GENERIC_TRAIT = {'animal ears': 'ear'}       # a tag that any trait of its kind expresses (mixed looks)
# cues in a caption that explain a missing hair / eye description (mixed looks), chosen on the donor rows: the share of
# rows with the cue that write an eye colour (all rows 96.4 %): 'closed eyes' 5.5 %, 'smiling / happy / curved eyes'
# 1.9 / 6.5 / 2.9 %, 'eyes closed' 21 %, 'closed-eye(s / d)' 0-14 %, 'blindfold' 7.1 %, 'asleep' 27 %; not
# 'half-closed', 'one eye closed' (94 % describe the eyes), 'from behind' (92 %), 'silhouette' (96 %), 'facing away'
# (56 %). Hair: no cue of the data explains it (hood up 98-100 %, headscarf / wimple / coif 100 %, 'hair tucked behind
# her ear' 100 %): only explicit wording ('hair hidden under ...', 'shaved head', 'head out of frame') hides the hair
HAIR_HIDDEN_RX = re.compile(r'\bhair (is )?(completely |fully |entirely )?(hidden|concealed|not visible|out of frame|'
                            r'(covered|tucked) (under|beneath|inside|into|by) (a|an|the|her))\b|\b(bald|shaved|shaven) '
                            r'head\b|\b(head|face and hair) (is )?(out of|outside the) frame\b|\bhead (cropped|cut '
                            r'off) (out|by the frame)\b')
EYES_HIDDEN_RX = re.compile(r'(?<!half-)(?<!half )(?<!stylized )(?<!printed )\b(closed|shut|squeezed|squeezed-shut|'
                            r'tightly closed|smiling|happy|curved|crescent|crescent-shaped) eyes\b|\beyes ((are|were|'
                            r'being|kept|remain|remaining|stay|staying|still) )?((gently|softly|tightly|firmly|'
                            r'peacefully|calmly|serenely|quietly|blissfully|happily|fully|completely|both) )?(closed|'
                            r'shut|squeezed shut)\b|\bclosed-eye[sd]?\b|\bclosed eye (happy )?(smile|smiling|'
                            r'grin|expression)\b|(?<!half )\basleep\b|\bblindfold(ed)?\b'
                            r'(?! in (her|one) )(?![\w -]*\b(one|right|left|other) eye\b)|\beyes (are )?(out of '
                            r'frame|hidden|not visible|fully covered)\b|\b(both|her) eyes (covered|hidden)\b')
# a cue about another figure is no cue: 'holding a cat with closed eyes', 'a man with a shaved head', 'his eyes closed'.
# An animal word that names her trait ('cat ears', 'fox girl') is no figure; 'her eyes closed' is always about her
_OTHER_FIG_RX = re.compile(
    r'\b(cats?|kittens?|kitty|dogs?|pupp(y|ies)|fox(es)?|rabbits?|bunn(y|ies)|birds?|owls?|chicks?|creatures?|animals?|'
    r'pets?|beasts?|monsters?|dragons?|wolf|wolves|bears?|teddy|hamsters?|mouse|mice|squirrels?|ferrets?|fish|'
    r'penguins?|ducks?|frogs?|seals?|horses?|deer|cows?|sheep|lambs?|pigs?|snakes?|cubs?|man|men|boys?|guys?|woman|'
    r'women|lady|person|people|child|children|kids?|bab(y|ies)|infant|mother|father|sister|brother|friends?|'
    r'companions?|partner|(another|other|second|older|younger|little) girls?|dolls?|plush\w*|stuffed|puppets?|'
    r'mannequins?|statues?|'
    r'figur(e|es|ine|ines)|mascots?|characters?|skeletons?|skulls?|corpses?|zomb(ie|ies)|ghosts?|spirits?|angels?|'
    r'demons?|knights?|soldiers?|robots?|seagulls?|gulls?|parrots?|crows?|doves?|pigeons?|tanuki|raccoons?|otters?|'
    r'bats?|persona|reflection|apparition|clone|chibi|version|inset|bubble|poster|photo\w*|portrait|picture|'
    r'screen|monitor)\b(?![- ](ears?|eared|tails?|girl|print|printed|hood|hoodie|shaped|motifs?|'
    r'patterns?|ornaments?|clips?|pins?|costume|onesie|pajamas|slippers|headband|hat|mask|horns?|wings?|eyes|pupils))')
# pronouns of another figure, of the main figure right before a cue, of a clause about her / him -- by the subject's
# gender (male: a 1boy caption; donor phrases still say 'her', rewritten to 'his' at the end, so 'her' is him too)
_OTHER_PRON_RX = {False: re.compile(r'\b(it|its|he|his|him|they|their|them|who|whose|which)\b'),   # 'sylveon who ...'
                  True: re.compile(r'\b(it|its|she|they|their|them|who|whose|which)\b')}
# 'with her eyes closed'; not 'a cat beside her with eyes closed'
_SELF_RX = {False: re.compile(r'\bher\s+(own\s+)?$'),
            True: re.compile(r'\b(his|her)\s+(own\s+)?$')}
_SELF_CLAUSE_RX = {False: re.compile(r'\b(while|as|and|before|after|then|when|until) (she|her)\b'),
                   True: re.compile(r'\b(while|as|and|before|after|then|when|until) (he|his|him|her)\b')}
_MALE_NOUN_RX = re.compile(r'\b(man|men|boys?|guys?)$')       # a male subject's own noun ('boy sleeping ...')
# a bare person noun after a verb / preposition is another figure ('holding a girl with closed eyes'); 'a close-up of a
# girl' and a phrase that starts with 'a girl' are her
_OBJ_PERSON_RX = re.compile(r'\b(?!of\b)[a-z]+ (a|an|the|one|two|another|other|second|small|little|young|older|'
                            r'younger) ([\w-]+ )?(girls?|woman|women|boys?|child|children|lady|kids?)\b')
_OBJ_RX = re.compile(r'\b(\w+ing|with|on|in|at|beside|behind|near|by|to|from|under|over|against|and|holds?|hugs?)\b')
_ING_CLAUSE_RX = re.compile(r'\b(while|as|and|before|after|then|when|until) \w+ing\b')


def main_cue(text, rx, male=False):
    """A cue (HAIR_HIDDEN_RX / EYES_HIDDEN_RX match) about the main figure (male: a boy; his / he / her are his, she
    is another figure; else her / she are hers, he / his / him another's). Her own possessive right before the cue
    wins. Else the last other-figure evidence before it in the phrase (a figure noun, a bare person noun after a
    verb, 'it / they / its / their / who' and the other gender's pronouns) takes the cue, unless a clause about her
    follows it: 'while she ...', or a participle clause when the figure is an object ('holding a cat while sleeping
    with closed eyes'; not 'a cat purring and sleeping with closed eyes', not 'holding a cat while it is asleep').
    Ambiguous -> not a cue (the part stays)."""
    for m in rx.finditer(text):
        pre = text[:m.start()]
        if _SELF_RX[male].search(pre):
            return True
        ev = [(x.start(), x.end(), 'pron') for x in _OTHER_PRON_RX[male].finditer(pre)] + \
            [(x.start(), x.end(), 'noun') for x in _OTHER_FIG_RX.finditer(pre)
             if not (male and _MALE_NOUN_RX.search(x.group(0)))] + \
            [(x.start(), x.end(), 'obj') for x in _OBJ_PERSON_RX.finditer(pre)]
        if not ev:
            return True
        start, end, kind = max(ev, key=lambda e: e[1])
        after = pre[end:]
        obj = kind == 'obj' or (kind == 'noun' and _OBJ_RX.search(pre[:start]))
        if _SELF_CLAUSE_RX[male].search(after) or (obj and _ING_CLAUSE_RX.search(after)):
            return True
    return False


def trait_kinds(text):
    """Body-trait kinds a phrase describes: wing / horn / halo / ear / tail ('fox ears', 'small white angel wings',
    'pointed ears'); not hairstyles ('twin tails') or eye make-up ('winged eyeliner')."""
    t = _NOT_TRAIT_RX.sub(' ', text)
    out = {p for _, p in _animal_parts(t) if p in ('ear', 'tail', 'horn', 'wing')}
    return out | {k for k, rx in _TRAIT_RX.items() if rx.search(t)}


def look_slot(text, fac):
    """Slot of an appearance phrase whose facet bits (fac) conflict (hair + eyes, hair / eyes + body): BODY for body
    traits without hair words ('light blue wolf ears with inner white fluff'), EYES for eyes without hair words ('large
    blue eyes with intricate highlights'), HAIR for hair that only falls over an eye, else MIXED / MIXED_BODY (a body
    trait written with the hair or an eye colour: 'brown hair with droopy fox ears', 'fox ears ..., orange eyes')."""
    hair = bool(_HAIR_WORD_RX.search(text))
    eyes = bool(fac & T.FBIT['eyes'])
    t = _LINER_RX.sub(' ', text)
    traits = trait_kinds(t)        # 'fluffy ears', 'pointed ears'; a tail only next to no hair words ('two tails')
    if _animal_parts(t) or _BODY_TRAIT_RX.search(t) or traits - {'tail'} or ('tail' in traits and not hair):
        return MIXED_BODY if hair or eyes else BODY
    if hair and eyes:
        return HAIR if _EYE_OVER_RX.search(text) and not _EYE_WORD_RX.search(_EYE_OVER_RX.sub(' ', text)) else MIXED
    return HAIR if hair else EYES if eyes else BODY


@dataclass
class Sample:
    text: str
    length: str
    seed: object
    donors: dict = field(default_factory=dict)          # group -> donor row id
    subject: str = ''
    realized: dict = field(default_factory=dict)        # constraint -> phrase realizing it
    patched: list = field(default_factory=list)         # constraints realized by a patch phrase
    unrealized: list = field(default_factory=list)      # constraints with tag-text fallback
    raw: list = field(default_factory=list)             # free phrases inserted verbatim
    warnings: list = field(default_factory=list)
    rejected: list = field(default_factory=list)          # constraints refused (1other)
    phrases: list = field(default_factory=list)         # (text, group, source row, realizes a constraint)
    dropped: list = field(default_factory=list)         # [{'text', 'reason'}]: meta / artist tags, not in the text
    picked_character: object = None                     # character picked for a series constraint
    approx_tokens: int = 0                              # token estimate (model calib, affine words fit)
    template: object = None                             # donor row id whose caption gave the phrase order
    look_parts: dict = field(default_factory=dict)      # mixed look: hair / eyes shown, constrained, hidden, omitted
    # not in to_dict (its JSON contract is fixed); eval / explain_text read them
    coupled: bool = False                               # one coupled row is the action and the place donor
    sources: list = field(default_factory=list)         # (donor role, row id) of every phrase from a data row
    excluded: list = field(default_factory=list)        # active exclusions ('-x' constraints) of the request

    def to_dict(self):
        """Plain JSON-serializable dict (a fixed contract)."""
        return {'text': self.text, 'length': self.length,
                'seed': [None if x is None else int(x) for x in self.seed] if self.seed is not None else None,
                'subject': self.subject, 'donors': {str(k): int(v) for k, v in self.donors.items()},
                'realized': {str(k): str(v) for k, v in self.realized.items()}, 'patched': [str(x) for x in self.patched],
                'unrealized': [str(x) for x in self.unrealized], 'raw': [str(x) for x in self.raw],
                'warnings': [str(x) for x in self.warnings], 'rejected': [str(x) for x in self.rejected],
                'dropped': [{'text': str(d['text']), 'reason': str(d['reason'])} for d in self.dropped],
                'picked_character': None if self.picked_character is None else str(self.picked_character),
                'approx_tokens': int(self.approx_tokens),
                'phrases': [{'text': str(t), 'group': str(g), 'donor': _as_int(src), 'realizes': bool(r)}
                            for t, g, src, r in self.phrases]}

    def explain_text(self):
        out = ['  donors: ' + ', '.join(f'{k}={v}' for k, v in self.donors.items()) +
               (' (action + place: one coupled row)' if self.coupled else '') +
               (f'; order template={self.template}' if self.template is not None else '')]
        donor_rows = {(k, v) for k, v in self.donors.items()}
        extra = [f'{k}={v}' for k, v in self.sources if (k, v) not in donor_rows and
                 not (self.coupled and k == 'action + place' and v == self.donors.get('action'))]
        if extra:
            out.append('  other phrase sources (patches, replacements): ' + ', '.join(extra))
        for c, p in self.realized.items():
            out.append(f'  [{c}] <- "{p}"' + ('  (patch)' if c in self.patched else ''))
        if self.raw:
            out.append('  raw: ' + '; '.join(self.raw))
        if self.unrealized:
            out.append('  UNREALIZED (not expressed, or only as tag text): ' + ', '.join(self.unrealized))
        for w in self.warnings:
            out.append('  warning: ' + w)
        if self.rejected:
            out.append('  REJECTED: ' + ', '.join(self.rejected))
        if self.dropped:
            out.append('  dropped: ' + ', '.join(f'{d["text"]} ({d["reason"]})' for d in self.dropped))
        if self.picked_character:
            out.append('  picked character: ' + self.picked_character)
        for x in self.excluded:
            out.append(f'  [{x}] excluded')
        return '\n'.join(out)


def is_dependent(text):
    """Does the phrase refer back to the phrase before it ('resting another hand on ...', 'matching socks')?"""
    return bool(_DEPENDENT_RX.search(text))


def _as_int(x):
    if isinstance(x, (bool, str)) or x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _gather(off, flat, idx):
    idx = np.asarray(idx, dtype=np.int64)
    starts = off[idx]; lens = off[idx + 1] - starts
    tot = int(lens.sum())
    if tot == 0:
        return flat[:0], np.zeros(0, dtype=np.int64), lens
    owner = np.repeat(np.arange(len(idx)), lens)
    base = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return flat[base + np.arange(tot)], owner, lens


_ANIMAL_RX = T._PLAIN_ANIMAL_EARS


def _animal_parts(text):
    return {(m.group(1), m.group(2).rstrip('s').replace('eared', 'ear')) for m in _ANIMAL_RX.finditer(text)
            if m.group(2) not in ('girl',)}


_PREP = r'(behind|beside|around|near|above|below|over|under|toward|towards|at|to|for|with|on|in|from|of|against|' \
        r'onto|into|before|after|across|between|surrounding|framing|about|by)'


def _his(text):
    """Pronoun swap for the 1boy path: 'her' -> 'his' (possessive) or 'him' (object), herself -> himself."""
    t = re.sub(r'\bherself\b', 'himself', text)
    t = re.sub(r'\b' + _PREP + r' her\b(?=\s*(,|$|and\b|while\b|as\b|' + _PREP + r'\b))', r'\1 him', t)
    t = re.sub(r'\bher\b(?=\s*(,|$))', 'him', t)
    return re.sub(r'\bher\b', 'his', t)


class Generator:
    """Not thread-safe (generate() keeps scratch state on the instance): callers sharing one instance must lock."""

    def __init__(self, meta, arrays, pmi, tagdict=None, char_series=None, known_series=()):
        self.m = meta
        self.a = arrays
        self.pmi = pmi
        self.cfg = meta['cfg']
        self.ft = meta['freq_tags']
        self.F = len(self.ft)
        self.fid = {t: i for i, t in enumerate(self.ft)}
        self.tags_all = meta['tags_all']
        self.tag_freq_all = dict(zip(meta['tags_all'], meta['tag_freq']))
        self.phr = meta['phrases']
        self.chars = meta['chars']
        self.char_id = {c: i for i, c in enumerate(self.chars)}
        self.series = set(meta['series'])
        self.alias = meta['alias']
        self.tag_norm = meta['tag_norm']
        self.fac_pos = meta['fac_pos']
        self.keep = meta['keep']
        self.calib = meta['calib']
        self.ids = meta['donor_ids']
        self.N = len(self.ids)
        a = arrays
        self.nonempty = [np.nonzero(a['row_glen'][:, g] > 0)[0] for g in range(NG)]
        self.ne_mask = [a['row_glen'][:, g] > 0 for g in range(NG)]
        self._ne_ap = self.nonempty[T.ACTION][self.ne_mask[T.PLACE][self.nonempty[T.ACTION]]]
        # order templates: per occurrence its key (primary facet, TAIL_KEY for the tail), per row the key counts
        self._t_key = np.where(a['occ_group'] == T.TAIL, TAIL_KEY, a['pprim'][a['occ_pid']]).astype(np.int8)
        self._t_cnt = np.bincount(a['occ_row'].astype(np.int64) * NK + self._t_key,
                                  minlength=self.N * NK).reshape(self.N, NK)[:, :TAIL_KEY].astype(np.int16)
        self.order_pick = ORDER_PICK
        self.pscale = float(self.cfg['pmi_scale'])
        self._tok_cache = {}
        self._must_cache = {}
        self._mark = {}             # pid -> text_mark class ('' = none), filled lazily
        self._cur_drop = frozenset()
        self._cur_look = {}
        self._cur_donors = {}       # group -> donor row of the caption being generated
        self._cur_exempt = None     # its requested character's identity (exempt from the one-role rule)
        self._cur_distinct = True   # distinct_donor_chars of the caption being generated
        self._cur_src = {}          # donor role -> source rows (donors, patches, replacements, repairs)
        self._cur_coupled = False   # one coupled row is the action and the place donor
        self._cur_excl = []
        self._ban_cache = {}        # exclusion set -> _ban_masks (bounded)
        self._compound = {}         # index token -> merged from plain words (_is_compound; <= vocabulary size)
        self._cur_ban = None        # exclusions: banned rows per group / slot (_ban_masks), None without exclusions
        self._cur_items = []        # the final phrase items of the caption generated last
        self._irows = {}            # character identity -> its donor rows (memo)
        self._body_tag = {}
        self._build_slots()
        self.td = tagdict
        self.char_series_source = 'model' if char_series is None else 'given'     # load() sets metadata / json
        self.char_n = np.diff(a['cr_off']).astype(np.int64)            # solo donor rows per character
        self.series |= set(known_series)
        # characters without unambiguous metadata evidence: the series named by their qualifier / subject
        cs = {c: set(v) for c, v in (char_series or {}).items() if c in self.char_id and v}
        for c, v in self._char_series_from_model().items():
            cs.setdefault(c, v)
        self.char_series = cs
        self._build_subjects()
        # seen only in rows of another character (cosplay / crossover) without a subject naming it: its series is
        # unknown, so it is never picked for a series (explicit use is unchanged); absent in models before v2v3
        # background characters (binah drawn with maki) neither: their rows' subjects name another character
        self.crossover_chars = set(self.m.get('crossover_chars', ())) | set(self.m.get('background_chars', ()))
        self.series_chars = {}
        for c in self.chars:
            if self.char_n[self.char_id[c]] > 0 and c not in self.multi_subject and c not in self.crossover_chars:
                for sr in self.char_series.get(c, ()):
                    self.series_chars.setdefault(sr, []).append(c)
        self.series |= set(self.series_chars)
        self.identity = char_identities(self.chars, self.char_series, self.subject_of)
        self._ident_chars = {}
        for c, k in self.identity.items():
            self._ident_chars.setdefault(k, []).append(c)
        self._series_cache = {}
        self._build_suggest()

    @classmethod
    def load(cls, model_dir=os.path.join(HERE, 'model'), tags_path=TAGS_DEFAULT, metadata_path=_DEFAULT,
             char_series_path=_DEFAULT):
        """model_dir: meta.json (runtime format 2) or meta.pkl, arrays.npz, pmi.npy. tags_path: Danbooru tag
        dictionary (default data/danbooru-tags.json, else .parquet, if it exists; None = none, v1 behaviour); a .json
        dictionary and meta.json need only numpy + the stdlib, pyarrow only reads .parquet. The character -> series
        map, first that exists: metadata_path (training metadata, only character_name / series_name are read; default
        METADATA_DEFAULT), char_series_path (the same map exported by export_runtime.py; default CHAR_SERIES_DEFAULT,
        next to this file), else derived from the model's character subjects. None disables a source. The defaults
        are read at call time. self.char_series_source names the source used ('metadata', 'json', 'model')."""
        if metadata_path is _DEFAULT:
            metadata_path = METADATA_DEFAULT
        if char_series_path is _DEFAULT:
            char_series_path = CHAR_SERIES_DEFAULT
        meta = load_meta(model_dir)
        with np.load(os.path.join(model_dir, 'arrays.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        pmi = np.load(os.path.join(model_dir, 'pmi.npy'), allow_pickle=False)
        if meta.get('noun_heads'):
            T.NOUN_HEADS.clear(); T.NOUN_HEADS.update(meta['noun_heads'])
        td = None
        if tags_path is not None and (os.path.exists(tags_path) or tags_path != TAGS_DEFAULT):
            td = TD.TagDict.load(tags_path)
        cs, known, src = None, (), 'model'
        if metadata_path is not None and os.path.exists(metadata_path):
            cs, known = char_series_from_metadata(metadata_path, set(meta['chars']))
            src = 'metadata'
        elif char_series_path is not None and os.path.exists(char_series_path):
            cs, known = char_series_from_json(char_series_path)
            src = 'json'
        g = cls(meta, arrays, pmi, td, cs, known)
        g.char_series_source = src
        return g

    # ------------------------------------------------------------ appearance slots, text marks
    def _build_slots(self):
        """Per occurrence its appearance slot (HAIR / EYES / BODY / MIXED; -1 outside the generic-usable APPEARANCE
        occurrences) from the facet bits; only phrases whose bits conflict are read (look_slot). Rows per slot and the
        share of appearance rows that have each slot."""
        a = self.a
        pf = a['pfac'][a['occ_pid']]
        H, E, B = T.FBIT['hair'], T.FBIT['eyes'], T.FBIT['body']
        ap = (a['occ_group'] == T.APPEAR) & ~a['occ_leak'] & ~a['occ_ident']
        h, e = (pf & H) > 0, (pf & E) > 0
        slot = np.where(h, HAIR, np.where(e, EYES, BODY)).astype(np.int8)
        chk = ap & ((h & e) | ((h | e) & ((pf & B) > 0)))
        pids = np.unique(a['occ_pid'][chk])
        if pids.size:
            lut = np.zeros(len(self.phr), dtype=np.int8)
            lut[pids] = [look_slot(self.phr[int(p)], int(a['pfac'][p])) for p in pids]
            slot[chk] = lut[a['occ_pid'][chk]]
        slot[~ap] = -1
        self._occ_slot = slot
        self._occ_ix = np.arange(len(slot), dtype=np.int64)
        row = a['occ_row'].astype(np.int64)
        cnt = np.bincount(row[ap] * 5 + slot[ap], minlength=self.N * 5).reshape(self.N, 5) > 0
        self.slot_rows = [np.nonzero(cnt[:, s])[0] for s in range(3)]
        # an unconstrained slot donor must write a phrase of the slot that survives the insertion filters (no leak, no
        # character identity, no overlay / scene-text mark, no outfit sentence, no phrase that refers back) and, for
        # HAIR / EYES, describes the part (names_part): 'white highlights', 'red eyeliner', 'one hand running through
        # her hair' or "artist signature ... below hair" alone is no hair / eyes description
        perm = ~a['occ_leak'].astype(bool) & ~a['occ_ident'].astype(bool)
        self._slot_good = {}
        for s in (HAIR, EYES, BODY):
            m = (slot == s) & perm
            pids = np.unique(a['occ_pid'][m])
            ok = np.zeros(len(self.phr), dtype=bool)
            ok[pids] = [(s == BODY or names_part(self.phr[p], s)) and not is_dependent(self.phr[p]) and
                        not re.match(r'(wearing|dressed|clad)\b', self.phr[p]) and not self.mark_of(p)
                        for p in pids.tolist()]
            self._slot_good[s] = m & ok[a['occ_pid']]
            self.slot_rows[s] = np.unique(row[self._slot_good[s]])
        # presence: how often a real caption describes the part at all (a MIXED phrase counts for both parts)
        n = max(1, int(cnt.any(1).sum()))
        self.slot_share = [float(x.sum()) / n for x in (cnt[:, HAIR] | cnt[:, MIXED] | cnt[:, MIXED_BODY],
                                                         cnt[:, EYES] | cnt[:, MIXED],
                                                         cnt[:, BODY] | cnt[:, MIXED_BODY])]
        # HAIR / EYES presence by caption length, from the medium / short captions of the 16,772 solo 1girl rows
        # without a character (a part named anywhere, the subject included: 'a blonde girl'): hair 99.9 / 98.6 /
        # 59.3 %, eyes 97.2 / 85.0 / 18.6 % (long / medium / short). >= 90 % = always (no entry); eyes given the hair
        # is shown / not shown: medium 85.7 / 37.3 %, short 26.8 / 6.7 %
        self.look_rate = {'medium': {EYES: (0.857, 0.373)}, 'short': {HAIR: 0.593, EYES: (0.268, 0.067)}}

    def mark_of(self, p):
        """text_mark class of phrase p ('overlay', 'scene' or ''), cached."""
        v = self._mark.get(p)
        if v is None:
            v = self._mark[p] = T.text_mark(self.phr[p]) or ''
        return v

    def _dropped(self, o):
        """Is occurrence o a text mark filtered out of this caption (overlay_text / scene_text)?"""
        return bool(self._cur_drop) and self.mark_of(int(self.a['occ_pid'][o])) in self._cur_drop

    def _keep_marks(self, occ, c=None):
        """occ without the filtered text marks (unless constraint c asks for a mark)."""
        if not self._cur_drop or (c is not None and c.get('mark')) or not len(occ):
            return occ
        return occ[np.fromiter((not self._dropped(o) for o in occ), dtype=bool, count=len(occ))]

    # ------------------------------------------------------------ character subjects
    def _build_subjects(self):
        """subject_of[c]: the character's subject phrase, one person only. A canonical subject naming more beings
        ('nahida ... with two aranara', 'hakui koyori and kokoro') is replaced by the most frequent clean subject of
        the character's own rows that names it; characters without one keep it and are in multi_subject (never
        picked for a series; the explicit path warns)."""
        self._ser_norm = {_norm_words(x) for x in self.series}
        lex = set()
        for c in self.chars:
            b = _norm_words(_base_name(c))
            if len(b) >= 4 and (' ' in b or (b not in self.tag_freq_all and
                                             (self.td is None or self.td.category(b) != 'general'))):
                lex.add(b)
        self._name_lex = lex
        self.subject_of, self.multi_subject = {}, set()
        for i, c in enumerate(self.chars):
            subj = self.m['char_subject'][i]
            if self.multi_person(subj, c):
                alt = {}
                for r in self.char_rows(c):
                    x = self.m['donor_subject'][int(r)]
                    if x != subj and not self.multi_person(x, c) and _norm_words(_base_name(c)) in _norm_words(x):
                        alt[x] = alt.get(x, 0) + 1
                if alt:
                    subj = max(sorted(alt), key=lambda x: alt[x])
                else:
                    self.multi_subject.add(c)
            self.subject_of[c] = subj

    def multi_person(self, subject, char=None):
        """Does the subject phrase name more than one being? Checked outside the franchise name after 'from':
        coordination (and, &, together with, alongside), number words, plural person / creature nouns, 'with' + a
        person / creature, or a second known character name."""
        m = _SUBJ_OF_RX.match(subject)
        body = m.group(1) if m else subject
        who, _, frm = body.partition(' from ')
        ft = _norm_words(frm).split()
        n0 = len(ft)
        while ft:          # franchise names ('gunslinger girl and girls' frontline'); the rest is the tail
            k = next((k for k in range(min(len(ft), 12), 0, -1) if ' '.join(ft[:k]) in self._ser_norm), 0)
            if not k:
                break
            ft = ft[k:]
            if not (len(ft) > 1 and ft[0] in ('and', 'x', '&')):
                break
            if not any(' '.join(ft[1:k + 1]) in self._ser_norm for k in range(min(len(ft) - 1, 12), 0, -1)):
                break
            ft = ft[1:]
        tail = ''
        if frm and len(ft) < n0 and ft:     # the original text of the rest (hyphens kept: 'nine-tailed')
            tail = frm[[m.start() for m in _WORDS_RX.finditer(frm.lower())][n0 - len(ft)]:]
        for part in (who, tail):
            if part and (_MULTI_RX.search(part) or _WITH_BEING_RX.search(part)):
                return True
        own = _norm_words(_base_name(char)).split() if char else []
        if own and not tail and _norm_words(who) == ' '.join(own):
            return False    # 'anime illustration of X from Y'
        own_tag = ' ' + _norm_words(char) + ' ' if char else ''      # qualifier words ('lapis lazuli's first dream')
        if own:
            wt, on = ' ' + _norm_words(who + ' ' + tail) + ' ', ' '.join(own) + ' '
            if any(' with ' + d + on in wt for d in ('', 'the ', 'a ', 'an ', 'her ', 'his ', 'their ')):
                return True     # 'a girl with sylveon': the character is the companion
        # costumes name other characters without adding a person: 'X cosplaying Y', 'X as Y', 'in Y's outfit'
        person = _COSTUME_RX.split(who)[0]
        full_tail, tail = tail, _OUTFIT_RX.sub(' ', _COSTUME_RX.split(tail)[0])
        if own:
            # the character must be the HEAD subject: its name in the person part, not preceded by another person
            on = ' ' + ' '.join(own) + ' '
            pw = _norm_words(_QUAL_RX.sub(' ', person)).split()
            i = next((i for i in range(len(pw) - len(own) + 1) if pw[i:i + len(own)] == own), None)
            if i is None and on not in ' ' + _norm_words(person) + ' ' and \
                    on in ' ' + _norm_words(who + ' ' + full_tail) + ' ':
                return True     # 'koseki bijou cosplaying shimizu hinako', 'nikki ... dressed as momo rain
            if i:
                before = ' '.join(pw[:i])
                if _HEAD_PERSON_RX.search(before) and not _PERSONIFY_RX.search(before):
                    return True     # 'a girl holding galarian corsola': the head subject is someone else
        for part in (person, tail):
            toks = _norm_words(_QUAL_RX.sub(' ', part)).split()
            n = len(own)
            spans = [i for i in range(len(toks) - n + 1) if n and toks[i:i + n] == own]
            keep = [t if not any(i <= j < i + n for i in spans) else None for j, t in enumerate(toks)]
            for a in range(len(keep)):
                for b in range(a + 1, min(len(keep), a + 4) + 1):
                    seg = keep[a:b]
                    if None in seg:
                        break
                    g = ' '.join(seg)
                    if g in self._name_lex and (' ' + g + ' ') not in own_tag:
                        return True
        return False

    def _char_series_from_model(self):
        """Fallback (no metadata file, or a character seen only in crossover rows): the series named by the
        character's tag qualifier ('kanami (strinova)') or else by its subject ('... from strinova'), when that is
        a known series; otherwise no series ('ness (mother 2)' seen only next to inugami korone: mother 2, not the
        hololive of its subject)."""
        out = {}
        for i, c in enumerate(self.chars):
            v = set(q for q in re.findall(r'\(([^()]+)\)', c) if q in self.series)
            mm = re.search(r' from (.+)$', self.m['char_subject'][i])
            if not v and mm and mm.group(1) in self.series:     # the qualifier wins over a crossover row's subject
                v.add(mm.group(1))
            if v:
                out[c] = v
        return out

    # ------------------------------------------------------------ autocomplete
    def _build_suggest(self):
        td = self.td
        names = set(self.tags_all) | set(self.chars) | set(self.series)
        if td is not None:          # artist / meta names are dropped by parse(), unless the data knows them
            names |= {n for n, (cat, _) in td.d.items() if cat not in TD.DROP_CATEGORIES}
        names = sorted(names)
        train = dict(zip(self.chars, self.char_n.tolist()))
        for sr, cl in self.series_chars.items():
            if sr not in self.tag_freq_all:
                train[sr] = int(np.unique(np.concatenate([self.char_rows(c) for c in cl])).size)
        train.update(self.tag_freq_all)
        train = np.array([train.get(n, 0) for n in names], dtype=np.int64)
        dd = td.d if td is not None else {}
        post = np.array([dd.get(n, (0, 0))[1] for n in names], dtype=np.int64)
        rank = np.empty(len(names), dtype=np.int64)
        rank[np.lexsort((np.arange(len(names)), -post, -train))] = np.arange(len(names))   # names already sorted
        self._sg_names, self._sg_train, self._sg_post = names, train, post
        self._sg = TD.PrefixIndex(names, rank)

    def suggest_info(self, prefix, k=20):
        """Autocomplete: [{'tag', 'category', 'training_count', 'post_count', 'in_training'}]. Class 0 (the tag starts
        with the prefix) before class 1 (a later word starts with it), then training count, Danbooru posts, name.
        Artist and meta tags are excluded (they are dropped from captions). A leading '-' (an exclusion) suggests for
        the rest and prefixes each 'tag' with '-'."""
        out = []
        m = re.match(r'\s*-\s*', prefix)
        neg = bool(m) and bool(prefix[m.end():].strip())
        if neg:
            prefix = prefix[m.end():]
        for i, _ in self._sg.query(TD.norm_prefix(prefix), k):
            n = self._sg_names[i]
            out.append({'tag': '-' + n if neg else n, 'category': self.td.category(n) if self.td is not None else None,
                        'training_count': int(self._sg_train[i]), 'post_count': int(self._sg_post[i]),
                        'in_training': bool(self._sg_train[i] > 0)})
        return out

    # ------------------------------------------------------------ parsing
    def parse(self, constraints):
        """Constraints -> [dict(key, text, kind, fid, char, must, ...)]. A constraint '-x' (spaces after the '-'
        allowed) EXCLUDES x: x is resolved like a positive constraint; a tag (also a free phrase -> tag, or a known
        tag without a home group) -> kind 'exclude' (fid, must; the tag's and x's words are excluded lexically too),
        any other x with content words -> 'exclude' with fid None (its words, lexically); a character, series,
        subject, count or implicit tag (exact names only: '-red' is no 'red (among us)') -> 'exclude_unsupported'
        (ignored). text keeps the '-'. An exclusion a requirement contains (same tag or text, its words in a required
        text, series name or character subject) -> 'rejected' (reason 'also required'; the requirement wins)."""
        out, seen = [], set()
        for raw in constraints:
            s = re.sub(r'\s+', ' ', raw.strip().lower().replace('_', ' '))
            neg = s.startswith('-')
            if neg:
                s2 = s[1:].lstrip()
                if not s2:
                    continue
                if any(ch.isalnum() for ch in s2):        # a letter / digit of any script
                    s = '-' + s2
                else:
                    neg = False         # an emoticon tag ('-_-'), not an exclusion
            if not s or s in seen:
                continue
            seen.add(s)
            c = dict(key=len(out), text=s, kind='raw', fid=None, char=None, must=None)
            if neg:
                self._resolve_exclusion(c, s2)
            else:
                self._resolve(c, s)
            out.append(c)
        # an exclusion that a requirement contains (same tag, same text, or its words in the required text: 'blonde
        # hair' and '-hair') could never hold: the requirement wins
        pos = [c for c in out if not c.get('neg')]
        fids = {c['fid'] for c in pos if c['kind'] == 'tag'}
        # ... a series / character / subject requirement too ('blue archive' and '-blue': its subjects have it)
        texts = [t for c in pos if c['kind'] in ('tag', 'tag_unsupported', 'raw', 'series', 'char', 'subject')
                 for t in {c['text'], c.get('tag') or c['text'], c.get('subject') or c['text'],
                           self.subject_of.get(c.get('char'), c['text'])}]
        if any(c['kind'] == 'count' for c in pos) and not any(c['kind'] in ('series', 'char', 'subject') for c in pos):
            texts.append(BOY_SUBJECT)      # 1boy fixes the subject ('1boy, -boy')
        toks = [T.tokens(t) for t in texts]
        for c in out:
            if c['kind'] == 'exclude' and ((c['fid'] is not None and c['fid'] in fids) or
                                           any(x['text'] == c['text'][1:] for x in pos) or
                                           any(T.match_tokens(ct, t, head_rule=False) for ct in c['excl_ct'] for t in toks) or
                                           (c['excl_rx'] and any(c['excl_rx'].search(t) for t in texts))):
                c['kind'] = 'rejected'; c['reason'] = 'also required'
        return out

    def _resolve(self, c, s, mark=True, fuzzy=True):
        """parse() of one positive constraint s (normalized): sets c's kind, fid, char, must, subject, reason, tag,
        mark. An exclusion's resolution: mark=False (not computed, nothing cached), fuzzy=False (only an exact
        character tag is a character: '-red' is the colour, not 'red (among us)')."""
        if s in T.IMPLICIT_TAGS:
            c['kind'] = 'implicit'
        elif s == '1boy':
            c['kind'] = 'count'
        elif s == '1other' or (s not in self.series and s not in self.char_id and (_is_plural_request(s) or (
                (s.startswith('anime illustration') or T.is_subject_like(s)) and _is_coordinated(s)))):
            c['kind'] = 'rejected'      # an exact series / character name is never a plural request
        elif s.startswith('anime illustration'):
            c['kind'] = 'subject'; c['subject'] = s
        elif s in self.char_id:
            c['kind'] = 'char'; c['char'] = s
        elif s == ORIGINAL:
            c['kind'] = 'implicit'; c['reason'] = 'original'        # "not from a franchise": a generic subject
        elif self.td is not None and self.td.drop_reason(s) and not (fuzzy and self._char_base(s)):
            # a training character's base name ('serika', 'shigure ui') wins over a Danbooru artist / meta tag
            c['kind'] = 'dropped'; c['reason'] = self.td.drop_reason(s)
        elif (self.td is not None and self.td.category(s) == 'copyright') or s in self.series:
            c['kind'] = 'series'
        elif s in self.fid and self.a['home'][self.fid[s]] >= 0:
            c['kind'] = 'tag'; c['fid'] = self.fid[s]
        elif s in self.tag_freq_all or s in self.fid:
            c['kind'] = 'tag_unsupported'
        else:
            cb = self._char_base(s) if fuzzy else None
            key = ' '.join(T.content_tokens(s))
            al = self.alias.get(' '.join(T.tokens(s)))
            if _is_subject_np(s):
                # 'a boy', 'a cat girl': a subject noun phrase replaces the subject, never an action phrase
                c['kind'] = 'subject'
                c['subject'] = 'anime illustration of ' + (s if re.match(r'(a|an|the|one) ', s)
                                                           else ('an ' if s[0] in 'aeiou' else 'a ') + s)
            elif cb:
                best = max(cb, key=lambda x: self.a['cr_off'][self.char_id[x] + 1] - self.a['cr_off'][self.char_id[x]])
                c['kind'] = 'char'; c['char'] = best
            elif key in self.tag_norm:
                c['kind'] = 'tag'; c['fid'] = self.tag_norm[key]
            elif al and len(set(al)) == 1:
                # the free phrase is more specific than its tag ('coffee cup' -> cup): keep its words as a
                # lexical requirement on the realizing phrase
                c['kind'] = 'tag'; c['fid'] = al[0]; c['must'] = T.content_tokens(s)
        if c['kind'] == 'tag':
            c['tag'] = self.ft[c['fid']]
        # asks for a text mark (signature, watermark, english text, speech bubble): its phrases are never filtered
        if not mark:
            return
        c['mark'] = c['kind'] in ('tag', 'tag_unsupported', 'raw') and \
            (T.is_mark_request(c['text']) or T.is_mark_request(c.get('tag') or c['text']) or
             (c['kind'] == 'tag' and self._mostly_marks(c)))

    _EXCL_UNSUPPORTED = {'char': 'a character cannot be excluded', 'series': 'a series cannot be excluded',
                         'subject': 'a subject cannot be excluded', 'count': 'a count tag cannot be excluded',
                         'implicit': 'every caption implies it'}

    def _resolve_exclusion(self, c, s2):
        """parse() of an exclusion '-s2': s2 resolved like a positive constraint, then mapped (see parse)."""
        r = dict(c, text=s2)
        self._resolve(r, s2, mark=False, fuzzy=False)
        k = r['kind']
        c.update(neg=True, mark=False)
        if k in self._EXCL_UNSUPPORTED:
            c['kind'] = 'exclude_unsupported'
            c['reason'] = self._EXCL_UNSUPPORTED[k]
            c['char'] = r.get('char')
            return
        if k == 'tag':
            c.update(kind='exclude', fid=r['fid'], must=r['must'], tag=r['tag'])
        elif k == 'tag_unsupported':
            c.update(kind='exclude', fid=self.fid.get(s2), tag=s2)
        cts = [T.content_tokens(s2)]
        if c.get('tag') is not None and not c['must']:
            cts.append(T.content_tokens(c['tag']))
        c['excl_ct'] = tuple(dict.fromkeys(ct for ct in cts if ct))
        c['excl_rx'] = _plain_rx([s2] if c['must'] or c.get('tag') is None else [s2, T.tag_text(c['tag'])])
        if c['kind'] != 'exclude':
            if c['excl_ct']:
                c['kind'] = 'exclude'           # lexical only: its words
            else:
                c['kind'] = 'exclude_unsupported'; c['reason'] = 'no words to exclude'
                return
        if c['fid'] is None and all(w in T.STOP or len(w) < 2 for w in
                                    [w for ct in c['excl_ct'] for w in ct] + _plain_words(s2)):
            c['kind'] = 'exclude_unsupported'; c['reason'] = 'too short to exclude'     # '-e', '-in'
        elif all(self._violations(t, [], [c]) for t in self._subject_texts()):
            # 'anime illustration of' starts every subject (the FT captions need it): never removable
            c['kind'] = 'exclude_unsupported'; c['reason'] = "part of every caption's subject"

    def _subject_texts(self):
        """Every generic subject phrase (pick_subject's candidates and its default)."""
        v = self._must_cache.get('subject_texts')
        if v is None:
            v = self._must_cache['subject_texts'] = sorted({x['text'] for x in self.m['generic']} |
                                                           {'anime illustration of a girl'})
        return v

    def _mostly_marks(self, c):
        """A tag the data writes (almost) only as a text mark ('anniversary': "text reading '1st anniversary' in the
        top left") asks for one: >= 80 % of its first 200 realizing phrases are marks."""
        key = ('marks', c['fid'], c.get('must'))
        v = self._must_cache.get(key)
        if v is None:
            occ = self.key_occ(c)[:200]
            v = self._must_cache[key] = bool(occ.size) and \
                sum(bool(self.mark_of(int(self.a['occ_pid'][o]))) for o in occ) >= 0.8 * occ.size
        return v

    def _char_base(self, s):
        return self.m['char_base'].get(' '.join(T.raw_tokens(re.sub(r'\s*\([^)]*\)', ' ', s))))

    def suggest(self, prefix, k=20):
        return [d['tag'] for d in self.suggest_info(prefix, k)]

    # ------------------------------------------------------------ index helpers
    def rows_tg(self, f, g, lex=False):
        a = self.a
        k = f * NG + g
        if lex:
            return a['tgl_flat'][a['tgl_off'][k]:a['tgl_off'][k + 1]]
        return a['tgr_flat'][a['tgr_off'][k]:a['tgr_off'][k + 1]]

    def key_rows(self, c, g, lex):
        """Rows realizing constraint c in group g (free-phrase constraints: rows whose phrase has its words)."""
        if c.get('must'):
            return self.must_index(c['fid'], c['must'])[1][g]
        return self.rows_tg(c['fid'], g, lex)

    def key_occ(self, c):
        """Occurrences realizing constraint c (lexical first, alias only if nobody writes it)."""
        a = self.a
        f = c['fid']
        if c.get('must'):
            return self.must_index(f, c['must'])[0]
        occ = a['tol_flat'][a['tol_off'][f]:a['tol_off'][f + 1]]
        if occ.size == 0:
            occ = a['to_flat'][a['to_off'][f]:a['to_off'][f + 1]]
        return occ

    def occ_realizes(self, o, c):
        if c['fid'] not in self.occ_tags(o):
            return False
        return not c.get('must') or T.match_tokens(c['must'], self._ptoks(int(self.a['occ_pid'][o])))

    def _joint(self, cs, g, restrict, used, lex):
        sets = [self.key_rows(c, g, lex) for c in cs]
        cand = sets[0]
        for s in sets[1:]:
            cand = np.intersect1d(cand, s, assume_unique=True)
        if restrict is not None:
            cand = np.intersect1d(cand, restrict)
        return (cand[~np.isin(cand, used)] if used else cand), sets

    def _ptoks(self, p):
        t = self._tok_cache.get(p)
        if t is None:
            t = self._tok_cache[p] = T.tokens(self.phr[p])
        return t

    def must_index(self, f, must):
        """Occurrences realizing tag f whose text also contains the free phrase words `must`."""
        key = (f, must)
        v = self._must_cache.get(key)
        if v is None:
            a = self.a
            occ = a['to_flat'][a['to_off'][f]:a['to_off'][f + 1]]
            ok = np.array([T.match_tokens(must, self._ptoks(int(a['occ_pid'][o]))) for o in occ], dtype=bool) \
                if occ.size else np.zeros(0, dtype=bool)
            occ = occ[ok]
            rows = {g: np.unique(a['occ_row'][occ[a['occ_group'][occ] == g]]) for g in GROUPS}
            v = self._must_cache[key] = (occ, rows)
        return v

    def char_rows(self, c, pure=False):
        a = self.a; i = self.char_id[c]
        if pure:
            r = a['crp_flat'][a['crp_off'][i]:a['crp_off'][i + 1]]
            if r.size:
                return r
        return a['cr_flat'][a['cr_off'][i]:a['cr_off'][i + 1]]

    def text_occurrences(self, ct, fid=None):
        """Non-leak occurrences whose phrase lexically realizes the token pattern ct (any tag, also rare ones);
        compounds that mean something else are skipped (head-noun rule; learned 'shrine' maiden bigrams)."""
        key = ('text', ct, fid)
        v = self._must_cache.get(key)
        if v is None:
            a = self.a
            ids = [self.m['tvocab'].get(w) for w in set(ct)]
            if not ct or any(i is None for i in ids):
                v = np.zeros(0, dtype=np.int32)
            else:
                posts = sorted((a['tp_flat'][a['tp_off'][i]:a['tp_off'][i + 1]] for i in ids), key=len)
                pids = posts[0]
                for q in posts[1:]:
                    pids = np.intersect1d(pids, q, assume_unique=True)
                bad = set(self.m.get('bad_next', {}).get(fid, ())) if fid is not None else set()

                def ok(toks):
                    if not T.match_tokens(ct, toks):
                        return False
                    return not bad or any(w == ct[-1] and not (k + 1 < len(toks) and toks[k + 1] in bad)
                                          for k, w in enumerate(toks))
                pids = [int(x) for x in pids[:5000] if ok(self._ptoks(int(x)))]
                occ, _, _ = _gather(a['po_off'], a['po_flat'], pids) if pids else (np.zeros(0, dtype=np.int32), 0, 0)
                v = occ[~a['occ_leak'][occ] & ~a['occ_ident'][occ] & (a['occ_group'][occ] < NG)] if len(occ) else occ
            self._must_cache[key] = v
        return v

    def row_occ(self, r):
        return range(int(self.a['row_off'][r]), int(self.a['row_off'][r + 1]))

    def occ_tags(self, o):
        a = self.a
        return a['ot_flat'][a['ot_off'][o]:a['ot_off'][o + 1]]

    def group_tags(self, r, g):
        a = self.a; k = r * NG + g
        return a['rgt_flat'][a['rgt_off'][k]:a['rgt_off'][k + 1]]

    def tag_vector(self, committed):
        """Mean clipped PMI of every tag against the committed set + per-tag conflict flag."""
        if not committed:
            return np.zeros(self.F), np.zeros(self.F, dtype=bool)
        C = np.fromiter(committed, dtype=np.int64)
        sub = self.pmi[C]
        conf = (sub == -128).any(axis=0)
        v = np.where(sub == -128, -self.cfg['pmi_clip'] * self.pscale, sub).astype(np.float64).sum(0)
        return v / (len(C) * self.pscale), conf

    def score_rows(self, rows, g, vec):
        v, conf = vec
        vals, owner, lens = _gather(self.a['rgt_off'], self.a['rgt_flat'], np.asarray(rows, dtype=np.int64) * NG + g)
        s = np.zeros(len(rows)); bad = np.zeros(len(rows), dtype=bool)
        if len(vals):
            s = np.bincount(owner, weights=v[vals], minlength=len(rows))
            bad = np.bincount(owner, weights=conf[vals], minlength=len(rows)) > 0
        s = s / np.maximum(lens, 1)
        return s, bad

    def _softmax_pick(self, rng, score, bad, beta, reject=True, bias=None):
        ok = ~bad if reject else np.ones(len(score), dtype=bool)
        if not ok.any():
            ok = np.ones_like(bad)
        z = beta * COMPAT_SCALE * score + (bias if bias is not None else 0.0)
        z = np.where(ok, z, -np.inf)
        z = z - z.max()
        p = np.exp(z); p /= p.sum()
        return int(rng.choice(len(p), p=p))

    # ------------------------------------------------------------ donor choice
    def pick_donor(self, rng, g, cs, ctx, restrict=None):
        """Return (row, missing constraints) for group g; cs = constraints assigned to g. Candidate tiers: rows
        writing every constraint lexically, then with aliases, then the best partial rows (the rest is patched), then
        the restriction, then every row with the group. Within the first non-empty tier the rows of no named character
        already used by another donor role are taken (distinct_donor_chars); only if the tier has none, its rows of
        such a character (constraint realization beats character separation; _take warns). A tier restricted to the
        requested character's rows (restrict) is not separated: those rows are that character's own data."""
        used = ctx['used']
        own = restrict is not None          # the candidates are the requested character's rows
        if cs:
            # prefer donors that write every constraint out lexically; aliases only when nobody does
            cand, sets = self._joint(cs, g, restrict, used, True)
            if cand.size == 0:
                cand, sets = self._joint(cs, g, restrict, used, False)
            if cand.size == 0:
                allr = np.concatenate(sets) if sets else np.zeros(0, dtype=np.int32)
                if restrict is not None:
                    allr = allr[np.isin(allr, restrict)]
                if used:
                    allr = allr[~np.isin(allr, used)]
                if allr.size:
                    u, cnt = np.unique(allr, return_counts=True)
                    cand = u[cnt == cnt.max()]
                elif restrict is not None and restrict.size:
                    cand = restrict if not used else restrict[~np.isin(restrict, used)]
                else:
                    cand = self.nonempty[g]
                    own = False
        else:
            cand = self.nonempty[g] if restrict is None else self._isect_ne(restrict, g)
            if restrict is not None and cand.size == 0:
                cand = restrict
            if used and cand.size:
                c2 = cand[~np.isin(cand, used)]
                cand = c2 if c2.size else cand
        if cand.size == 0:
            return None, list(cs)
        cand = self._unbanned(cand, g)
        if not own:
            cand = self._sep_cand(ctx, cand)
        if cand.size > K_CAND:
            cand = rng.choice(cand, K_CAND, replace=False)
        beta, reject = ctx['beta'], ctx['reject']
        if beta == 0 and not reject:
            r = int(cand[int(rng.integers(len(cand)))])        # wild: uniform among the candidates
        else:
            score, bad = self.score_rows(cand, g, self.tag_vector(ctx['committed']))
            r = int(cand[self._softmax_pick(rng, score, bad, beta, reject)])
        missing = [c for c in cs if not self._row_realizes(r, g, c, restrict is not None)]
        return r, missing

    def _row_realizes(self, r, g, c, allow_ident=False):
        a = self.a
        for o in self.row_occ(r):
            if a['occ_group'][o] != g or a['occ_leak'][o] or (a['occ_ident'][o] and not allow_ident):
                continue
            if self.occ_realizes(o, c) and not (self._dropped(o) and not c.get('mark')):
                return True
        return False

    # ------------------------------------------------------------ mixed appearance (hair / eyes / body slots)
    def _slot_of(self, c):
        """The slot whose phrases realize constraint c most often (else by the tag's own words)."""
        key = ('slot', c['fid'], c.get('must'))
        s = self._must_cache.get(key)
        if s is None:
            sl = self._occ_slot[self.key_occ(c)]
            sl = sl[(sl >= 0) & (sl < MIXED)]
            if sl.size:
                s = int(np.argmax(np.bincount(sl, minlength=3)))
            else:
                fs = T.facets_of(T.tag_text(c.get('tag') or c['text']))
                s = HAIR if 'hair' in fs else EYES if 'eyes' in fs else BODY
            self._must_cache[key] = s
        return s

    def _slot_cand(self, c, s, lex):
        """Rows realizing constraint c in a phrase of slot s (lexical realizations only when lex)."""
        key = ('slotrows', c['fid'], c.get('must'), s, lex)
        v = self._must_cache.get(key)
        if v is None:
            a, f = self.a, c['fid']
            if c.get('must'):
                occ = self.must_index(f, c['must'])[0]
            else:
                occ = (a['tol_flat'][a['tol_off'][f]:a['tol_off'][f + 1]] if lex else
                       a['to_flat'][a['to_off'][f]:a['to_off'][f + 1]])
            v = self._must_cache[key] = np.unique(a['occ_row'][occ[self._occ_slot[occ] == s]])
        return v

    def _row_realizes_slot(self, r, s, c):
        return any(self._occ_slot[o] == s and self.occ_realizes(o, c) for o in self.row_occ(r))

    def _row_ids(self, r):
        return {self.identity[x] for x in self.m['donor_chars'][r] if x in self.identity}

    def _slot_tags(self, r, s):
        """Tags realized by row r's slot-s phrases; BODY adds the row's body-type tags (cat ears, horse girl, ...)
        that subjects key on, never its hair / eye tags."""
        out = {int(t) for o in self.row_occ(r) if self._occ_slot[o] == s for t in self.occ_tags(o)}
        if s == BODY:
            a = self.a
            for j in a['row_tag_flat'][a['row_tag_off'][r]:a['row_tag_off'][r + 1]]:
                j = int(j)
                if a['home'][j] in (T.APPEAR, T.SUBJ) and self._is_body_tag(j):
                    out.add(j)
        return out

    def _is_body_tag(self, j):
        v = self._body_tag.get(j)
        if v is None:
            fs = T.facets_of(T.tag_text(self.ft[j]))
            v = self._body_tag[j] = not (fs & {'hair', 'eyes', 'eyestate', 'hairacc'})
        return v

    def _ident_rows(self, ids):
        """Donor rows of the character identities ids (every tag of each identity)."""
        rows = [self._rows_of_ident(k) for k in ids]
        return np.unique(np.concatenate(rows)) if rows else np.zeros(0, dtype=np.int64)

    def _rows_of_ident(self, k):
        v = self._irows.get(k)
        if v is None:
            rows = [self.char_rows(c) for c in self._ident_chars.get(k, ())]
            v = self._irows[k] = np.unique(np.concatenate(rows)).astype(np.int64) if rows else \
                np.zeros(0, dtype=np.int64)
        return v

    # ------------------------------------------------------------ one donor role per named character
    def _excl_rows(self, ids, exempt=False):
        """The rows an identity exclusion removes: the donor rows of the identities ids, never a row of the requested
        character (exempt: its identity; default the caption's, with distinct_donor_chars). Every identity filter
        (donor picks, slots, patches, lost tags, backdrop replacement) goes through here."""
        rows = self._ident_rows(ids)
        if exempt is False:
            exempt = self._cur_exempt if self._cur_distinct else None
        if exempt is not None and rows.size:
            rows = rows[~np.isin(rows, self._rows_of_ident(exempt))]
        return rows

    def _ids_mask(self, ids):
        """Boolean row mask of the rows an exclusion of the identities ids removes (_excl_rows)."""
        m = np.zeros(self.N, dtype=bool)
        if ids:
            m[self._excl_rows(ids)] = True
        return m

    def _named(self, r, exempt):
        """The named identities row r contributes: none for a row of the requested character (identity exempt; its
        co-tagged identities, e.g. an alter ego, are part of that character's own data), else all of them."""
        ids = self._row_ids(r)
        return set() if exempt is not None and exempt in ids else ids

    def _take(self, ctx, r, role):
        """Commit donor row r of a donor role (a group, a look slot, 'action + place' for the coupled row). With
        distinct_donor_chars its named characters (_named) are excluded from every later donor pick (ctx['ids'],
        ctx['imask']) and recorded as the role's sources (_reg); a row that shares one (only such rows realize the
        role's constraints) warns."""
        ctx['used'].append(r)
        if not ctx.get('distinct'):
            return
        ids = self._named(r, ctx.get('exempt'))
        shared = ids & ctx['ids']
        if shared:
            ctx['warn'].append(f'{role}: no row of another character has it; two donor roles come from '
                               f'{", ".join(sorted(shared))}')
        new = ids - ctx['ids']
        if new:
            ctx['imask'][self._excl_rows(new, ctx.get('exempt'))] = True
        ctx['ids'] |= ids
        self._reg(role, r)

    def _reg(self, role, r):
        """Record row r as a source of donor role role (donor, patch, replacement, repair; never removed, so the
        earlier sources of a role whose protected phrases survive stay)."""
        if role is not None:
            self._cur_src.setdefault(role, []).append(int(r))

    def _group_role(self, g):
        """The donor role of group g ('action + place' for both groups of the coupled row)."""
        if g is None:
            return None
        if self._cur_coupled and g in (T.ACTION, T.PLACE):
            return 'action + place'
        return T.GROUP_NAMES[g]

    def _cons_role(self, c):
        """The donor role of a constraint assigned to a group (its look slot in a mixed look)."""
        if c.get('slot') is not None and self._cur_look:
            return SLOT_NAMES[c['slot']]
        return self._group_role(c.get('group'))

    def _occ_role(self, o):
        """The donor role a phrase occurrence would join (its group; its look slot in a mixed look)."""
        a = self.a
        g = int(a['occ_group'][o])
        if g == T.APPEAR and self._cur_look:
            s = int(self._occ_slot[o])
            return SLOT_NAMES[s] if 0 <= s < MIXED else SLOT_NAMES[HAIR] if s >= MIXED else SLOT_NAMES[BODY]
        return self._group_role(g)

    def _all_source_rows(self):
        return {r for rows in self._cur_src.values() for r in rows} | set(self._cur_look.values()) | \
            set(self._cur_donors.values())

    def _sep_cand(self, ctx, cand):
        """The candidates that share no named character with the donors taken so far (distinct_donor_chars); all of
        them if every candidate does (the candidate tier is kept: realizing constraints beats separating
        characters, _take warns)."""
        if not ctx.get('distinct') or not ctx.get('ids') or not len(cand):
            return cand
        cand = np.asarray(cand)
        e = cand[~ctx['imask'][cand]]
        return e if e.size else cand

    def _other_ids(self, role=None, slot=None):
        """Named identities (_named) of the sources of every donor role but role (None: every role).
        distinct_donor_chars=False: the mixed look's slots other than slot (phase 4)."""
        if not self._cur_distinct:
            look = self._cur_look
            return set().union(*(self._row_ids(r) for s, r in look.items() if s != slot)) if look else set()
        ex, out = self._cur_exempt, set()
        for k, rows in self._cur_src.items():
            if k != role:
                for r in rows:
                    out |= self._named(r, ex)
        return out

    def _lost_sep(self, occ):
        """The occurrences whose rows have no named identity of a donor role other than the one each would join
        (_occ_role)."""
        a, ex, cache = self.a, self._cur_exempt, {}
        keep = np.ones(len(occ), dtype=bool)
        for i, o in enumerate(occ.tolist()):
            role = self._occ_role(o)
            if role not in cache:
                cache[role] = self._other_ids(role)
            if cache[role] and self._named(int(a['occ_row'][o]), ex) & cache[role]:
                keep[i] = False
        return occ[keep]


    def _pick_slot(self, rng, s, cs, ctx, ids):
        """(donor row, reused) for slot s. Eligible rows are not used yet and share no character identity (ids) with
        the other slots. Candidate tiers, each restricted to eligible rows: rows writing all of the slot's constraints
        cs lexically, then with aliases, then the best partial rows (the rest is patched), then every row with the
        slot. Only if no tier has an eligible row is a used row / identity reused (reused=True, the caller warns).
        Uniform at wild >= 0.5, PMI-weighted (the row's slot tags vs the committed tags) below."""
        beta, reject = ctx['beta'], ctx['reject']
        # rows of the identities ids (ctx['imask'] when ids are the caption's donor identities) + used rows
        bad = (ctx['imask'] if ids is ctx.get('ids') and ctx.get('imask') is not None else self._ids_mask(ids)).copy()
        bad[np.asarray(ctx['used'], dtype=np.int64)] = True
        if not cs and beta == 0 and not reject and self.slot_rows[s].size:
            cand = self.slot_rows[s]
            sban = self._cur_ban['slot', s] if self._cur_ban is not None else None
            for _ in range(16):         # wild, unconstrained: a few uniform draws find an eligible row
                r = int(cand[int(rng.integers(len(cand)))])
                if not bad[r] and (sban is None or not sban[r]):
                    return r, False

        def elig(x):
            return x[~bad[x]] if x.size else x
        tiers = []
        if cs:
            for lex in (True, False):
                sets = [self._slot_cand(c, s, lex) for c in cs]
                joint = sets[0]
                for x in sets[1:]:
                    joint = np.intersect1d(joint, x, assume_unique=True)
                tiers.append(joint)
            allr = elig(np.concatenate(sets))
            if allr.size:
                u, cnt = np.unique(allr, return_counts=True)
                tiers.append(u[cnt == cnt.max()])
            if s in (HAIR, EYES):       # first the rows that also name the part ('eyepatch' rows with an eye colour)
                tiers = [x for t in tiers for x in (np.intersect1d(t, self.slot_rows[s], assume_unique=True), t)]
        tiers.append(self.slot_rows[s])
        cand = next((e for e in (elig(t) for t in tiers) if e.size), None)
        reused = cand is None
        if reused:
            cand = next((t for t in tiers if t.size), None)
            if cand is None:
                return None, False
        cand = self._unbanned(cand, ('slot', s))
        if cand.size > K_CAND:
            cand = rng.choice(cand, K_CAND, replace=False)
        if beta == 0 and not reject:
            return int(cand[int(rng.integers(len(cand)))]), reused
        v, conf = self.tag_vector(ctx['committed'])
        occ, owner, _ = _gather(self.a['row_off'], self._occ_ix, cand)
        m = self._occ_slot[occ] == s
        vals, own2, _ = _gather(self.a['ot_off'], self.a['ot_flat'], occ[m])
        own = owner[m][own2]
        n = np.maximum(np.bincount(own, minlength=len(cand)), 1)
        score = np.bincount(own, weights=v[vals], minlength=len(cand)) / n if len(vals) else np.zeros(len(cand))
        bad = np.bincount(own, weights=conf[vals], minlength=len(cand)) > 0 if len(vals) else np.zeros(len(cand), bool)
        return int(cand[self._softmax_pick(rng, score, bad, beta, reject)]), reused

    def _mark_free_tail(self, rng, used, avoid=frozenset()):
        """A random donor row whose style tail has a style phrase and no text mark (bounded tries), preferring a row
        of none of the named characters avoid (else the first such row of one of them). Returns (row or None,
        shares an avoided identity)."""
        a, cand = self.a, self.nonempty[T.TAIL]
        first = banned = None
        ban = self._cur_ban[T.TAIL] if self._cur_ban is not None else None
        for _ in range(K_TAIL_FALLBACK if cand.size else 0):
            r = int(cand[int(rng.integers(len(cand)))])
            occ = [o for o in self.row_occ(r) if a['occ_group'][o] == T.TAIL and not a['occ_leak'][o] and
                   not a['occ_ident'][o]]
            if r not in used and occ and not any(self.mark_of(int(a['occ_pid'][o])) for o in occ) and \
                    any(a['pfac'][a['occ_pid'][o]] & T.FBIT['style'] for o in occ):
                if ban is not None and ban[r]:      # its tail has an excluded item: only if nothing else is found
                    banned = r if banned is None else banned
                    continue
                if not (avoid and self._named(r, self._cur_exempt) & avoid):
                    return r, False
                if first is None:
                    first = r
        if first is None and banned is not None:
            return banned, False
        return first, first is not None

    def _pick_look(self, rng, cs, soft, ctx):
        """Mixed appearance: one donor row per slot (HAIR, EYES, BODY), distinct rows and named characters. A slot
        without constraints is present with the share of appearance rows that have it (BODY ~42 %: not every girl
        gets ears or wings). Soft constraints (body traits a chosen outfit / scene phrase writes) narrow the slot's
        candidates when possible and are never patched. A slot that can only reuse a row / character warns
        (ctx['warn']). Returns ({slot: row}, missing constraints)."""
        by = {s: [] for s in range(3)}
        for c in cs:
            c['slot'] = self._slot_of(c)
            by[c['slot']].append(c)
        sby = {s: [] for s in range(3)}
        for c in soft:
            sby[self._slot_of(c)].append(c)
        look, missing = {}, []
        # distinct_donor_chars: the caption's donor identities (groups picked before the look too); else the look's
        distinct = ctx.get('distinct')
        ids = ctx['ids'] if distinct else set()
        for s in sorted(range(3), key=lambda s: (-len(by[s]), s)):
            # hair and eyes are always drawn (a cue / the caption length may still leave them out, _look_parts); body
            # traits at the share of real rows that have them
            if s == BODY and not by[s] and not sby[s] and rng.random() >= self.slot_share[s]:
                continue
            r = None
            if sby[s] and all(self._slot_cand(c, s, False).size for c in sby[s]):
                hard = by[s] + sby[s]
                joint = self._slot_cand(hard[0], s, False)
                for c in hard[1:]:
                    joint = np.intersect1d(joint, self._slot_cand(c, s, False), assume_unique=True)
                if joint.size:
                    r, reused = self._pick_slot(rng, s, hard, ctx, ids)
                    if r is not None and (reused or not all(self._row_realizes_slot(r, s, c) for c in hard)):
                        r = None        # soft traits only from an eligible row that also writes the hard ones
            if r is None:
                r, reused = self._pick_slot(rng, s, by[s], ctx, ids)
                if not distinct and r is not None and reused and self._row_ids(r) & ids:
                    ctx['warn'].append(f'{SLOT_NAMES[s]}: no row of another character has it; two appearance parts '
                                       f'come from {", ".join(sorted(self._row_ids(r) & ids))}')
            if r is None:
                missing += by[s]
                continue
            look[s] = r
            if distinct:
                self._take(ctx, r, SLOT_NAMES[s])      # warns when only a used character's row has the slot
            else:
                ctx['used'].append(r)
                ids |= self._row_ids(r)
            missing += [c for c in by[s] if not self._row_realizes_slot(r, s, c)]
            ctx['committed'].update(self._slot_tags(r, s))
        return look, missing

    def pick_coupled(self, rng, csA, csP, ctx):
        """wild < 0.5: one row is both the ACTION and the PLACE donor (it must satisfy both groups' constraints)."""
        used = ctx['used']

        def cands(g, cs):
            if not cs:
                return self.nonempty[g]
            c1, _ = self._joint(cs, g, None, used, True)
            return c1 if c1.size else self._joint(cs, g, None, used, False)[0]
        if not csA and not csP:
            cand = self._ne_ap
        elif not csP:
            cand = self._isect_ne(cands(T.ACTION, csA), T.PLACE)
        elif not csA:
            cand = self._isect_ne(cands(T.PLACE, csP), T.ACTION)
        else:
            cand = np.intersect1d(cands(T.ACTION, csA), cands(T.PLACE, csP))
        if used:
            cand = cand[~np.isin(cand, used)]
        if cand.size == 0:
            return None
        if self._cur_ban is not None:
            keep = cand[~(self._cur_ban[T.ACTION][cand] | self._cur_ban[T.PLACE][cand])]
            cand = keep if keep.size else cand
        cand = self._sep_cand(ctx, cand)
        if cand.size > K_CAND:
            cand = rng.choice(cand, K_CAND, replace=False)
        beta, reject = ctx['beta'], ctx['reject']
        if beta == 0 and not reject:
            return int(cand[int(rng.integers(len(cand)))])
        vec = self.tag_vector(ctx['committed'])
        s1, b1 = self.score_rows(cand, T.ACTION, vec)
        s2, b2 = self.score_rows(cand, T.PLACE, vec)
        return int(cand[self._softmax_pick(rng, (s1 + s2) / 2, b1 | b2, beta, reject)])

    def _isect_ne(self, x, g):
        """np.intersect1d(x, self.nonempty[g]) without hashing the 74 k-row side."""
        x = np.asarray(x)
        x = x[self.ne_mask[g][x]]
        return x if x.size < 2 or bool((x[1:] > x[:-1]).all()) else np.unique(x)

    def choose_occ(self, rng, occ, ctx, avoid=(), c=None):
        """One occurrence out of occ: short single-purpose phrases preferred at any wildness, PMI at low wild. Text
        marks filtered from this caption are skipped (unless constraint c asks for one); None if nothing is left."""
        a = self.a
        full = occ
        if occ.size > K_PATCH:
            occ = rng.choice(occ, K_PATCH, replace=False)
        if self._cur_drop:
            occ = self._keep_marks(occ, c)
            if not occ.size:            # the subsample was all marks: the whole list, filtered
                occ = self._keep_marks(full, c)
                if not occ.size:
                    return None
                if occ.size > K_PATCH:
                    occ = rng.choice(occ, K_PATCH, replace=False)
        beta, reject = ctx['beta'], ctx['reject']
        need = beta > 0 or reject
        v, conf = self.tag_vector(ctx['committed']) if need else (None, None)
        score = np.zeros(len(occ)); bad = np.zeros(len(occ), dtype=bool); bias = np.zeros(len(occ))
        for k, o in enumerate(occ):
            ts = self.occ_tags(o)
            other = [t for t in ts if t not in avoid]
            bias[k] = -0.4 * len(other) - 0.05 * a['pwords'][a['occ_pid'][o]]
            if need:
                score[k] = v[other].mean() if other else 0.0
                bad[k] = bool(conf[ts].any()) if len(ts) else False
        return int(occ[self._softmax_pick(rng, score, bad, beta, reject, bias)])

    def pick_subject(self, rng, S, subj_cons, warn, need_boy=False):
        if need_boy:
            warn.append('1boy: the solo data has (almost) no male rows; the caption body comes from girl donors')
            return BOY_SUBJECT
        cands, w = [], []
        for gsub in self.m['generic']:
            if any(j not in S for j in gsub['key']):
                continue
            if any(f not in gsub['real'] for f in subj_cons):
                continue
            cands.append(gsub['text']); w.append(gsub['w'])
        if not cands:
            return 'anime illustration of a girl'
        if self._cur_ban is not None:       # exclusions: a subject without the excluded words, if any
            ok = [not self._violations(x, [], self._cur_excl) for x in cands]
            if any(ok) and not all(ok):
                cands = [x for x, k in zip(cands, ok) if k]; w = [x for x, k in zip(w, ok) if k]
        w = np.array(w); w /= w.sum()
        return cands[int(rng.choice(len(cands), p=w))]

    # ------------------------------------------------------------ items
    def item_realizes(self, it, c):
        """Does phrase item `it` express constraint c (full matcher, not literal presence)?"""
        k = c['kind']
        if k == 'raw':
            return c['text'] in it['text']
        if k in ('tag', 'tag_unsupported'):
            ct = c.get('must') or T.content_tokens(c.get('tag') or c['text'])
            toks = T.tokens(it['text'])
            if T.match_tokens(ct, toks):
                return True
            if it.get('occ') is not None and c.get('fid') is not None and self.occ_realizes(it['occ'], c):
                # an indexed realization counts only as an alias (the tag's own words are absent); a lexical one
                # must pass the matcher ('schools of fish' is not a school)
                return not (ct and all(w in toks for w in ct))
            kind = _GENERIC_TRAIT.get(c.get('tag'))
            # a mixed look: any animal ear is an 'animal ears' (one ear description, not 'wolf ears' + 'animal ears')
            return bool(kind and self._cur_look and kind in trait_kinds(it['text']) and
                        not _ACCESSORY_RX.search(it['text']))
        return False

    def _item_occ(self, o, keys, origin):
        a = self.a
        p = int(a['occ_pid'][o])
        return dict(text=self.phr[p], pid=p, occ=int(o), prim=T.PRIMARY_ORDER[a['pprim'][p]], group=int(a['occ_group'][o]),
                    src=self.ids[int(a['occ_row'][o])], keys=set(keys), words=int(a['pwords'][p]),
                    fac=int(a['pfac'][p]), origin=origin, role=None)

    def _item_text(self, txt, keys, origin):
        fs = T.facets_of(txt)
        g = T.group_of(fs)
        if g == T.SUBJ:
            g = T.ACTION
        fac = 0
        for f in fs:
            fac |= T.FBIT.get(f, 0)
        return dict(text=txt, pid=None, prim=T.primary_facet(fs, g), group=g, src=origin, keys=set(keys),
                    words=len(txt.split()), fac=fac, origin=origin, role=None)

    # ------------------------------------------------------------ generation
    def _one(self, cons, rng, length, beta, wild=1.0, pre_warn=(), order='template', overlay_text=False,
             scene_text=True, mix_look=True, distinct_donor_chars=True):
        a = self.a
        neg = [c for c in cons if c.get('neg')]          # exclusions: never a constraint (generate() checks them)
        if neg:
            cons = [c for c in cons if not c.get('neg')]
        wild = min(1.0, max(0.0, float(wild)))
        # ids / imask: named character identities of the donors taken so far and their rows (distinct_donor_chars)
        ctx = dict(beta=(1.0 - wild) * beta, reject=wild < 0.5, committed=set(), used=[],
                   distinct=bool(distinct_donor_chars), ids=set(), imask=np.zeros(self.N, dtype=bool), exempt=None)
        self._cur_drop = frozenset(([] if overlay_text else [T.OVERLAY]) + ([] if scene_text else [T.SCENE_TEXT]))
        coupled = wild < 0.5
        warn, rejected = list(pre_warn), []
        ctx['warn'] = warn
        self._cur_look = {}
        self._cur_donors = {}
        self._cur_distinct = ctx['distinct']
        self._cur_exempt = None
        self._cur_src = {}
        self._cur_coupled = False
        byk = {c['key']: c for c in cons}
        groups_cons = {g: [] for g in GROUPS}
        subj_cons = []
        tagc = [c for c in cons if c['kind'] == 'tag']
        # a series constraint with a picked character takes the character path
        charc = [c for c in cons if c['kind'] == 'char'] or [c for c in cons if c['kind'] == 'series' and c.get('char')]
        rawc = [c for c in cons if c['kind'] == 'raw']
        lost = [c for c in cons if c['kind'] == 'tag_unsupported']       # need a phrase search
        subjc = [c for c in cons if c['kind'] == 'subject']
        if any(_is_plural_request(c['subject']) for c in subjc):
            for c in subjc:
                if _is_plural_request(c['subject']):
                    c['kind'] = 'rejected'
            subjc = [c for c in subjc if c['kind'] == 'subject']
        # an explicit subject wins over a contradicting count tag (1boy vs 'a girl', 1girl vs 'a boy')
        subj_gender = None
        if subjc:
            if re.search(r'\b(boy|man|male|gentleman|husband|son|brother)\b', subjc[0]['subject']):
                subj_gender = 'boy'
            elif re.search(r'\b(girl|woman|lady|female|maid|sister|wife|daughter|catgirl|foxgirl|schoolgirl)\b',
                           subjc[0]['subject']):
                subj_gender = 'girl'
        count_conflict = [c for c in cons if (c['kind'] == 'count' and subj_gender == 'girl') or
                          (c['kind'] == 'implicit' and c['text'] == '1girl' and subj_gender == 'boy')]
        for c in count_conflict:
            warn.append(f'{c["text"]} contradicts the subject "{subjc[0]["subject"]}"; the subject wins, '
                        f'{c["text"]} not realized')
        need_boy = any(c['kind'] == 'count' and c not in count_conflict for c in cons) or subj_gender == 'boy'
        for msg in _user_conflicts(cons):
            warn.append(msg)
        for c in cons:
            if c['kind'] == 'rejected':
                rejected.append(c['text'])
                warn.append(f'{c["text"]}: rejected, v1 generates single-character (solo) captions only')
        for c in neg:
            if c['kind'] == 'rejected':
                rejected.append(c['text'])
                warn.append(f'{c["text"]}: rejected, {c["text"][1:]} is also required (the requirement wins)')
            elif c['kind'] == 'exclude_unsupported':
                warn.append(f'{c["text"]}: ignored, {c["reason"]}')
        committed = ctx['committed']
        for c in list(tagc):
            f = c['fid']; committed.add(f)
            h = int(a['home'][f])
            if c.get('must'):
                occ, rows = self.must_index(f, c['must'])
                if occ.size == 0:
                    # nobody writes it like that: insert the free phrase verbatim
                    c['kind'] = 'raw'; tagc.remove(c); rawc.append(c); continue
                h = max(GROUPS, key=lambda g: rows[g].size)
            elif h == T.SUBJ:
                subj_cons.append(c); continue
            elif h < 0 or self.rows_tg(f, h).size == 0:
                alt = [g for g in GROUPS if self.rows_tg(f, g).size]
                if not alt:
                    lost.append(c); continue
                h = max(alt, key=lambda g: a['grp_hist'][f, g])
            c['group'] = h
            groups_cons[h].append(c)
        char = charc[0]['char'] if charc else None
        if len(charc) > 1:
            warn.append('only one character per caption (solo): using ' + char)
        crow = self.char_rows(char) if char else None
        crow_pure = self.char_rows(char, pure=True) if char else None
        char_set = set(int(x) for x in crow) if char else set()
        if char and char in self.fid:
            committed.add(self.fid[char])
        # the requested character gives the appearance and the costume: exempt from the one-role-per-character rule
        ctx['exempt'] = self._cur_exempt = self.identity.get(char) if char else None

        # ---- donors: constrained groups first (most constrained first)
        def ncons(g):
            n = len(groups_cons[g]) + (1 if char and g == T.APPEAR else 0)
            if coupled and g == T.ACTION:
                n += len(groups_cons[T.PLACE])
            return n
        gorder = sorted([g for g in GROUPS if not (coupled and g == T.PLACE)], key=lambda g: (-ncons(g), GROUPS.index(g)))
        donors, missing_all = {}, {}
        self._cur_donors = donors
        used = ctx['used']
        soft = []
        look = {}             # mixed appearance: slot -> donor row (generic captions; characters keep their own look)
        for g in gorder:
            if g == T.APPEAR and mix_look and not char:
                look, missing_all[g] = self._pick_look(rng, groups_cons[g], soft, ctx)
                if look:
                    donors[g] = look.get(HAIR, next(iter(look.values())))
                continue
            if coupled and g == T.ACTION:
                r = self.pick_coupled(rng, groups_cons[T.ACTION], groups_cons[T.PLACE], ctx)
                if r is not None:
                    for gg in (T.ACTION, T.PLACE):
                        donors[gg] = r
                        missing_all[gg] = [c for c in groups_cons[gg] if not self._row_realizes(r, gg, c)]
                        committed.update(int(x) for x in self.group_tags(r, gg))
                    self._cur_coupled = True
                    self._take(ctx, r, 'action + place')      # one coupled row: one donor role
                    continue
            restrict = None
            if char and g == T.APPEAR:
                restrict = crow_pure
            elif char and g == T.OUTFIT and rng.random() < P_CHAR_OUTFIT:
                restrict = crow
            r = None
            if g == T.APPEAR and soft:
                # body traits written by already chosen outfit / scene constraint phrases ('rabbit ears poking
                # through a witch hat'): ask the appearance donor for them too, but never patch them
                hard = groups_cons[g] + soft
                cand, _ = self._joint(hard, g, restrict, used, False)
                if cand.size:
                    r, missing = self.pick_donor(rng, g, hard, ctx, restrict)
                    missing = [c for c in missing if c in groups_cons[g]]
            if r is None:
                r, missing = self.pick_donor(rng, g, groups_cons[g], ctx, restrict)
            if r is None:
                missing_all[g] = list(groups_cons[g]); continue
            if restrict is not None and g == T.OUTFIT and missing:
                # costume cannot satisfy the outfit constraints: take an unrestricted donor instead
                r2, m2 = self.pick_donor(rng, g, groups_cons[g], ctx, None)
                if r2 is not None and len(m2) < len(missing):
                    r, missing = r2, m2
            donors[g] = r
            self._take(ctx, r, self._group_role(g))
            missing_all[g] = missing
            committed.update(int(x) for x in self.group_tags(r, g))
            if coupled and g == T.ACTION and T.PLACE not in donors:
                # no row satisfies ACTION and PLACE jointly: PLACE gets its own donor
                rp, mp = self.pick_donor(rng, T.PLACE, groups_cons[T.PLACE], ctx, None)
                if rp is not None:
                    donors[T.PLACE] = rp; missing_all[T.PLACE] = mp
                    self._take(ctx, rp, T.GROUP_NAMES[T.PLACE])
                    committed.update(int(x) for x in self.group_tags(rp, T.PLACE))
                else:
                    missing_all[T.PLACE] = list(groups_cons[T.PLACE])
            if g != T.APPEAR and T.APPEAR not in donors and groups_cons[g]:
                for o in self.row_occ(r):
                    if a['occ_group'][o] == g and not a['occ_leak'][o] and \
                            any(self.occ_realizes(o, c) for c in groups_cons[g]) and \
                            T._PLAIN_ANIMAL_EARS.search(self.phr[int(a['occ_pid'][o])]):
                        for t in self.occ_tags(o):
                            t = int(t)
                            if a['home'][t] == T.APPEAR and not any(s['fid'] == t for s in soft + tagc):
                                soft.append(dict(key=-1 - len(soft), text=self.ft[t], kind='tag', fid=t, must=None))

        self._cur_look = look
        # ---- donor items, with the always-on cross-group consistency
        items = []
        realized_at = {}           # constraint key -> first item that realizes it
        appear_parts, appear_comp = set(), set()
        src = []           # (group, donor row, appearance slot or None)
        for g in GROUPS:
            if g in donors:
                src += [(g, r, s) for s, r in sorted(look.items())] if g == T.APPEAR and look else \
                    [(g, donors[g], None)]
        body_kinds = set()          # mixed look: the body traits the BODY donor writes
        for g, ra, s in src:
            if g != T.APPEAR or (look and s != BODY):
                continue
            for o in self.row_occ(ra):
                if a['occ_group'][o] == T.APPEAR and not a['occ_leak'][o] and \
                        not (a['occ_ident'][o] and ra not in char_set) and (s is None or self._occ_slot[o] == s):
                    pt = self.phr[int(a['occ_pid'][o])]
                    appear_parts.update(m.group(1).rstrip('s') for m in BODY_PART_RX.finditer(pt))
                    appear_comp.update(_animal_parts(pt))
                    body_kinds |= trait_kinds(pt)
        assigned_ok = {c['key'] for c in tagc if c.get('group') in donors and c not in missing_all.get(c['group'], [])}
        for g, r, s in src:
            own_char = r in char_set          # series / character vocabulary is fine on the same character
            for o in self.row_occ(r):
                if a['occ_group'][o] != g or a['occ_leak'][o] or (a['occ_ident'][o] and not own_char) or \
                        (s is not None and self._occ_slot[o] != s):
                    continue
                p = int(a['occ_pid'][o])
                hit_any = [c for c in tagc if self.occ_realizes(o, c)]
                # a constraint protects its phrase in the group (appearance slot) it was assigned to; elsewhere only
                # when its own group's donor does not write it
                hit = [c for c in hit_any if (c.get('group') == g and c.get('slot', s) == s) or
                       (c['key'] not in assigned_ok and c['key'] not in realized_at)]
                pt = self.phr[p]
                if self._cur_drop and self.mark_of(p) in self._cur_drop and not any(c.get('mark') for c in hit):
                    continue                     # watermark / signature / corner text (overlay_text=False)
                if s is not None and not hit and is_dependent(pt) and \
                        not (o > a['row_off'][r] and self._occ_slot[o - 1] == s):
                    continue                     # 'matching ...' whose antecedent is another slot's (not here)
                if not hit and g == T.APPEAR and re.match(r'(wearing|dressed|clad)\b', pt):
                    continue                     # an outfit sentence inside a hair phrase belongs to the outfit donor
                if need_boy and not hit and FEMALE_RX.search(pt):
                    continue
                if not hit and g in (T.OUTFIT, T.ACTION, T.PLACE):
                    # garments belong to the outfit donor, ears/tail/horns/wings/halo (and which animal they
                    # belong to) to the appearance donor
                    if look:        # body traits ('bushy grey tail behind her') only those of the BODY donor
                        extra = set() if _ACCESSORY_RX.search(pt) else trait_kinds(pt) - body_kinds
                    else:
                        extra = {m.group(1).rstrip('s') for m in BODY_PART_RX.finditer(pt)} - appear_parts
                    if (g in T.SCENE_GROUPS and a['pfac'][p] & GARMENT_BITS) or extra \
                            or (_animal_parts(pt) - appear_comp):
                        continue
                it = self._item_occ(o, [c['key'] for c in hit], 'donor')
                it['group'] = g
                it['slot'] = s
                it['role'] = SLOT_NAMES[s] if s is not None else self._group_role(g)
                for c in hit:
                    realized_at.setdefault(c['key'], it)
                items.append(it)
        if self._cur_drop and T.TAIL in donors and \
                not any(x['group'] == T.TAIL and x['fac'] & T.FBIT['style'] for x in items):
            # the tail donor's style phrases were all marks: the style tail of another donor without marks (of none
            # of the other donor roles' named characters, if possible)
            avoid = self._other_ids('tail') if ctx['distinct'] else frozenset()
            r2, shared = self._mark_free_tail(rng, used, avoid)
            if shared:
                warn.append(f'tail: no row of another character has a mark-free style tail; two donor roles come '
                            f'from {", ".join(sorted(self._named(r2, self._cur_exempt) & avoid))}')
            if r2 is not None:
                items = [x for x in items if x['group'] != T.TAIL or x['keys']]
                items += [dict(self._item_occ(o, [], 'donor'), role='tail') for o in self.row_occ(r2)
                          if a['occ_group'][o] == T.TAIL and not a['occ_leak'][o] and not a['occ_ident'][o]]
                donors[T.TAIL] = r2; used.append(r2)
                if ctx['distinct']:
                    self._reg('tail', r2)
        # single-valued attributes: hair / eye colour from APPEARANCE only, shot / framing from ACTION only
        for gown, bit in ((T.APPEAR, T.FBIT['hair']), (T.APPEAR, T.FBIT['eyes']), (T.ACTION, T.FBIT['shot'])):
            if any(x['fac'] & bit for x in items if x['group'] != gown and x['keys']):
                # a constraint phrase from another donor already describes it ('smiling ... with red eyes and long
                # wavy blonde hair'): the owner's unprotected description goes
                items = [x for x in items if x['group'] != gown or x['keys'] or not (x['fac'] & bit)]
            if any(x['fac'] & bit for x in items if x['group'] == gown):
                # a mixed look's hair / eye phrases are no shot ('hair with bangs framing her face')
                items = [x for x in items if x['group'] == gown or x['keys'] or not (x['fac'] & bit) or
                         x.get('slot') in (HAIR, EYES)]

        # ---- patches for constraints the donors miss
        patched, unreal, clash = [], [], []
        for g in GROUPS:
            for c in missing_all.get(g, []):
                if c['key'] in realized_at:
                    continue
                it = self._patch(rng, c, items, ctx, realized_at, warn)
                if it == 'clash':
                    clash.append(c); continue        # reported, nothing inserted (it would contradict)
                if it is None:
                    unreal.append(c); continue
                patched.append(c['text'])
        # ---- tags with no statistics / no usable group: find a phrase that writes them
        for c in lost:
            ct = T.content_tokens(c['text'])
            have = next((x for x in items if T.match_tokens(ct, T.tokens(x['text']))), None)
            if have is not None:          # a donor phrase already writes it
                have['keys'].add(c['key']); realized_at[c['key']] = have; continue
            occ = self.text_occurrences(ct, c.get('fid') if c.get('fid') is not None else self.fid.get(c['text']))
            if look and occ.size:       # a mixed look: not a phrase describing two appearance parts, if possible
                pure = occ[~np.isin(self._occ_slot[occ], (MIXED, MIXED_BODY))]
                occ = pure if pure.size else occ
                if not ctx['distinct']:
                    # ... and no appearance phrase of a character the look's donors already are
                    sep, other = self._sep_ident(occ, None)
                    keep = occ[(a['occ_group'][occ] != T.APPEAR) | np.isin(occ, sep)]
                    if keep.size:
                        occ = keep
                    elif other:
                        warn.append(f'{c["text"]}: only rows of {", ".join(sorted(other))} write it; two appearance '
                                    f'parts come from one character')
            pools = [occ]
            if ctx['distinct'] and occ.size:
                # ... from a row of no named character of another donor role than the one the phrase joins (its
                # group / look slot; the requested character excepted), if such a phrase realizes it
                keep = self._lost_sep(occ)
                if 0 < keep.size < occ.size:
                    pools = [keep, occ]
            if occ.size:
                for k, pool in enumerate(pools):
                    last = k + 1 == len(pools)
                    w = warn if last else []
                    pool, status = self._agree(pool, items, c, w)
                    o = self.choose_occ(rng, pool, ctx, c=c) if status == 'ok' else None
                    if o is not None or last:
                        if not last:
                            warn += w
                        break
                if status == 'clash':
                    clash.append(c); continue
                if status == 'text':
                    unreal.append(c); continue
                if o is None:
                    unreal.append(c); continue
                it = self._item_occ(o, [c['key']], 'patch')
                it['role'] = self._occ_role(o)
                if ctx['distinct']:
                    row = int(a['occ_row'][o])
                    shared = self._named(row, self._cur_exempt) & self._other_ids(it['role'])
                    if shared:
                        warn.append(f'{c["text"]}: only rows of {", ".join(sorted(shared))} write it; two donor '
                                    f'roles come from one character')
                    self._reg(it['role'], row)
                self._insert(it, items, realized_at)
                committed.update(int(x) for x in self.occ_tags(o))
                patched.append(c['text'])
            else:
                unreal.append(c)
        # ---- subject
        S = set(committed)
        # subject-type tags (horse girl, monster girl) of the appearance donor; of the BODY donor when mixed
        for r in ([look[BODY]] if BODY in look else []) if look else ([donors[T.APPEAR]] if T.APPEAR in donors else []):
            S.update(int(j) for j in a['row_tag_flat'][a['row_tag_off'][r]:a['row_tag_off'][r + 1]]
                     if a['home'][j] == T.SUBJ)
        subj_real = {}
        if char:
            subject = self.subject_of[char]
            if char in self.multi_subject:
                warn.append(f'the only subject phrase of {char} in the data names more than one character')
            subj_real[charc[0]['key']] = subject
            for c in cons:
                if c['kind'] == 'series' and c.get('via'):
                    subj_real[c['key']] = subject           # the explicit character is from this series
            if need_boy:
                warn.append('1boy with a character constraint: the character subject is kept')
            for c in subjc:
                warn.append(f'{c["text"]}: a character constraint already fixes the subject')
        elif subjc:
            subject = subjc[0]['subject']
            subj_real[subjc[0]['key']] = subject
            for c in subjc[1:]:
                warn.append(f'{c["text"]}: only one subject per caption')
            if need_boy:
                warn.append('boy subject: the solo data has (almost) no male rows; the caption body comes from girl donors')
                for c in cons:
                    if c['kind'] == 'count' and c not in count_conflict:
                        subj_real[c['key']] = subject
        else:
            subject = self.pick_subject(rng, S, [c['fid'] for c in subj_cons], warn, need_boy)
            for c in cons:
                if c['kind'] == 'count' and c not in count_conflict:
                    subj_real[c['key']] = subject
        real_subj = {j for gsub in self.m['generic'] if gsub['text'] == subject for j in gsub['real']}
        for c in subj_cons:
            if not char and c['fid'] in real_subj:
                subj_real[c['key']] = subject
            else:
                unreal.append(c)
        # ---- verbatim phrases: free phrases, then tag-text fallback for what nobody writes
        for c in rawc:
            self._insert(self._item_text(c['text'], [c['key']], 'raw'), items, realized_at)
        for c in unreal:
            self._insert(self._item_text(T.tag_text(c['text']), [c['key']], 'tag'), items, realized_at)
        if look:            # one source per body-trait kind (no two different wings)
            items = self._one_trait_source(items, look, warn)
        # ---- one backdrop: two different locations never stay together
        items = self._one_backdrop(items, donors, rng, ctx, warn)
        # ---- order, de-duplicate (merging constraint keys), trim
        tail = [x for x in items if x['group'] == T.TAIL]
        body_items = [x for x in items if x['group'] != T.TAIL]
        seqs = []
        for g in GROUPS:
            if g == T.TAIL:
                continue
            seq = [x for x in body_items if x['group'] == g and x['origin'] == 'donor']
            if seq:
                seqs.append(seq)
        seqs += [[x] for x in body_items if x['origin'] != 'donor']
        ordered = self._merge(seqs)
        tail.sort(key=lambda x: (x['origin'] != 'donor', self.fac_pos.get(x['prim'], 0.9)))
        items = self._dedupe(ordered + tail, subject)
        # hair / eyes: after every step that drops phrases, so a cue that hides a part is still in the caption
        parts = {}
        if look:
            items = self._look_parts(items, cons, length, rng, parts, look, warn, patched, need_boy)
        items = self._trim(items, length, subject)
        # ---- final validation: every reported realization must be in the final text
        holders, final_unreal = {}, []
        present = {id(x) for x in items}
        for c in cons:
            k = c['key']
            if c in count_conflict or c.get('conflict'):
                final_unreal.append(c['text']); continue
            if c['kind'] in ('implicit', 'rejected', 'dropped') or k in subj_real:
                continue
            if c['kind'] in ('char', 'subject', 'series') or c in clash:
                final_unreal.append(c['text']); continue
            # the holder must still EXPRESS the constraint (matcher), not just carry its key
            holder = next((x for x in items if k in x['keys'] and self.item_realizes(x, c)), None)
            if holder is None:
                holder = next((x for x in items if self.item_realizes(x, c)), None)
                if holder is not None:
                    holder['keys'].add(k)
            if holder is None:
                # repair: the original realizing phrase was trimmed or merged away -> put it back
                orig = realized_at.get(k)
                if orig is not None and id(orig) not in present and self.item_realizes(orig, c):
                    orig['keys'].add(k)
                    items = self._reinsert(items, orig)
                    present.add(id(orig))
                    holder = orig
            if holder is None:
                final_unreal.append(c['text']); continue
            if c in unreal or c in clash:
                final_unreal.append(c['text'])
            holders[c['text']] = holder
        live = {x.get('slot') for x in items}       # slots whose donor still writes a phrase
        if need_boy and not char:
            for x in items:
                x['text'] = _his(x['text'])
        realized = {t: h['text'] for t, h in holders.items()}
        realized.update({byk[k]['text']: s for k, s in subj_real.items()})
        trow = None
        if order == 'template' and items:
            # the last step, after every content decision: same phrase set as order='facet' for the same seed
            trow = self._pick_template(items, rng)
            items = self._order_template(items, trow)
        text = ', '.join([subject] + [x['text'] for x in items])
        self._cur_items = items         # generate()'s exclusion check (_violations) reads the final phrase items
        for c in cons:   # the text check behind the explanation
            if c['text'] in realized and realized[c['text']] not in text:
                final_unreal.append(c['text']); realized.pop(c['text'])
        picked = next((c['char'] for c in cons if c['kind'] == 'series' and c.get('char')), None)
        dn = {}
        for g, r in sorted(donors.items()):
            if g == T.APPEAR and look:
                dn.update({SLOT_NAMES[s]: self.ids[r2] for s, r2 in sorted(look.items()) if s in live})
            else:
                dn[T.GROUP_NAMES[g]] = self.ids[r]
        return Sample(text=text, length=length, seed=None, picked_character=picked if picked == char else None,
                      dropped=[{'text': c['text'], 'reason': c['reason']} for c in cons if c['kind'] == 'dropped'],
                      approx_tokens=self.approx_tokens(text, length),
                      template=None if trow is None else self.ids[trow],
                      donors=dn, subject=subject,
                      realized=realized, patched=[p for p in patched if p in realized and p not in final_unreal],
                      unrealized=sorted(set(final_unreal)), raw=[c['text'] for c in rawc], warnings=warn,
                      rejected=rejected,
                      phrases=[(x['text'], T.GROUP_NAMES[x['group']], x['src'], bool(x['keys'])) for x in items],
                      look_parts={SLOT_NAMES[s]: v for s, v in parts.items()}, coupled=self._cur_coupled,
                      sources=sorted({(x['role'], int(x['src'])) for x in items
                                      if x.get('role') is not None and _as_int(x['src']) is not None}))

    # ------------------------------------------------------------ patching
    def _patch(self, rng, c, items, ctx, realized_at, warn):
        """Realize constraint c with one phrase from another row. Patches for single-valued attributes (hair,
        eye colour) must agree with the protected phrase already describing them, or replace it with one phrase
        that realizes both. The patch row is a source of the constraint's donor role."""
        it = self._patch_pick(rng, c, items, ctx, realized_at, warn)
        if isinstance(it, dict):
            it['role'] = self._cons_role(c)
            if self._cur_distinct:
                self._reg(it['role'], int(self.a['occ_row'][it['occ']]))
        return it

    def _patch_pick(self, rng, c, items, ctx, realized_at, warn):
        a = self.a
        occ = self.key_occ(c)
        if occ.size == 0:
            return None
        ing = occ[a['occ_group'][occ] == c.get('group', -1)]
        if ing.size:
            occ = ing
        if c.get('slot') is not None:
            # a mixed look's slot constraint: a pure phrase of its slot ('multicolored feathered wings', not 'long
            # multicolored hair ..., multicolored feathered wings', which would bring a second hair description)
            pure = occ[self._occ_slot[occ] == c['slot']]
            if pure.size:
                occ = pure
        slot = c.get('slot') if self._cur_look else None
        if self._cur_distinct:
            # ... of no named character of the other donor roles (all groups, the other slots; the requested
            # character excepted); if no such phrase realizes it (agreeing with the hair / eyes already chosen),
            # one of theirs, warned
            role = self._cons_role(c)
            other = self._other_ids(role)
            if other:
                keep = occ[~np.isin(a['occ_row'][occ], self._excl_rows(other))]
                if keep.size == occ.size:
                    other = set()
                elif keep.size:
                    w = []
                    it = self._patch_occ(rng, c, keep, items, ctx, realized_at, w)
                    if it is not None and it != 'clash':
                        warn += w
                        return it
            if other:
                it = self._patch_occ(rng, c, occ, items, ctx, realized_at, warn)
                shared = self._named(int(a['occ_row'][it['occ']]), self._cur_exempt) & other \
                    if isinstance(it, dict) else set()
                if shared:
                    look_ids = {k for sl in SLOT_NAMES if sl != role for r in self._cur_src.get(sl, ())
                                for k in self._named(r, self._cur_exempt)}
                    what = 'two appearance parts' if slot is not None and shared <= look_ids else 'two donor roles'
                    warn.append(f'{c["text"]}: only rows of {", ".join(sorted(shared))} write it; {what} come from '
                                f'one character')
                return it
        elif slot is not None:
            # ... of another character than the other slots' donors
            other = self._other_ids(None, slot)
            if other:
                keep = occ[~np.isin(a['occ_row'][occ], self._excl_rows(other))]
                if keep.size:
                    occ = keep
                else:
                    warn.append(f'{c["text"]}: only rows of {", ".join(sorted(other))} write it; two appearance '
                                f'parts come from one character')
        return self._patch_occ(rng, c, occ, items, ctx, realized_at, warn)

    def _patch_occ(self, rng, c, occ, items, ctx, realized_at, warn):
        """_patch's choice among the candidate occurrences occ: agreeing with the hair / eyes already chosen."""
        a = self.a
        prims = np.array([T.PRIMARY_ORDER[x] for x in a['pprim'][a['occ_pid'][occ]]])
        for F in ('hair', 'eyes'):
            mask = prims == F
            if not mask.any():
                continue
            kept = [x for x in items if x['prim'] == F and x['keys']]
            if not kept:
                break
            others = [self._cur_cons.get(k) for x in kept for k in x['keys']]
            others = [o for o in others if o is not None and o.get('fid') is not None]
            joint = occ[mask]
            for oc in others:
                joint = joint[np.isin(joint, self.key_occ(oc))]
            o = self.choose_occ(rng, joint, ctx, c=c) if joint.size else None
            if o is not None:
                # one phrase realizes the new and the already protected hair constraints: it replaces them
                it = self._item_occ(o, [c['key']] + [oc['key'] for oc in others], 'patch')
                for x in kept:
                    items.remove(x)
                self._insert(it, items, realized_at, replace=True)
                self._warn_mixed_patch(c, o, warn)
                return it
            ref = set().union(*(_attrs(x['text'])[0] for x in kept)), set().union(*(_attrs(x['text'])[1] for x in kept))
            comp = np.array([_compatible(_attrs(self.phr[int(a['occ_pid'][o])]), ref) for o in occ])
            if comp.any():
                occ = occ[comp]
                break
            if _compatible(_attrs(T.tag_text(c['text'])), ref):
                warn.append(f'{c["text"]}: no data phrase agrees with the {F} already chosen; tag text used')
                return None
            warn.append(f'{c["text"]}: contradicts the {F} already chosen ({", ".join(sorted(ref[0] | ref[1]))}); '
                        f'not realized')
            return 'clash'
        o = self.choose_occ(rng, occ, ctx, avoid={c['fid']}, c=c)
        if o is None:
            return None
        it = self._item_occ(o, [c['key']], 'patch')
        self._insert(it, items, realized_at)
        ctx['committed'].update(int(x) for x in self.occ_tags(o))
        self._warn_mixed_patch(c, o, warn)
        return it

    def _warn_mixed_patch(self, c, o, warn):
        """A mixed look's slot constraint patched by a phrase that also describes another part (no pure phrase
        exists): _insert dropped the other hair / eye descriptions, _one_trait_source the other body traits."""
        if c.get('slot') is not None and self._occ_slot[o] != c['slot']:
            r = int(self.a['occ_row'][o])
            who = ', '.join(sorted(self._row_ids(r))) or f'row {self.ids[r]}'
            warn.append(f'{c["text"]}: only a phrase that also describes another appearance part writes it ("'
                        f'{self.phr[int(self.a["occ_pid"][o])]}", {who}); it replaces that part')

    def _look_parts(self, items, cons, length, rng, parts, look, warn, patched, male=False):
        """Mixed look, after every step that drops phrases and before trimming: is the hair / are the eyes in this
        caption? A constraint phrase that names the part (an eye colour, a hairstyle) -> 'constrained', and an
        unprotected cue phrase about her goes. Else the part is wanted with the real rate of its length (look_rate;
        long always, medium hair always, eyes given the hair); a wanted part that a cue about her hides (main_cue:
        'closed eyes', a requested 'blindfold', 'hair hidden under a hood'; not the EYES donor's own eye state) ->
        'hidden': the slot's unprotected phrases go, the cue is kept through trimming; a wanted part -> 'shown': one
        phrase that names it is kept through trimming (one with a colour word, then the shortest; an 'eyepatch'
        constraint does not name the eyes); not wanted -> 'omitted': the slot's unprotected phrases go. A requested
        cue ('closed eyes') and a requested colour both stay; next to a requested cue about her, a constraint phrase
        that names the eyes without an eye-colour request ('eyelashes' as "red eyes with long eyelashes") is replaced
        by a colourless phrase that realizes the same constraints (warning if none exists). A wanted part whose donor
        phrases did not survive is repaired from another eligible row (look[slot] is updated)."""
        asked = any(_EYE_COLOUR_RX.search(c['text']) for c in cons if c.get('text'))
        for slot, bit, rx in ((HAIR, T.FBIT['hair'], HAIR_HIDDEN_RX), (EYES, T.FBIT['eyes'], EYES_HIDDEN_RX)):
            cues = [x for x in items if (x.get('slot') != slot or x['keys']) and main_cue(x['text'], rx, male)]
            ids = {id(x) for x in cues}
            prot = [x for x in items if x['keys'] and (x.get('slot') == slot or x.get('fac', 0) & bit) and
                    names_part(x['text'], slot, male)]
            if prot and slot == EYES and not asked and any(x['keys'] for x in cues):
                # her closed eyes were asked for, an eye colour was not: no colour from the EYES donor, and a
                # constraint phrase with a colour gives way to a colourless one that realizes the same constraints
                items = [x for x in items if x['keys'] or x.get('slot') != EYES or not _EYE_COLOUR_RX.search(x['text'])]
                for x in prot:
                    if _EYE_COLOUR_RX.search(x['text']):
                        alt = self._colourless(x, rng, warn)
                        if alt is None:
                            warn.append(f'eye colour kept next to "{next(c["text"] for c in cues if c["keys"])}": '
                                        f'only phrases with an eye colour write "{x["text"]}"\'s constraints')
                        else:
                            items = [alt if y is x else y for y in items]
                            patched += [self._cur_cons[k]['text'] for k in sorted(x['keys']) if k in self._cur_cons]
                prot = [x for x in items if x['keys'] and (x.get('slot') == slot or x.get('fac', 0) & bit) and
                        names_part(x['text'], slot, male)]
                if not prot:            # 'heavy black eyeliner' next to the requested 'closed eyes'
                    items = self._drop_part(items, slot, ids, male)
                    parts[slot] = 'hidden'
                    continue
            if prot:
                items = [x for x in items if x['keys'] or id(x) not in ids]
                parts[slot] = 'constrained'
                continue
            r = self.look_rate.get(length, {}).get(slot)
            if isinstance(r, tuple):
                r = r[0] if parts.get(HAIR) in ('shown', 'constrained') else r[1]
            if r is not None and rng.random() >= r:
                parts[slot] = 'omitted'
            elif cues:
                parts[slot] = 'hidden'
                for x in cues:
                    x['cue'] = True
            else:
                own = [x for x in items if x.get('slot') == slot and names_part(x['text'], slot, male)]
                if not own:         # the donor's description did not survive: another eligible row's
                    rep = self._repair_slot(slot, look, rng)
                    if rep is not None:
                        items = [x for x in items if x.get('slot') != slot or x['keys']]
                        items = self._reinsert(items, rep)
                        own = [rep]
                if own:
                    min(own, key=lambda x: (not _COLOR_RX.search(x['text']), x['words']))['need'] = True
                parts[slot] = 'shown' if own else 'missing'
                continue
            items = self._drop_part(items, slot, ids, male)
        return items

    @staticmethod
    def _drop_part(items, slot, cue_ids, male):
        """A hidden / omitted part: the slot's unprotected phrases go, and so does any other unprotected phrase that
        describes the part (an action donor's 'hazel eyes', a style tail's 'heterochromatic ... eyes'); cues stay."""
        return [x for x in items if x['keys'] or id(x) in cue_ids or
                (x.get('slot') != slot and not names_part(x['text'], slot, male))]

    def _sep_ident(self, occ, slot):
        """Mixed look: the occurrences whose rows share no character identity with the other donor roles (slot: the
        look slot they are for; None: every role; distinct_donor_chars=False: only the other slots' donors), and
        those identities."""
        other = self._other_ids(SLOT_NAMES[slot] if slot is not None else None, slot) if self._cur_look else set()
        if not other or not occ.size:
            return occ, other
        return occ[~np.isin(self.a['occ_row'][occ], self._excl_rows(other))], other

    def _colourless(self, x, rng, warn):
        """A pure EYES phrase without an eye colour or a cue that realizes every constraint of item x (full matcher),
        as a patch item, from a row of another character than the other slots' donors (if only such rows write one:
        that row, with the warning _patch gives); None if no row writes one."""
        cs = [self._cur_cons[k] for k in sorted(x['keys']) if k in self._cur_cons]
        if not cs:
            return None
        a = self.a
        occ = self.key_occ(cs[0])
        occ = occ[(self._occ_slot[occ] == EYES) & ~a['occ_leak'][occ].astype(bool) & ~a['occ_ident'][occ].astype(bool)]
        sep, other = self._sep_ident(occ, EYES)
        for pool in ((sep, occ) if sep.size < occ.size else (occ,)):
            seen, cand = set(), []
            for o in pool.tolist():
                p = int(a['occ_pid'][o])
                if p in seen:
                    continue
                seen.add(p)
                t = self.phr[p]
                if _EYE_COLOUR_RX.search(t) or EYES_HIDDEN_RX.search(t) or is_dependent(t) or self.mark_of(p) or \
                        _OTHER_FIG_RX.search(t) or _OTHER_PRON_RX[False].search(t):     # not 'a portrait of ...'
                    continue
                it = self._item_occ(o, x['keys'], 'patch')
                if all(self.item_realizes(it, c) for c in cs):
                    cand.append(it)
            if cand:
                if pool is not sep and other and sep.size < occ.size:
                    warn.append(f'{", ".join(c["text"] for c in cs)}: only rows of {", ".join(sorted(other))} write '
                                f'it without an eye colour; two appearance parts come from one character')
                it = cand[int(rng.integers(len(cand)))]
                it['group'], it['slot'], it['role'] = T.APPEAR, EYES, SLOT_NAMES[EYES]
                if self._cur_distinct:
                    self._reg(it['role'], int(self.a['occ_row'][it['occ']]))
                return it
        return None

    def _repair_slot(self, slot, look, rng):
        """An eligible row for a HAIR / EYES slot whose donor phrases were all filtered: not used by the look and no
        shared character identity; its best phrase that names the part becomes the slot item, look[slot] = the row."""
        cand = self.slot_rows[slot]
        if self._cur_distinct:      # no row / named character of another donor role (groups too)
            taken = self._all_source_rows()
            ids = self._other_ids(SLOT_NAMES[slot])
        else:
            taken = set(look.values())
            others = [r for s, r in look.items() if s != slot]
            ids = set().union(*[self._row_ids(r) for r in others]) if others else set()
        ex = self._cur_exempt if self._cur_distinct else None
        for _ in range(32):
            r = int(cand[int(rng.integers(len(cand)))])
            if r in taken or self._named(r, ex) & ids:
                continue
            occ = [o for o in self.row_occ(r) if self._slot_good[slot][o]]
            if not occ:
                continue
            o = min(occ, key=lambda o: (not _COLOR_RX.search(self.phr[int(self.a['occ_pid'][o])]),
                                        int(self.a['pwords'][int(self.a['occ_pid'][o])])))
            it = self._item_occ(o, [], 'donor')
            it['group'], it['slot'], it['role'] = T.APPEAR, slot, SLOT_NAMES[slot]
            look[slot] = r
            if self._cur_distinct:
                self._reg(it['role'], r)
            return it
        return None

    def _one_trait_source(self, items, look, warn=None):
        """Mixed look: one source per body-trait kind (wings, tail, ears, horns, halo). A constraint phrase wins, else
        the BODY donor; other unprotected phrases describing that kind go ('small white angel wings' next to a
        patched 'multicolored feathered wings'). Accessories ('cat ear headband'), garment tails ('jacket with ruffled
        tails') and the HAIR / EYES slots' phrases ('hair tied in a low tail') are not body traits."""
        body = self.ids[look[BODY]] if BODY in look else None
        kinds = {}
        for x in items:
            if x['group'] != T.TAIL and x.get('slot') not in (HAIR, EYES) and not _ACCESSORY_RX.search(x['text']):
                for k in trait_kinds(x['text']):
                    kinds.setdefault(k, []).append(x)
        drop = set()
        for k, xs in kinds.items():
            xs = [x for x in xs if id(x) not in drop]
            srcs = {x['src'] for x in xs}
            if len(srcs) < 2:
                continue
            # constraint phrases of two rows: one that also expresses the other's constraints takes them over
            # ('multicolored feathered wings' realizes 'wings' too: 'small white wings at her back' goes)
            prot = [x for x in xs if x['keys']]
            for y in sorted(prot, key=lambda x: (len(x['keys']), x.get('words', 0))):
                hosts = [x for x in prot if x is not y and id(x) not in drop and x['src'] != y['src'] and
                         all(q in self._cur_cons and self.item_realizes(x, self._cur_cons[q]) for q in y['keys'])]
                if hosts:
                    host = max(hosts, key=lambda x: (len(x['keys']), x.get('words', 0)))
                    host['keys'] |= y['keys']
                    drop.add(id(y))
            prot = [x for x in prot if id(x) not in drop]
            if len({x['src'] for x in prot}) > 1 and warn is not None:
                warn.append(f'two {k} descriptions kept ("{prot[0]["text"]}", "{prot[1]["text"]}"): each realizes a '
                            f'constraint the other does not')
            win = {x['src'] for x in prot} or ({body} if body in srcs else {xs[0]['src']})
            drop |= {id(x) for x in xs if not x['keys'] and x['src'] not in win}
        return [x for x in items if id(x) not in drop]

    def _agree(self, occ, items, c, warn):
        """Keep only candidate phrases whose hair / eye description agrees with the protected one already chosen.
        Returns (occ, 'ok' | 'text' (use the compatible tag text) | 'clash')."""
        a = self.a
        prims = np.array([T.PRIMARY_ORDER[x] for x in a['pprim'][a['occ_pid'][occ]]])
        for F in ('hair', 'eyes'):
            mask = prims == F
            kept = [x for x in items if x['prim'] == F and x['keys']]
            if not mask.any() or not kept:
                continue
            ref = set().union(*(_attrs(x['text'])[0] for x in kept)), set().union(*(_attrs(x['text'])[1] for x in kept))
            comp = np.array([not m or _compatible(_attrs(self.phr[int(a['occ_pid'][o])]), ref)
                             for o, m in zip(occ, mask)])
            if comp.any():
                return occ[comp], 'ok'
            if _compatible(_attrs(T.tag_text(c['text'])), ref):
                warn.append(f'{c["text"]}: no data phrase agrees with the {F} already chosen; tag text used')
                return occ[:0], 'text'
            warn.append(f'{c["text"]}: contradicts the {F} already chosen ({", ".join(sorted(ref[0] | ref[1]))}); '
                        f'not realized')
            return occ[:0], 'clash'
        return occ, 'ok'

    def _insert(self, it, items, realized_at, replace=False):
        """Add an inserted (patch / raw / fallback) phrase; for exclusive facets it replaces the donors' unprotected
        phrases of that facet (no two hair colours, shots, lighting setups)."""
        if it['prim'] in EXCLUSIVE:
            items[:] = [x for x in items if x['keys'] or x['prim'] != it['prim']]
        # every single-valued facet the inserted phrase describes (a hair phrase that also names the eye colour)
        for bit in (T.FBIT['hair'], T.FBIT['eyes'], T.FBIT['shot']):
            if it.get('fac', 0) & bit:
                items[:] = [x for x in items if x['keys'] or not (x.get('fac', 0) & bit)]
        items.append(it)
        for k in it['keys']:
            realized_at[k] = it

    def _one_backdrop(self, items, donors, rng, ctx, warn):
        """At most one location. Backdrop phrases (overall background / 'in a <place>') may only come from one
        source. A requested location wins (its constraint's phrase); a constraint that is NOT a location (rain,
        umbrella) but whose phrase carries another location ('outdoors on a rainy street' next to a requested
        beach) is re-realized by a location-neutral phrase ('falling rain'); if none exists, it is reported.
        Otherwise priority: constraint-realizing > ACTION's place phrase > PLACE donor."""
        cons = self._cur_cons
        bd = [x for x in items if (x['group'] in T.SCENE_GROUPS or x['origin'] != 'donor') and T.is_backdrop(x['text'])]
        if len({x['src'] for x in bd}) <= 1:
            return items
        prot = [x for x in bd if x['keys']]
        locprot = [x for x in prot if any(_is_location(cons.get(k)) for k in x['keys'])]
        if locprot:
            win = {x['src'] for x in locprot}
            if len(win) > 1:
                warn.append('two requested locations kept (both were asked for)')
        elif prot:
            win = {prot[0]['src']}
        else:
            act = [x for x in bd if x['group'] == T.ACTION and x['origin'] == 'donor']
            win = {act[0]['src']} if act else {bd[0]['src']}
        out = list(items)
        for x in prot:
            if x['src'] in win:
                continue
            repl = []
            for k in x['keys']:
                c = cons.get(k)
                occ = self.key_occ(c) if c is not None and c['kind'] == 'tag' else np.zeros(0, dtype=np.int32)
                if occ.size > 400:
                    occ = rng.choice(occ, 400, replace=False)
                occ = np.array([o for o in occ if not T.is_backdrop(self.phr[int(self.a['occ_pid'][o])])],
                               dtype=np.int64)
                other = self._other_ids(x.get('role')) if self._cur_distinct and occ.size else set()
                o = None
                if other:       # of no named character of another donor role, if possible
                    keep = occ[~np.isin(self.a['occ_row'][occ], self._excl_rows(other))]
                    if 0 < keep.size < occ.size:
                        o = self.choose_occ(rng, keep, ctx, c=c)
                if o is None:
                    o = self.choose_occ(rng, occ, ctx, c=c) if occ.size else None
                    shared = self._named(int(self.a['occ_row'][o]), self._cur_exempt) & other \
                        if o is not None and other else set()
                    if shared:
                        warn.append(f'{c["text"]}: only rows of {", ".join(sorted(shared))} write it without a '
                                    f'location; two donor roles come from one character')
                if o is None:
                    repl = None
                    break
                it = self._item_occ(o, [k], 'patch')
                it['role'] = x.get('role')
                if self._cur_distinct:
                    self._reg(it['role'], int(self.a['occ_row'][o]))
                repl.append(it)
            if repl is None:
                warn.append(f'two locations kept: no location-neutral phrase for "{x["text"]}"')
                continue
            i = out.index(x)
            out[i:i + 1] = repl
        return [x for x in out if not (x in bd and not x['keys'] and x['src'] not in win)]

    def _dedupe(self, items, subject):
        """Exact duplicates (same words in the same order) and contiguous containment ('indoor library setting'
        inside 'indoor library setting with wooden bookshelves') are dropped. Constraint keys move to the kept
        phrase only if it really expresses them (full matcher); free phrases stay verbatim."""
        cons = self._cur_cons
        seen, uniq = {}, []
        for x in items:
            if x['text'] == subject:
                continue
            n = ' '.join(T.raw_tokens(x['text']))
            if n in seen:
                seen[n]['keys'] |= x['keys']
                continue
            seen[n] = x; uniq.append(x)
        toks = [' ' + ' '.join(T.raw_tokens(x['text'])) + ' ' for x in uniq]
        drop = set()
        for i in range(len(uniq)):
            if uniq[i].get('origin') == 'raw' or toks[i].count(' ') < 3:
                continue
            for j in range(len(uniq)):
                if i == j or j in drop or len(toks[j]) <= len(toks[i]) or toks[i] not in toks[j]:
                    continue
                if uniq[i].get('slot') in (HAIR, EYES) and uniq[j].get('slot') != uniq[i]['slot']:
                    continue        # 'white hair' is no duplicate of the outfit's 'small white hair clip'
                if all(self.item_realizes(uniq[j], cons[k]) for k in uniq[i]['keys'] if k in cons):
                    drop.add(i)
                    uniq[j]['keys'] |= uniq[i]['keys']
                    break
        return [x for i, x in enumerate(uniq) if i not in drop]

    def _reinsert(self, items, it):
        k = self.fac_pos.get(it['prim'], 0.6)
        pos = next((i for i, x in enumerate(items) if x['group'] == T.TAIL or self.fac_pos.get(x['prim'], 0.6) > k),
                   len(items))
        return items[:pos] + [it] + items[pos:]

    @staticmethod
    def _okey(x):
        return TAIL_KEY if x['group'] == T.TAIL else T.PRIMARY.get(x['prim'], T.PRIMARY['misc'])

    def _pick_template(self, items, rng):
        """Donor row whose caption gives the phrase order (self.order_pick, default ORDER_PICK)."""
        pick = self.order_pick
        if pick == 'random':
            return int(rng.integers(self.N))
        rows = [int(self.a['occ_row'][x['occ']]) for x in items if x['origin'] == 'donor' and x.get('occ') is not None]
        if pick == 'donor':
            if not rows:
                return int(rng.integers(self.N))
            u, cnt = np.unique(rows, return_counts=True)
            top = u[cnt == cnt.max()]
            return int(top[int(rng.integers(len(top)))])
        cand = rng.integers(self.N, size=K_TEMPLATE)
        nd = 0
        if pick == 'mix':
            own = np.unique(np.array(rows, dtype=np.int64)); nd = len(own)
            cand = np.concatenate([own, cand])
        v = np.bincount([self._okey(x) for x in items if x['group'] != T.TAIL], minlength=NK)[:TAIL_KEY]
        tc = self._t_cnt[cand]
        score = np.minimum(tc, v).sum(1) / np.maximum(np.maximum(tc, v).sum(1), 1)
        score[:nd] += DONOR_BONUS
        return int(cand[int(np.argmax(score))])

    def _order_template(self, items, r):
        """Order items like the caption of donor row r. The k-th item of facet X takes the k-th free X slot (the
        row's own phrases keep their slots); extras follow their donor predecessor, else the last X item, else
        X's learned position. The tail stays one block (donor order) at the start of the row's last tail run; body
        slots after it (lighting / setting after the style words) stay after it."""
        a = self.a
        s, e = int(a['row_off'][r]), int(a['row_off'][r + 1])
        keys = self._t_key[s:e]
        L = e - s
        if L == 0:
            return items
        anchor = L
        tl = np.nonzero(keys == TAIL_KEY)[0]
        if tl.size:
            anchor = int(tl[-1])
            while anchor > 0 and keys[anchor - 1] == TAIL_KEY:
                anchor -= 1
        body = [x for x in items if x['group'] != T.TAIL]
        tail = sorted((x for x in items if x['group'] == T.TAIL),
                      key=lambda x: (x['origin'] != 'donor', x['occ'] if x['origin'] == 'donor' else 0))
        pos, taken = {}, set()
        for x in body:          # the template row's own phrases keep their slots
            o = x.get('occ')
            if x['origin'] == 'donor' and o is not None and s <= o < e and keys[o - s] == self._okey(x):
                pos[id(x)] = float(o - s); taken.add(o - s)
        free = {}
        for i, k in enumerate(keys.tolist()):
            if k != TAIL_KEY and i not in taken:
                free.setdefault(k, []).append(i)
        extras = []
        at = {x['occ']: pos[id(x)] for x in body if id(x) in pos}
        for x in body:
            if id(x) in pos:
                continue
            f = free.get(self._okey(x))
            if f:
                o = x.get('occ') if x['origin'] == 'donor' else None
                p = at.get(o - 1) if o and a['occ_row'][o - 1] == a['occ_row'][o] else None
                # a free X slot right after its donor predecessor keeps the two together
                i = f.index(p + 1) if p is not None and p + 1 in f else 0
                pos[id(x)] = float(f.pop(i))
                if o is not None:
                    at[o] = pos[id(x)]
            else:
                extras.append(x)
        out = sorted((x for x in body if id(x) in pos), key=lambda x: pos[id(x)])

        def pred(x):        # x's predecessor in its donor caption, if it is in the output
            o = x.get('occ')
            if x['origin'] != 'donor' or o is None or o == 0 or a['occ_row'][o - 1] != a['occ_row'][o]:
                return None
            return next((i for i, y in enumerate(out) if y['origin'] == 'donor' and y.get('occ') == o - 1), None)
        def donor_pred(x, seq):     # index in seq of the nearest earlier phrase of x's donor caption
            o = x['occ']
            js = [i for i, y in enumerate(seq) if y['origin'] == 'donor' and y.get('occ') is not None and
                  y['occ'] < o and a['occ_row'][y['occ']] == a['occ_row'][o]]
            return max(js, key=lambda i: seq[i]['occ']) if js else None
        for x in extras:
            j = pred(x)
            if j is None:
                k = self._okey(x)
                j = next((i for i in range(len(out) - 1, -1, -1) if self._okey(out[i]) == k), None)
            if j is None:       # the template has no X: X's learned position, before the tail
                pos[id(x)] = min(self.fac_pos.get(x['prim'], 0.6) * L - 1, anchor - 0.5)
                out.insert(next((i for i, y in enumerate(out) if pos[id(y)] > pos[id(x)]), len(out)), x)
                continue
            pos[id(x)] = pos[id(out[j])]
            out.insert(j + 1, x)
        # a phrase that refers back ('matching cat tail', 'resting another hand on ...') follows the phrase before it
        # in its donor caption (the nearest earlier one present), in donor order so chains stay in order
        for x in sorted((x for x in out if is_dependent(x['text']) and x['origin'] == 'donor'), key=lambda x: x['occ']):
            out.remove(x)
            j = donor_pred(x, out)
            if j is None:
                out.insert(next((i for i, y in enumerate(out) if pos[id(y)] > pos[id(x)]), len(out)), x)
                continue
            pos[id(x)] = pos[id(out[j])]
            out.insert(j + 1, x)
        j = next((i for i, x in enumerate(out) if pos[id(x)] >= anchor), len(out))
        return out[:j] + tail + out[j:]

    def _merge(self, seqs):
        heads = [0] * len(seqs)
        out = []
        total = sum(len(s) for s in seqs)
        while len(out) < total:
            best, bk = None, None
            for i, s in enumerate(seqs):
                if heads[i] < len(s):
                    k = self.fac_pos.get(s[heads[i]]['prim'], 0.6)
                    if bk is None or k < bk:
                        best, bk = i, k
            out.append(seqs[best][heads[best]]); heads[best] += 1
        return out

    def _trim(self, items, length, subject):
        """Drop the facets with the lowest medium/short keep rate first (never a constraint phrase), keep at least
        FLOOR substantive phrases (appearance, outfit or action, place when available) and refill unused budget."""
        if length == 'long':
            return items
        budget = self.calib[length]['word_budget']
        keep = self.keep[length]
        words = len(subject.split()) + sum(x['words'] for x in items)
        if words <= budget:
            return items
        seen = {}
        pri = []
        for i, x in enumerate(items):
            k = seen.get(x['prim'], 0); seen[x['prim']] = k + 1
            pri.append(keep.get(x['prim'], 0.3) * (1.0 if k == 0 else 0.5 ** k))
        order = sorted(range(len(items)), key=lambda i: (pri[i], -i))
        drop = set()
        for i in order:
            if words <= budget:
                break
            if items[i]['keys'] or items[i].get('cue') or items[i].get('need'):    # hair / eyes kept (_look_parts)
                continue
            drop.add(i); words -= items[i]['words']
        # floor: one appearance, one outfit-or-action, one place (when the donors have them)
        kept = [i for i in range(len(items)) if i not in drop and items[i]['group'] != T.TAIL]
        for cat in ((T.APPEAR,), (T.OUTFIT, T.ACTION), (T.PLACE,)):
            if len(kept) >= FLOOR[length]:
                break
            if any(items[i]['group'] in cat for i in kept):
                continue
            cand = [i for i in drop if items[i]['group'] in cat]
            if cand:
                i = min(cand, key=lambda i: (-pri[i], items[i]['words']))
                drop.discard(i); kept.append(i); words += items[i]['words']
        while len(kept) < FLOOR[length]:
            cand = [i for i in drop if items[i]['group'] != T.TAIL]
            if not cand:
                break
            i = min(cand, key=lambda i: (-pri[i], items[i]['words']))
            drop.discard(i); kept.append(i); words += items[i]['words']
        # refill: dropped phrases that still fit into the budget, best keep rate first
        for i in sorted(drop, key=lambda i: (-pri[i], items[i]['words'])):
            if words + items[i]['words'] <= budget:
                drop.discard(i); words += items[i]['words']
        return [x for i, x in enumerate(items) if i not in drop]

    def generate(self, constraints, n=5, length='long', seed=None, beta=1.0, explain=False, wild=1.0,
                 order='template', overlay_text=False, scene_text=True, mix_look=True, distinct_donor_chars=True):
        """n distinct samples (distinct phrase sets). With a seed the run is reproducible; the RNG also mixes a
        digest of the request (normalized constraints, length, wild), so different requests with the same seed do not
        share donors. order: 'template' (default; the phrase order of a real donor caption, drawn after every content
        decision) or 'facet' (learned facet positions, one fixed order); both give the same phrases for a seed.
        overlay_text=False drops marks on the picture (watermark, artist signature, corner text, logo, handle) from
        every donor phrase group; scene_text=False also drops text on scene objects (shirt print, sign, speech
        bubble). A constraint asking for a mark keeps its phrase. mix_look=True takes a generic caption's hair, eyes
        and body traits from three different donor rows (different named characters); False = one appearance donor.
        distinct_donor_chars=True: a named character (costumes merged) gives at most one donor role per caption (each
        look slot, the appearance, outfit, action, place, tail donor; the coupled action + place row is one role);
        the requested character is exempt; only if no row of another character realizes a role's constraints is a
        character reused, with a warning. False = phase 4 (only the look's slots are kept apart)."""
        if isinstance(constraints, str):
            constraints = [x for x in constraints.split(',')]
        if length not in ('long', 'medium', 'short'):
            raise ValueError('length must be long, medium or short')
        if order not in ('template', 'facet'):
            raise ValueError('order must be template or facet')
        cons = self.parse(constraints)
        # an ignored exclusion (not supported, or also required) does not change the samples
        req = json.dumps([sorted(c['text'] for c in cons if c['kind'] not in ('dropped', 'exclude_unsupported') and
                                 not (c.get('neg') and c['kind'] == 'rejected')), length, round(float(wild), 4)])
        digest = int.from_bytes(hashlib.sha256(req.encode()).digest()[:8], 'little')
        excl = [c for c in cons if c['kind'] == 'exclude']
        self._cur_ban = self._ban_masks(excl) if excl else None
        self._cur_excl = excl
        out, seen, picked, discarded = [], set(), set(), []
        attempts = 0
        budget = max(4 * n, n + 8) + (32 if excl else 0)
        while len(out) < n and attempts < budget:
            sd = (seed, attempts) if seed is not None else None
            rng = np.random.default_rng([digest, seed, attempts] if seed is not None else None)
            self._cur_cons = {c['key']: c for c in cons}
            cs = [dict(c) for c in cons]
            pre = self._resolve_series(cs, rng, wild, picked)
            s = self._one(cs, rng, length, beta, wild, pre, order, overlay_text, scene_text, mix_look,
                          distinct_donor_chars)
            s.seed = sd
            attempts += 1
            if excl:
                s.excluded = [c['text'] for c in excl]
                bad = self._violations(s.subject, self._cur_items, excl)
                if bad:         # an excluded item is in the caption: another attempt (phrase removal at the end)
                    discarded.append((s, self._cur_items, bad))
                    continue
            key = (s.subject, tuple(sorted(x[0] for x in s.phrases)))      # independent of the order
            if key in seen:
                continue
            seen.add(key)
            if s.picked_character:
                picked.add(self.identity[s.picked_character])
            out.append(s)
        for s, items, bad in discarded:
            if len(out) >= n:
                break
            if any(i < 0 for i, _ in bad):
                continue        # the subject itself has it: no removal
            s = self._remove_phrases(s, items, bad)
            key = (s.subject, tuple(sorted(x[0] for x in s.phrases)))
            if key not in seen:
                seen.add(key)
                out.append(s)
        if not out and discarded:
            # every attempt's subject has an excluded item (a character's name, 'anime'): one sample, warned
            s, items, bad = discarded[0]
            s = self._remove_phrases(s, items, [(i, c) for i, c in bad if i >= 0])
            s.warnings += [f'excluded {c["text"][1:]}: in the subject "{s.subject}", not removed'
                           for i, c in bad if i < 0]
            out.append(s)
        self._cur_ban = None
        return out

    def _remove_phrases(self, s, items, bad):
        """Sample s (final phrase items `items`) without the phrases bad = [(item index, exclusion)] names: text
        rebuilt the way _one joins it, realizations whose phrase is gone -> unrealized, one warning per phrase."""
        drop = {i for i, _ in bad}
        keep = [i for i in range(len(items)) if i not in drop]
        text = ', '.join([s.subject] + [items[i]['text'] for i in keep])
        realized = {c: p for c, p in s.realized.items() if p in text}
        gone = [c for c in s.realized if c not in realized]
        warn = list(s.warnings)
        for i, c in bad:
            w = f'excluded {c["text"][1:]}: removed phrase "{items[i]["text"]}"'
            if w not in warn:
                warn.append(w)
        return dataclasses.replace(s, text=text, approx_tokens=self.approx_tokens(text, s.length), realized=realized,
                       patched=[p for p in s.patched if p in realized],
                       unrealized=sorted(set(s.unrealized) | set(gone)), warnings=warn,
                       phrases=[s.phrases[i] for i in keep])

    # ------------------------------------------------------------ exclusions ('-x' constraints)
    def _violations(self, subject, items, excl):
        """[(item index or -1 for the subject, exclusion)] for every output phrase that has an excluded item: (a) it
        comes from an occurrence indexed with the excluded tag (fid; a free-phrase exclusion: realizing the tag with
        its words, occ_realizes), (b) its text has the exclusion's words (excl_ct: the excluded text's and its tag's
        content tokens; T.match_tokens without the head-noun rule, so 'heart-shaped' has a heart), (c) its plain
        words do (excl_rx: 'hair' in 'hair ribbons', which the tokens merge into 'hairribbon')."""
        out = []
        for i, text in [(-1, subject)] + [(i, x['text']) for i, x in enumerate(items)]:
            o = items[i].get('occ') if i >= 0 else None
            toks = None
            for c in excl:
                if o is not None and c['fid'] is not None and \
                        (self.occ_realizes(o, c) if c['must'] else c['fid'] in self.occ_tags(o)):
                    out.append((i, c)); continue
                if toks is None:
                    toks = T.tokens(text)
                if any(T.match_tokens(ct, toks, head_rule=False) for ct in c['excl_ct']) or \
                        (c['excl_rx'] is not None and c['excl_rx'].search(text.lower())):
                    out.append((i, c))
        return out

    def _excl_occ(self, c):
        """Occurrences _violations would flag for exclusion c (the data phrases). Not cached here: only the bounded
        ban masks built from it are (_ban_masks), so a stream of distinct exclusions cannot grow any cache."""
        a, parts = self.a, []
        if c['fid'] is not None:
            f = c['fid']
            occ = a['to_flat'][a['to_off'][f]:a['to_off'][f + 1]]
            if c['must']:       # must_index without its (unbounded) cache
                occ = occ[np.array([T.match_tokens(c['must'], self._ptoks(int(a['occ_pid'][o]))) for o in occ],
                                   dtype=bool)] if occ.size else occ
            parts.append(occ)
        for ct in c['excl_ct']:
            ids = [self.m['tvocab'].get(w) for w in set(ct)]
            if any(i is None for i in ids):
                continue
            posts = sorted((a['tp_flat'][a['tp_off'][i]:a['tp_off'][i + 1]] for i in ids), key=len)
            pids = posts[0]
            for q in posts[1:]:
                pids = np.intersect1d(pids, q, assume_unique=True)
            if len(ct) > 1:
                pids = [int(p) for p in pids if T.match_tokens(ct, self._ptoks(int(p)), head_rule=False)]
            if len(pids):
                parts.append(_gather(a['po_off'], a['po_flat'], pids)[0])
        if c['excl_rx'] is not None:
            # plain words: candidate phrases from every index token that contains a word (or its normal form:
            # 'silver' -> grey), then the pattern on the phrase text
            sets = []
            for w in _plain_words(c['text'][1:]):
                norm = {x for v in (w, w + 's', w + 'es') for x in T.norm_token(v).split()}
                ids = [i for t, i in self.m['tvocab'].items() if t in norm or (w in t and self._is_compound(t))]
                sets.append(np.unique(np.concatenate([a['tp_flat'][a['tp_off'][i]:a['tp_off'][i + 1]] for i in ids]))
                            if ids else np.zeros(0, dtype=np.int64))
            pids = sets[0] if sets else np.zeros(0, dtype=np.int64)
            for q in sets[1:]:
                pids = np.intersect1d(pids, q, assume_unique=True)
            rx = c['excl_rx']
            pids = [int(p) for p in pids if rx.search(self.phr[int(p)].lower())]
            if pids:
                parts.append(_gather(a['po_off'], a['po_flat'], pids)[0])
        return np.unique(np.concatenate(parts)).astype(np.int32) if parts else np.zeros(0, dtype=np.int32)

    def _is_compound(self, t):
        """Is index token t a merge of plain words by T.tokens ('hair ribbon' -> hairribbon, 'thigh highs' ->
        thighhigh, 'over the knee' -> overknee)? Memo bounded by the token vocabulary."""
        v = self._compound.get(t)
        if v is None:
            v = self._compound[t] = any(T.tokens(x) == [t] for i in range(1, len(t))
                                        for x in (t[:i] + ' ' + t[i:], t[:i] + ' ' + t[i:] + 's',
                                                  t[:i] + ' the ' + t[i:]))
        return v

    def _ban_masks(self, excl):
        """Soft avoidance of the exclusions at the main donor picks: per group (and appearance slot) the rows that
        have a phrase _violations would flag in that group (slot). A row stays usable for the other groups."""
        key = tuple(sorted((c['fid'] if c['fid'] is not None else -1, c['must'] or (), c['excl_ct'])
                                      for c in excl))
        ban = self._ban_cache.pop(key, None)
        if ban is None:
            ban = self._ban_masks_of(excl)
        self._ban_cache[key] = ban          # the 16 latest exclusion sets (~0.6 MB each)
        if len(self._ban_cache) > 16:
            self._ban_cache.pop(next(iter(self._ban_cache)))
        return ban

    def _ban_masks_of(self, excl):
        a = self.a
        occ = np.unique(np.concatenate([self._excl_occ(c) for c in excl]))
        ban = {}
        for g in GROUPS:
            m = np.zeros(self.N, dtype=bool)
            m[a['occ_row'][occ[a['occ_group'][occ] == g]]] = True
            ban[g] = m
        for sl in (HAIR, EYES, BODY):
            m = np.zeros(self.N, dtype=bool)
            m[a['occ_row'][occ[(a['occ_group'][occ] == T.APPEAR) & (self._occ_slot[occ] == sl)]]] = True
            ban['slot', sl] = m
        return ban

    def _unbanned(self, cand, key):
        """cand without the rows banned for key (a group or ('slot', s)); all of cand if every row is (soft)."""
        if self._cur_ban is None or not len(cand):
            return cand
        cand = np.asarray(cand)
        keep = cand[~self._cur_ban[key][cand]]
        return keep if keep.size else cand

    # ------------------------------------------------------------ series / dictionary
    def series_candidates(self, series):
        """Characters (with solo donor rows) belonging to every series in `series`."""
        key = tuple(series)
        v = self._series_cache.get(key)
        if v is None:
            v = set(self.series_chars.get(series[0], ()))
            for sr in series[1:]:
                v &= set(self.series_chars.get(sr, ()))
            v = self._series_cache[key] = sorted(v)
        return v

    def _series_subjects(self, series):
        """Candidates grouped by character identity (costume tags and characters sharing a subject merge):
        identity -> chars, identity -> solo rows per char, identity -> distinct solo rows; identities sorted."""
        key = ('subj',) + series
        v = self._series_cache.get(key)
        if v is None:
            subj = {}
            for ch in self.series_candidates(list(series)):
                subj.setdefault(self.identity[ch], []).append(ch)
            subj = {k: subj[k] for k in sorted(subj)}
            # identity weight from the UNION of its tags' donor rows (base and costume tags share rows)
            v = self._series_cache[key] = (subj, {k: [int(self.char_n[self.char_id[ch]]) for ch in cl]
                                                  for k, cl in subj.items()},
                                           {k: int(np.unique(np.concatenate([self.char_rows(ch) for ch in cl])).size)
                                            for k, cl in subj.items()})
        return v

    def _resolve_series(self, cons, rng, wild, picked):
        """Series constraints, per sample: satisfied by an explicit character of that series, or a character of the
        series is picked (identity weight = its solo rows ** (1 - wild), then a variant tag of it the same way;
        identities not yet picked in this call first) and generation runs as if that character tag had been given. Returns warnings."""
        warn = []
        ser = [c for c in cons if c['kind'] == 'series']
        chars = [c for c in cons if c['kind'] == 'char']
        for c in cons:
            if c['kind'] == 'implicit' and c.get('reason') == 'original' and (ser or chars):
                c['conflict'] = True          # a series / character wins; 'original' is reported
                warn.append(f'original conflicts with {(chars or ser)[0]["text"]}; original not realized')
        if not ser:
            return warn
        if chars:
            ch = chars[0]['char']
            for c in ser:
                if ch in self.series_chars.get(c['text'], ()):
                    c['via'] = ch
                else:
                    warn.append(f'{ch} is not from {c["text"]}; {c["text"]} ignored')
            return warn
        use = [c for c in ser if self.series_chars.get(c['text'])]
        for c in ser:
            if c not in use:
                warn.append(f'{c["text"]}: no solo character of this series in the training data; not realized')
        if not use:
            return warn
        if len(use) > 1 and not self.series_candidates([c['text'] for c in use]):
            warn.append(f'no character is from both {" and ".join(c["text"] for c in use)}; using {use[0]["text"]} '
                        f'only')
            use = use[:1]
        subj, rows, irows = self._series_subjects(tuple(c['text'] for c in use))
        keys = [k for k in subj if k not in picked] or list(subj)
        e = 1.0 - min(1.0, max(0.0, float(wild)))
        w = np.array([float(irows[k]) ** e for k in keys])
        k = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        wc = np.array(rows[k], dtype=float) ** e
        ch = subj[k][int(rng.choice(len(subj[k]), p=wc / wc.sum()))]
        use[0]['char'] = ch
        for c in use[1:]:
            c['via'] = ch
        return warn

    def approx_tokens(self, text, length):
        """Token estimate: the affine words -> tokens fit of that length's real captions (model calib)."""
        c = self.calib[length]
        return int(round(c['affine_a'] * len(text.split()) + c['affine_b']))

    def describe(self, constraints):
        """How each input is understood: [{'text', 'kind', 'tag', 'char', 'reason', 'category'}]. Kinds: implicit,
        count, rejected, subject, char, dropped, series, tag, tag_unsupported, raw; for '-x' (exclusions): exclude
        (tag = the excluded tag or None; reason 'lexical' when only its words are excluded), exclude_unsupported
        (reason: why it is ignored), rejected (reason 'also required': x is also a constraint)."""
        if isinstance(constraints, str):
            constraints = constraints.split(',')
        out = []
        for c in self.parse(constraints):
            k = c['kind']
            neg = c.get('neg')
            tag = c.get('tag') if k in ('tag', 'exclude') or (neg and k == 'rejected') else \
                (c['text'] if k in ('series', 'tag_unsupported') and not neg else None)
            reason = c.get('reason')
            if k == 'rejected' and not neg:
                reason = 'solo only'
            if k == 'exclude' and c['fid'] is None:
                reason = 'lexical'
            name = c.get('char') or tag or (c['text'][1:] if neg else c['text'])
            out.append({'text': c['text'], 'kind': k, 'tag': tag, 'char': c.get('char'), 'reason': reason,
                        'category': self.td.category(name) if self.td is not None else None})
        return out


_LENS = ('long', 'short', 'medium', 'shoulder', 'waist', 'ankle')


FLOOR = {'short': 3, 'medium': 5}
_PLURAL_RX = re.compile(r'^(\d+\+?\s*(girls|boys|others|people)|multiple (girls|boys|others|people)|group|crowd|couple|'
                        r'yuri|yaoi|twins|siblings|sisters|brothers|duo|trio)$|\b(two|three|four|five|several|many|'
                        r'multiple|group of|pair of|couple of|crowd of)\s+([\w-]+\s+){0,2}(girls|boys|women|men|people|'
                        r'characters|friends|sisters|brothers|catgirls|maids|students)\b')


def _is_plural_request(s):
    return bool(T.PLURAL_COUNT_TAGS_RX.match(s) or _PLURAL_RX.search(s))


_PERSON = (r'(girl|boy|woman|man|lady|gentleman|maid|sister|brother|friend|mother|father|mom|dad|girlfriend|'
           r'boyfriend|child|kid|daughter|son|wife|husband|catgirl|foxgirl|person|partner|companion|twin)')
_COORD_RX = re.compile(r'\b' + _PERSON + r's?\b.*\b(and|with|&|plus|alongside|beside|next to)\s+(a|an|the|his|her|their|'
                       r'another|other|two|some|her own)?\s*([\w-]+\s+){0,2}' + _PERSON + r's?\b')


def _is_coordinated(s):
    """'a girl and a boy', 'a boy and his sister', 'girl with a boy': more than one person in a subject."""
    return bool(_COORD_RX.search(s))


def _base_name(c):
    return _QUAL_RX.sub('', c).strip()


def char_series_from_metadata(path, chars):
    """(character -> {series}, all series names). Evidence comes only from unambiguous rows: one series, or one
    base character (costume tags of one character: 'firefly', 'firefly (spring missive)'). A crossover row with
    several characters and several series (8 characters x 5 series) says nothing about who belongs where. A series
    belongs to a character if >= SERIES_MIN_SHARE of its unambiguous rows carry it. Both columns hold ', '-separated
    names (no commas inside names)."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=['character_name', 'series_name'])
    pairs = {}
    for c, sr in zip(t.column('character_name').to_pylist(), t.column('series_name').to_pylist()):
        if c:
            pairs[c, sr] = pairs.get((c, sr), 0) + 1
    nc, ncs, known = {}, {}, set()
    for (c, sr), n in pairs.items():
        ss = [x for x in (sr or '').split(', ') if x]
        known.update(ss)
        ch = c.split(', ')
        if len(ss) > 1 and len({_base_name(x) for x in ch}) > 1:
            continue
        for x in ch:
            if x not in chars:
                continue
            nc[x] = nc.get(x, 0) + n
            for y in ss:
                ncs[x, y] = ncs.get((x, y), 0) + n
    out = {}
    for (x, y), n in ncs.items():
        if n >= SERIES_MIN_SHARE * nc[x]:
            out.setdefault(x, set()).add(y)
    return out, known


def _meta_enc(v):
    """meta -> its JSON form; only str / int / float / bool / None / list / tuple / dict (str or int keys) have one."""
    t = type(v)
    if t is dict:
        if all(type(k) is str for k in v):
            if len(v) == 1 and next(iter(v)) in _META_TAGS:
                raise ValueError(f'meta: a dict with the one key {next(iter(v))!r}')
            return {k: _meta_enc(x) for k, x in v.items()}
        if all(type(k) is int for k in v):
            return {'__int_keys__': [[k, _meta_enc(x)] for k, x in v.items()]}
        raise TypeError('meta: a dict with keys other than all str or all int')
    if t is list:
        return [_meta_enc(x) for x in v]
    if t is tuple:
        return {'__tuple__': [_meta_enc(x) for x in v]}
    if v is None or t in (str, int, float, bool):
        return v
    raise TypeError(f'meta: {t.__name__} has no JSON form')


def _meta_hook(d):
    if len(d) == 1:
        k = next(iter(d))
        if k in _META_TAGS:
            return _META_TAGS[k](d[k])
    return d


def meta_to_json(meta):
    """The meta.json text (deterministic: the meta's own key order, ASCII escapes off, no NaN / inf)."""
    return json.dumps({'format': META_FORMAT, 'meta': _meta_enc(meta)}, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')) + '\n'


def load_meta(model_dir):
    """The model's metadata dict: model_dir/meta.json if it exists (stdlib json), else meta.pkl (build_model.py)."""
    p = os.path.join(model_dir, 'meta.json')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            d = json.loads(f.read(), object_hook=_meta_hook)
        if d.get('format') != META_FORMAT:
            raise ValueError(f'{p}: meta format {d.get("format")!r}, expected {META_FORMAT}')
        return d['meta']
    import pickle
    with open(os.path.join(model_dir, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)


def char_series_to_json(char_series, known):
    """The char_series.json document (deterministic: sorted keys and lists)."""
    return {'format': CHAR_SERIES_FORMAT, 'char_series': {c: sorted(v) for c, v in sorted(char_series.items())},
            'known_series': sorted(known)}


def char_series_from_json(path):
    """(character -> {series}, all series names) as char_series_from_metadata returned them at export time."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if d.get('format') != CHAR_SERIES_FORMAT:
        raise ValueError(f'{path}: char_series format {d.get("format")!r}, expected {CHAR_SERIES_FORMAT}')
    return {c: set(v) for c, v in d['char_series'].items()}, set(d['known_series'])


_WORDS_RX = re.compile(r'[a-z0-9]+')
_POSS_RX = re.compile(r"'s\b")
_QUAL_RX = re.compile(r'\s*\([^)]*\)')
_SUBJ_OF_RX = re.compile(r'^.*?illustration of (.+)$')


def _norm_words(s):
    return ' '.join(_WORDS_RX.findall(_POSS_RX.sub('', s.lower())))


_NUM = r'(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|multiple|pair of|couple of|' \
       r'group of|trio|duo|twins)'
_BEING = r'(girl|boy|woman|man|lady|child|kid|friend|sister|brother|companion|partner|cat|dog|kitten|puppy|bird|' \
         r'fox|wolf|rabbit|bunny|creature|monster|animal|pet|spirit|fairy|dragon|mascot|familiar|slime|clone)'
_PLURAL_BEING = r'(girls|boys|women|men|people|persons|friends|children|kids|sisters|brothers|siblings|companions|' \
                r'characters|cats|dogs|kittens|puppies|birds|foxes|wolves|rabbits|bunnies|creatures|monsters|' \
                r'animals|pets|spirits|fairies|dragons|slimes|clones|minions|mascots|familiars)'
_MULTI_RX = re.compile(r'(\band\b|&|\btogether with\b|\balongside\b|\baccompanied by\b|\b' + _NUM + r'\b(?!-)|\b' +
                       _PLURAL_BEING + r'\b)')
_WITH_BEING_RX = re.compile(r'\bwith (a |an |the |her |his |their |its )?([\w-]+ ){0,2}' + _BEING +
                            r'\b(?!-| ?(ears?|tails?|paws?|horns?|wings?|hood|hat|print|motif|mask|costume|outfit|'
                            r'shaped|plush|doll|hair|eyes|girl|boy))')


_HEAD_PERSON_RX = re.compile(r'\b(girls?|boys?|wom[ae]n|m[ae]n|lady|ladies|child|children|kids?|someone|person|'
                             r'people|cosplayer|human|trainer|figure|student|schoolgirl|catgirl|foxgirl|maid)\b')
_PERSONIFY_RX = re.compile(r'\b(inspired by|personification of|personified|version of|gijinka|humanized|'
                           r'representing|concept for|based on)\b')     # one being: a humanized character
_COSTUME_RX = re.compile(r'\b(?:cosplaying|cosplayed|dressed (?:up )?as|disguised as|as|crossed with)\b')
_OUTFIT_RX = re.compile(r"\bin (?:her |his |a |an |the )?[^,]*?\b(?:outfit|armor|armour|costume|attire|cosplay|dress|"
                        r"uniform|clothes|clothing|skin|look|style|form)\b")


def char_identities(chars, char_series, subject_of):
    """Character identity: costume tags of one character ('serika (swimsuit) (blue archive)', 'serika (idol) (blue
    archive)') are one identity: the same base name with a shared series (or both without one), or the same subject
    phrase. Returns char -> identity key (smallest member tag)."""
    parent = {c: c for c in chars}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    by_base, by_subj = {}, {}
    for c in chars:
        by_base.setdefault(_norm_words(_base_name(c)), []).append(c)
        by_subj.setdefault(subject_of[c], []).append(c)
    for cl in by_base.values():
        for i, a in enumerate(cl):
            for b in cl[i + 1:]:
                sa, sb = char_series.get(a, set()), char_series.get(b, set())
                if (sa & sb) or not (sa or sb):
                    union(a, b)
    for cl in by_subj.values():
        for b in cl[1:]:
            union(cl[0], b)
    return {c: find(c) for c in chars}


def _plain_words(text):
    """Lower-case words (any script) of a text, without the articles / pronouns T.TAG_DROP drops."""
    return [w for w in re.findall(r'[^\W_]+', text.lower()) if w not in T.TAG_DROP]


def _plain_rx(texts):
    """Regex: the plain words of any of texts, consecutive (articles / pronouns may stand between), each also as
    singular / plural (s, es); None if no text has words."""
    alts = []
    for t in texts:
        ws = _plain_words(t)
        if not ws:
            continue
        forms = []
        for w in ws:
            base = {w} | ({w[:-1]} if w.endswith('s') and len(w) > 2 else set()) | \
                ({w[:-2]} if w.endswith('es') and len(w) > 3 else set())
            vs = sorted({b + x for b in base for x in ('', 's', 'es')}, key=lambda v: (-len(v), v))
            forms.append('(?:' + '|'.join(map(re.escape, vs)) + ')')
        sep = r'(?:[\W_]+(?:' + '|'.join(sorted(T.TAG_DROP)) + r'))*[\W_]+'
        alts.append(sep.join(forms))
    return re.compile(r'(?<![^\W_])(?:' + '|'.join(alts) + r')(?![^\W_])') if alts else None


def _is_subject_np(s):
    return T.is_subject_like(s) and T.strip_subject_np(s) == ''


def _is_location(c):
    if c is None:
        return False
    txt = c.get('tag') or c['text']
    return any(w in T.LOCATION_WORDS for w in T.tokens(txt))


def _head_colours(text, head):
    """Colour words right before `head` ('hair' / 'eye') in a constraint text."""
    toks = T.tokens(text)
    out = set()
    for i, w in enumerate(toks):
        if w == head:
            for x in reversed(toks[max(0, i - 4):i]):
                if x in T.BLOCKERS or x in T.STOP:
                    break
                if x in T._PURE_COLOR_CANON:
                    out.add(x)
    return out


def _user_conflicts(cons):
    """Warnings for user constraints that ask for two different hair / eye colours ('blue eyes' +
    'blue hair and red eyes'). Heterochromia requests ('one yellow and one blue eye') are not conflicts."""
    msgs = []
    for head, name in (('eye', 'eye colour'), ('hair', 'hair colour')):
        per = []
        for c in cons:
            if c['kind'] not in ('tag', 'raw', 'tag_unsupported'):
                continue
            txt = c.get('tag') or c['text']
            if re.search(r'heterochromia|\bone\b.*\b(one|other)\b|multicolou?red|two-tone|streak|inner|gradient', txt):
                continue
            cols = _head_colours(txt, head)
            if cols:
                per.append(cols)
        for i in range(len(per)):
            for j in range(i + 1, len(per)):
                if not (per[i] & per[j]):
                    msgs.append(f'constraints conflict: {name} {"/".join(sorted(per[i]))} vs {"/".join(sorted(per[j]))}')
    return sorted(set(msgs))


def _attrs(text):
    """(colour words, length words) describing the hair / eyes in a phrase: the words right before 'hair' / 'eye'
    ('dark blue-grey long hair'), not every colour of the phrase ('... with a white accent braid')."""
    toks = T.tokens(text)
    heads = [i for i, w in enumerate(toks) if w in ('hair', 'eye')]
    if not heads:
        return ({w for w in toks if w in T._PURE_COLOR_CANON}, {w for w in toks if w in _LENS})
    cols, lens = set(), set()
    for h in heads:
        for w in reversed(toks[max(0, h - 5):h]):
            if w in T.BLOCKERS or w in T.STOP:
                break
            if w in T._PURE_COLOR_CANON:
                cols.add(w)
            elif w in _LENS:
                lens.add(w)
    return cols, lens


def _compatible(attr, ref):
    cols, lens = attr
    rc, rl = ref
    return (not cols or not rc or cols <= rc) and (not lens or not rl or lens <= rl)


def _cli_option(ap, x):
    """Would argparse read x as a defined option (or a negative number / a plain value)? '--...' (any long option,
    abbreviations included), '-h', '-n=1', '-n1' (a value of -n's type); not '-smile', '-nurse', '-hat'."""
    if not x.startswith('-') or x.startswith('--') or x == '-' or re.fullmatch(r'-\d+(\.\d*)?', x):
        return True
    short = {o: act for act in ap._actions for o in act.option_strings if not o.startswith('--')}
    if x.split('=', 1)[0] in short:
        return True
    act = short.get(x[:2])
    if act is None or act.nargs == 0:
        return False
    try:
        (act.type or str)(x[2:])
        return True
    except (TypeError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description='tag-constrained caption generator')
    ap.add_argument('constraints', help='comma-separated Danbooru tags / character tags / free phrases')
    ap.add_argument('-n', type=int, default=5)
    ap.add_argument('--length', default='long', choices=['long', 'medium', 'short'])
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--beta', type=float, default=1.0, help='max compatibility sharpness (used as (1-wild)*beta)')
    ap.add_argument('--wild', type=float, default=1.0,
                    help='1 = surprising combinations (uniform donors, no conflict rejection, independent action and '
                         'place); 0 = plausible (PMI-weighted donors, conflict rejection, coupled action/place)')
    ap.add_argument('--order', default='template', choices=['template', 'facet'],
                    help='phrase order: template = the order of a real donor caption (default), facet = learned '
                         'facet positions (one fixed order)')
    ap.add_argument('--overlay-text', action='store_true',
                    help='keep marks on the picture (watermark, artist signature, corner text, logo); default: dropped')
    ap.add_argument('--no-scene-text', action='store_true',
                    help='also drop text on scene objects (shirt prints, signs, book covers, speech bubbles)')
    ap.add_argument('--no-mix-look', action='store_true',
                    help='one appearance donor (hair + eyes + body traits of one real caption, often a named '
                         'character); default: hair, eyes and body from three different rows')
    ap.add_argument('--explain', action='store_true')
    ap.add_argument('--model', default=os.path.join(HERE, 'model'))
    ap.add_argument('--metadata', default=None, help='training metadata for the character -> series map (default '
                    'METADATA_DEFAULT)')
    ap.add_argument('--suggest', action='store_true', help='treat the argument as a prefix and list tags')
    ap.add_argument('--tags-dict', default=TAGS_DEFAULT, help='Danbooru tag dictionary (.json or .parquet)')
    ap.add_argument('--no-tags-dict', action='store_true', help='no tag dictionary (v1 parsing)')
    # an exclusion as the first constraint ('-smile, maid'; '-sm' with --suggest) is no option: a leading space
    # keeps argparse from reading it as one (parse() and suggest_info() strip it)
    a = ap.parse_args([x if _cli_option(ap, x) else ' ' + x for x in sys.argv[1:]])
    t0 = time.time()
    g = Generator.load(a.model, tags_path=None if a.no_tags_dict else a.tags_dict,
                       metadata_path=a.metadata if a.metadata else _DEFAULT)
    tl = time.time() - t0
    if a.suggest:
        for d in g.suggest_info(a.constraints):
            print(f'{d["tag"]}  [{d["category"] or "-"}]  train {d["training_count"]}  posts {d["post_count"]}')
        return
    t1 = time.time()
    res = g.generate(a.constraints.split(','), n=a.n, length=a.length, seed=a.seed, beta=a.beta, explain=a.explain,
                     wild=a.wild, order=a.order, overlay_text=a.overlay_text, scene_text=not a.no_scene_text,
                     mix_look=not a.no_mix_look)
    tg = time.time() - t1
    for i, s in enumerate(res):
        print(f'[{i + 1}] {s.text}')
        if a.explain:
            print(s.explain_text())
        else:
            for d in s.dropped:
                print(f'# [{i + 1}] dropped: {d["text"]} ({d["reason"]})', file=sys.stderr)
            if s.picked_character:
                print(f'# [{i + 1}] picked character: {s.picked_character}', file=sys.stderr)
            for w in s.warnings:
                print(f'# [{i + 1}] warning: {w}', file=sys.stderr)
            if s.unrealized:
                print(f'# [{i + 1}] unrealized: {", ".join(s.unrealized)}', file=sys.stderr)
    print(f'# load {tl:.2f}s, generate {1000 * tg / max(1, len(res)):.1f} ms/sample', file=sys.stderr)


if __name__ == '__main__':
    main()
