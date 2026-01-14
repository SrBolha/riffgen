# riffgen.py
"""
Riff generator for D-standard guitar tab + MIDI.

What it does
- Generates a metal-ish song in 16-bar segments (8-bar phrase + verbatim repeat).
- Every 8-bar phrase follows strict motif logic:
  bar1: motif A
  bar2: motif A' (modified)
  bar3: repeat A
  bar4: contrast B
  bar5: repeat A
  bar6: repeat A'
  bar7: repeat A
  bar8: contrast B2 (ending)

Song form (fixed; no random placement)
- intro (varied: gallops/bursts/offbeatgallops/downpicking)
- verse (downpicking)
- bridge (pedalpoint or chords) -> prepares chorus
- chorus (melodies or chords)
- intro (repeat)
- verse (repeat)
- bridge (repeat)
- chorus (repeat)
- instrumental (varied; may include classical)
- bridge (repeat)
- chorus (repeat)
- outro (varied: gallops/bursts)

Key/mode control
- --tone <NOTE> (e.g. D, Eb, F#). If omitted, random.
- --melody_mode <MODE>. If omitted, random dark mode (never Ionian/Mixolydian/Lydian).
- Mode identity is emphasized via characteristic tones (e.g. b2 in Phrygian, b5 in Locrian).

Tone changes
- --tonechange N (default 0): regenerate N structure labels (verse/chorus/bridge/etc) in a different key+mode.
  Repeats remain verbatim because phrases are cached per label.

Register
- All non-classical sections prefer D1..G2 (open low D up to +17 semitones).
- Classical section uses low D + 3 octaves, but avoids leaps > 1 octave and avoids 16ths.

Outputs
- riff.mid (unless --no_midi)
- riff.txt (ASCII tab + structure markers + chord-per-beat)

Notes
- Durations are quantized to 0.5 beats (no 16ths/32nds).
- Triplets (when allowed) always occur as 3 consecutive notes within 1 beat.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import secrets
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None


# ----------------------------
# Guitar setup (D standard)
# ----------------------------

# MIDI pitch of open strings (1 = high string, 6 = low string)
# D standard: D2 G2 C3 F3 A3 D4 (low->high)
OPEN: Dict[int, int] = {6: 38, 5: 43, 4: 48, 3: 53, 2: 57, 1: 62}
STRING_NAMES: Dict[int, str] = {6: "D", 5: "G", 4: "C", 3: "F", 2: "A", 1: "D"}

PITCH_LO_MIDI = OPEN[6]                # D2
PITCH_HI_MIDI = OPEN[6] + 17           # ~G3? (low register boundary)
CLASSICAL_HI_MIDI = OPEN[6] + 36       # + 3 octaves

BEATS_PER_BAR = 4
PHRASE_BARS = 8
SEGMENT_BARS = 16  # phrase + repeat

SONG_MIN_TOTAL_BARS = 192
SONG_DEFAULT_TOTAL_BARS = 192  # >=192 and divisible by 16

# Velocity policy overrides (user-specified)
PALM_MUTE_VELOCITY = 45
# Backward-compat alias; prefer PALM_MUTE_VELOCITY in new code.
FIXED_VELOCITY_45 = PALM_MUTE_VELOCITY
VOCAL_DEFAULT_VELOCITY = 80

# ----------------------------
# User-tuned generation policy
# ----------------------------

# Pedalpoint: reduce tonic/root overuse (was ~0.78).
PEDALPOINT_TONIC_WEIGHT = 0.45

# Section selection: all sections equal chance, except these are rarer.
SECTION_DOWNPICK_WEIGHT = 0.20
SECTION_MELODIES_WEIGHT = 0.20

# The complete "kind" pool.
SECTION_KINDS: List[str] = [
    "gallops",
    "gallopsopen",
    "bursts",
    "justbursts",
    "burstsopen",
    "offbeatgallops",
    "chords",
    "chordsprog",
    "chordprogression",
    "pedalpoint",
    "pedalpointoctave",
    "classical",
    "downpicking",
    "melodies",
]

# Random selection pool: chordprogression is superseded by chordsprog.
SECTION_KINDS_RANDOM: Tuple[str, ...] = tuple(k for k in SECTION_KINDS if k != "chordprogression")

# Chord-progression templates used by the "chordprogression" section.
#
# IMPORTANT: this section is *tonal* (relative to tonic) and does NOT have to follow --melody_mode.
# Templates use lightweight roman tokens with optional:
#   accidentals: b/# (e.g. bVI, #IV)
#   sevenths: 7 / maj7 (e.g. V7, Imaj7)
#   diminished: o / °, half-diminished: ø (e.g. vii°7, iiø7)
#   secondary dominants: V/V, V7/ii, vii°/V
#   tritone substitute dominant: subV7 or subV7/<degree> (e.g. subV7, subV7/ii)
#
# The generator guarantees triads/tetrads only (3–4 notes per chord voicing).
@dataclasses.dataclass(frozen=True)
class ChordProgressionTemplate:
    tokens: Tuple[str, ...]
    tags: Tuple[str, ...] = ("classical",)
    weight: float = 1.0


CHORDPROGRESSION_POOL: Dict[str, ChordProgressionTemplate] = {
    # --- Classical major / functional harmony
    "I_IV_V_I": ChordProgressionTemplate(("I", "IV", "V", "I"), ("classical", "major"), 1.2),
    "I_ii_V_I": ChordProgressionTemplate(("I", "ii", "V", "I"), ("classical", "major"), 1.3),
    "I_vi_ii_V_I": ChordProgressionTemplate(("I", "vi", "ii", "V", "I"), ("classical", "major"), 1.25),
    "I_iii_vi_ii_V_I": ChordProgressionTemplate(("I", "iii", "vi", "ii", "V", "I"), ("classical", "major"), 1.15),
    "circle_5ths_major": ChordProgressionTemplate(("I", "IV", "vii°", "iii", "vi", "ii", "V", "I"), ("classical", "major"), 1.2),
    "pachelbel": ChordProgressionTemplate(("I", "V", "vi", "iii", "IV", "I", "IV", "V"), ("classical", "major"), 1.25),
    "ii_V_I": ChordProgressionTemplate(("ii", "V7", "I"), ("classical", "major"), 1.25),
    "ii7_V7_Imaj7": ChordProgressionTemplate(("ii7", "V7", "Imaj7"), ("classical", "major", "tetrads"), 1.1),
    "I_V_vi_IV": ChordProgressionTemplate(("I", "V", "vi", "IV"), ("major",), 1.0),
    "I_V_ii_IV": ChordProgressionTemplate(("I", "V", "ii", "IV"), ("major",), 0.9),
    "I_IV_I_V": ChordProgressionTemplate(("I", "IV", "I", "V"), ("major",), 0.9),
    "I_V_I": ChordProgressionTemplate(("I", "V", "I"), ("classical", "major"), 1.0),
    "I_V_I_64": ChordProgressionTemplate(("I", "V", "I", "V"), ("classical", "major"), 0.8),

    # --- Classical minor / sad functional harmony
    "i_iv_V_i": ChordProgressionTemplate(("i", "iv", "V7", "i"), ("classical", "minor", "sad"), 1.35),
    "i_iiø7_V7_i": ChordProgressionTemplate(("i", "iiø7", "V7", "i"), ("classical", "minor", "sad", "tetrads"), 1.25),
    "iiø7_V7_i": ChordProgressionTemplate(("iiø7", "V7", "i"), ("classical", "minor", "sad", "tetrads"), 1.15),
    "lament_minor": ChordProgressionTemplate(("i", "bVII", "bVI", "V"), ("classical", "minor", "sad"), 1.35),
    "andalusian": ChordProgressionTemplate(("i", "bVII", "bVI", "V"), ("classical", "minor", "sad"), 1.2),
    "i_bVI_bIII_bVII": ChordProgressionTemplate(("i", "bVI", "bIII", "bVII"), ("minor", "sad"), 1.25),
    "i_bVII_bVI_V": ChordProgressionTemplate(("i", "bVII", "bVI", "V"), ("minor", "sad"), 1.2),
    "i_VI_III_VII": ChordProgressionTemplate(("i", "bVI", "bIII", "bVII"), ("minor", "sad"), 1.15),
    "i_iv_bVII_bIII": ChordProgressionTemplate(("i", "iv", "bVII", "bIII"), ("minor", "sad"), 1.1),
    "i_bII_V_i": ChordProgressionTemplate(("i", "bII", "V7", "i"), ("classical", "minor", "sad", "chromatic"), 1.15),  # Neapolitan flavor
    "i_bVI_iv_V": ChordProgressionTemplate(("i", "bVI", "iv", "V7"), ("minor", "sad"), 1.05),
    "i_iv_iio_V_i": ChordProgressionTemplate(("i", "iv", "ii°", "V7", "i"), ("classical", "minor", "sad"), 1.15),

    # --- Longer classical sequences (sad / dramatic)
    "minor_desc_5ths": ChordProgressionTemplate(("i", "iv", "bVII", "bIII", "bVI", "ii°", "V7", "i"), ("classical", "minor", "sad"), 1.25),
    "minor_circle": ChordProgressionTemplate(("i", "iv", "VII", "III", "VI", "ii°", "V7", "i"), ("classical", "minor", "sad"), 1.1),
    "passacaglia_minor": ChordProgressionTemplate(("i", "V", "bVI", "III", "iv", "i", "ii°", "V7"), ("classical", "minor", "sad"), 1.05),

    # --- Secondary dominants (functional, tonal)
    "I_V_V": ChordProgressionTemplate(("I", "V/V", "V", "I"), ("classical", "major", "chromatic"), 1.05),
    "I_Vii_V": ChordProgressionTemplate(("I", "vii°/V", "V", "I"), ("classical", "major", "chromatic"), 0.95),
    "I_V_ii": ChordProgressionTemplate(("I", "V/ii", "ii", "V7", "I"), ("classical", "major", "chromatic"), 1.0),
    "I_V_vi": ChordProgressionTemplate(("I", "V/vi", "vi", "ii", "V7", "I"), ("classical", "major", "chromatic"), 0.95),
    "chain_sec_dom": ChordProgressionTemplate(("I", "V/vi", "vi", "V/ii", "ii", "V/V", "V", "I"), ("classical", "major", "chromatic"), 0.95),
    "minor_sec_dom": ChordProgressionTemplate(("i", "V/bVI", "bVI", "V7", "i"), ("minor", "sad", "chromatic"), 0.9),

    # --- Substitute dominants / backdoor dominants
    "ii_subV7_I": ChordProgressionTemplate(("ii", "subV7", "I"), ("classical", "major", "chromatic"), 1.0),
    "iv_bVII7_I": ChordProgressionTemplate(("iv", "bVII7", "I"), ("major", "sad", "chromatic"), 0.95),
    "I_subV7_I": ChordProgressionTemplate(("I", "subV7", "I"), ("chromatic",), 0.9),
    "i_subV7_i": ChordProgressionTemplate(("i", "subV7", "i"), ("minor", "sad", "chromatic"), 0.95),
    "ii_subV7_i": ChordProgressionTemplate(("ii°", "subV7", "i"), ("minor", "sad", "chromatic"), 0.9),

    # --- Modal interchange / borrowed (sad color even in major)
    "I_iv_V_I": ChordProgressionTemplate(("I", "iv", "V7", "I"), ("classical", "major", "sad"), 1.05),
    "I_bVII_IV_I": ChordProgressionTemplate(("I", "bVII", "IV", "I"), ("major", "sad"), 0.95),
    "I_bVI_IV_I": ChordProgressionTemplate(("I", "bVI", "IV", "I"), ("major", "sad"), 0.9),
    "I_bIII_IV_I": ChordProgressionTemplate(("I", "bIII", "IV", "I"), ("major", "sad"), 0.85),
    "I_bVII7_I": ChordProgressionTemplate(("I", "bVII7", "I"), ("major", "sad", "chromatic"), 0.9),

    # --- Cadential variants / bittersweet endings
    "deceptive_V_vi": ChordProgressionTemplate(("I", "IV", "V", "vi"), ("major", "sad"), 0.9),
    "picardy": ChordProgressionTemplate(("i", "iv", "V7", "I"), ("classical", "minor", "sad"), 0.95),
    "minor_deceptive": ChordProgressionTemplate(("i", "V7", "bVI", "V7", "i"), ("minor", "sad"), 0.95),
    # --- More major classical / pop-classical hybrids
    "I_vi_IV_V": ChordProgressionTemplate(("I", "vi", "IV", "V"), ("major", "classical"), 1.0),
    "I_vi_ii_V": ChordProgressionTemplate(("I", "vi", "ii", "V"), ("major", "classical"), 0.95),
    "I_IV_ii_V": ChordProgressionTemplate(("I", "IV", "ii", "V"), ("major", "classical"), 1.0),
    "I_ii_IV_V": ChordProgressionTemplate(("I", "ii", "IV", "V"), ("major",), 0.9),
    "vi_IV_I_V": ChordProgressionTemplate(("vi", "IV", "I", "V"), ("major",), 0.9),
    "I_iii_IV_V": ChordProgressionTemplate(("I", "iii", "IV", "V"), ("major",), 0.9),
    "I_iii_vi_IV": ChordProgressionTemplate(("I", "iii", "vi", "IV"), ("major",), 0.9),
    "I_IV_V_vi": ChordProgressionTemplate(("I", "IV", "V", "vi"), ("major", "sad"), 0.9),
    "I_V_IV_I": ChordProgressionTemplate(("I", "V", "IV", "I"), ("major",), 0.9),
    "I_IV_V_IV": ChordProgressionTemplate(("I", "IV", "V", "IV"), ("major",), 0.85),
    "I_bVII_IV_V": ChordProgressionTemplate(("I", "bVII", "IV", "V"), ("major", "sad"), 0.85),
    "I_bVI_bVII_I": ChordProgressionTemplate(("I", "bVI", "bVII", "I"), ("major", "sad"), 0.8),
    "Imaj7_vi7_ii7_V7": ChordProgressionTemplate(("Imaj7", "vi7", "ii7", "V7"), ("major", "classical", "tetrads"), 0.85),
    "I_ii7_V7_Imaj7": ChordProgressionTemplate(("I", "ii7", "V7", "Imaj7"), ("major", "classical", "tetrads"), 0.85),
    "I_V7_vi7_IV": ChordProgressionTemplate(("I", "V7", "vi7", "IV"), ("major", "sad", "tetrads"), 0.8),

    # --- More minor / sad palettes
    "i_bVI_bVII_i": ChordProgressionTemplate(("i", "bVI", "bVII", "i"), ("minor", "sad"), 1.05),
    "i_bIII_bVII_bVI": ChordProgressionTemplate(("i", "bIII", "bVII", "bVI"), ("minor", "sad"), 1.0),
    "i_bVII_iv_V": ChordProgressionTemplate(("i", "bVII", "iv", "V7"), ("minor", "sad"), 1.1),
    "i_iv_bVI_V": ChordProgressionTemplate(("i", "iv", "bVI", "V7"), ("minor", "sad"), 1.0),
    "i_V_bVI_bVII": ChordProgressionTemplate(("i", "V7", "bVI", "bVII"), ("minor", "sad"), 0.95),
    "i_v_bVI_bVII": ChordProgressionTemplate(("i", "v", "bVI", "bVII"), ("minor", "sad"), 0.9),
    "i_v_iv_V": ChordProgressionTemplate(("i", "v", "iv", "V7"), ("minor", "sad"), 0.9),
    "i_iv_VII_III": ChordProgressionTemplate(("i", "iv", "VII", "III"), ("minor", "sad"), 0.9),
    "i_bVII_bVI_bVII": ChordProgressionTemplate(("i", "bVII", "bVI", "bVII"), ("minor", "sad"), 0.85),
    "i_III_VII_iv": ChordProgressionTemplate(("i", "bIII", "bVII", "iv"), ("minor", "sad"), 0.9),
    "i_iiø7_V7_bVI": ChordProgressionTemplate(("i", "iiø7", "V7", "bVI"), ("minor", "sad", "tetrads"), 0.85),
    "i_bII7_V7_i": ChordProgressionTemplate(("i", "bII7", "V7", "i"), ("minor", "sad", "chromatic"), 0.9),
    "i_vii°7_i": ChordProgressionTemplate(("i", "vii°7", "i"), ("minor", "sad", "tetrads"), 0.8),

    # --- More secondary-dominant / chromatic classical devices
    "I_V/IV_IV_I": ChordProgressionTemplate(("I", "V/IV", "IV", "I"), ("classical", "major", "chromatic"), 0.95),
    "I_V/iii_iii_vi_ii_V_I": ChordProgressionTemplate(("I", "V/iii", "iii", "vi", "ii", "V7", "I"), ("classical", "major", "chromatic"), 0.9),
    "I_V/vi_vi_V/V_V_I": ChordProgressionTemplate(("I", "V/vi", "vi", "V/V", "V", "I"), ("classical", "major", "chromatic"), 0.9),
    "ii_V/V_V_I": ChordProgressionTemplate(("ii", "V/V", "V7", "I"), ("classical", "major", "chromatic"), 0.9),
    "i_V/V_V_i": ChordProgressionTemplate(("i", "V/V", "V7", "i"), ("classical", "minor", "chromatic", "sad"), 0.9),
    "vii°7_V7_I": ChordProgressionTemplate(("vii°7", "V7", "I"), ("classical", "major", "chromatic", "tetrads"), 0.85),
    "vii°7_V7_i": ChordProgressionTemplate(("vii°7", "V7", "i"), ("classical", "minor", "chromatic", "sad", "tetrads"), 0.85),
    "bII7_V7_I": ChordProgressionTemplate(("bII7", "V7", "I"), ("classical", "major", "chromatic"), 0.85),
    "bII7_subV7_I": ChordProgressionTemplate(("bII7", "subV7", "I"), ("chromatic",), 0.75),
    "ii_subV7_V7_I": ChordProgressionTemplate(("ii", "subV7", "V7", "I"), ("classical", "major", "chromatic"), 0.85),
    "i_subV7_V7_i": ChordProgressionTemplate(("i", "subV7", "V7", "i"), ("minor", "sad", "chromatic"), 0.85),

    # --- Backdoor / plagal-ish sadness
    "IV_iv_I": ChordProgressionTemplate(("IV", "iv", "I"), ("major", "sad"), 0.85),
    "I_iv_bVII_I": ChordProgressionTemplate(("I", "iv", "bVII", "I"), ("major", "sad"), 0.8),
    "i_iv_bVII_i": ChordProgressionTemplate(("i", "iv", "bVII", "i"), ("minor", "sad"), 0.9),
    "i_bVII_IV_i": ChordProgressionTemplate(("i", "bVII", "IV", "i"), ("minor", "sad"), 0.85),

}

def _expand_chordprogression_pool(
    pool: Dict[str, ChordProgressionTemplate],
) -> Dict[str, ChordProgressionTemplate]:
    """Return a much larger progression pool with classical/sad variations."""
    out: Dict[str, ChordProgressionTemplate] = dict(pool)

    extras: Dict[str, ChordProgressionTemplate] = {
        "lament_i_bVII_bVI_V": ChordProgressionTemplate(("i", "bVII", "bVI", "V7"), ("classical", "minor", "sad"), 0.95),
        "lament_i_iv_bVII_bVI": ChordProgressionTemplate(("i", "iv", "bVII", "bVI"), ("classical", "minor", "sad"), 0.9),
        "lament_i_bVI_iv_V": ChordProgressionTemplate(("i", "bVI", "iv", "V7"), ("classical", "minor", "sad"), 0.9),
        "neapolitan_bII6_V_i": ChordProgressionTemplate(("i", "bII6", "V7", "i"), ("classical", "minor", "sad", "chromatic"), 1.0),
        "minor_circle_long": ChordProgressionTemplate(("i", "iv", "VII", "III", "VI", "iiø7", "V7", "i"), ("classical", "minor", "sad"), 0.85),
        "I_V7_ii_V7_I": ChordProgressionTemplate(("I", "V7/ii", "ii", "V7", "I"), ("classical", "major", "tension"), 0.95),
        "I_V7_vi_ii_V_I": ChordProgressionTemplate(("I", "V7/vi", "vi", "ii", "V7", "I"), ("classical", "major", "tension"), 0.9),
        "i_V7_iv_V7_i": ChordProgressionTemplate(("i", "V7/iv", "iv", "V7", "i"), ("classical", "minor", "sad", "tension"), 0.95),
        "ii_subV7_I": ChordProgressionTemplate(("ii", "subV7", "I"), ("classical", "major", "chromatic", "tension"), 0.85),
        "i_subV7_i": ChordProgressionTemplate(("i", "subV7", "i"), ("classical", "minor", "chromatic", "sad"), 0.8),
        "I_iv6_I": ChordProgressionTemplate(("I", "iv6", "I"), ("classical", "major", "sad"), 0.9),
        "I_bVI_V_I": ChordProgressionTemplate(("I", "bVI", "V7", "I"), ("classical", "major", "sad"), 0.85),
        "I_bIII_bVII_IV_I": ChordProgressionTemplate(("I", "bIII", "bVII", "IV", "I"), ("major", "sad"), 0.8),
    }
    for k, v in extras.items():
        out.setdefault(k, v)

    def strip_target(tok: str) -> str:
        t = str(tok).strip()
        if "/" in t:
            return t.split("/", 1)[1].strip()
        m = re.match(r"^([b#]*)(N|[ivIV]+)", t)
        if m:
            return f"{m.group(1) or ''}{m.group(2)}"
        return t

    def add_variant(base_name: str, toks: List[str], tags_extra: Tuple[str, ...], w_mul: float) -> None:
        tpl = tuple(str(x) for x in toks)
        key = f"{base_name}__{abs(hash(tpl)) % 10_000_000}"
        if key in out:
            return
        base_tpl = pool[base_name]
        tags = tuple(dict.fromkeys(tuple(base_tpl.tags) + ("derived",) + tuple(tags_extra)))
        out[key] = ChordProgressionTemplate(tpl, tags, float(base_tpl.weight) * float(w_mul))

    for name, tmpl in list(pool.items()):
        base = list(tmpl.tokens)

        # minor-dominant variants
        minor_dom: List[str] = []
        changed = False
        for t in base:
            if t == "V7":
                minor_dom.append("v7")
                changed = True
            elif t == "V":
                minor_dom.append("v")
                changed = True
            else:
                minor_dom.append(t)
        if changed:
            add_variant(name, minor_dom, ("minor_dom",), 0.75)

        # applied dominants inserted before up to 2 targets
        if len(base) <= 8:
            poss: List[int] = []
            for i in range(1, len(base)):
                t = str(base[i])
                if "/" in t:
                    continue
                if t.startswith(("V", "v", "subV", "vii", "VII")):
                    continue
                poss.append(i)
            poss = poss[:2]
            for i in poss:
                tgt = strip_target(base[i])
                for pre, tag in (("V7", "secDom"), ("subV7", "tritone"), ("vii°7", "leading")):
                    add_variant(name, base[:i] + [f"{pre}/{tgt}"] + base[i:], (tag,), 0.6)

        # cadence color if ends on I/i
        if len(base) >= 3 and strip_target(base[-1]) in ("I", "i"):
            cad1 = list(base)
            cad1[-2] = "V7"
            add_variant(name, cad1, ("cadence",), 0.7)
            cad2 = list(base)
            cad2[-2] = "subV7"
            add_variant(name, cad2, ("cadence", "tritone"), 0.6)

    # De-dupe identical token lists (keep highest weight)
    by_tokens: Dict[Tuple[str, ...], Tuple[str, ChordProgressionTemplate]] = {}
    for k, v in out.items():
        prev = by_tokens.get(v.tokens)
        if prev is None or float(v.weight) > float(prev[1].weight):
            by_tokens[v.tokens] = (k, v)
    return {k: v for k, v in by_tokens.values()}


CHORDPROGRESSION_POOL = _expand_chordprogression_pool(CHORDPROGRESSION_POOL)
CHORDPROGRESSION_CHOICES: Tuple[str, ...] = tuple(sorted(CHORDPROGRESSION_POOL.keys()))



# Debug/UX: surfaced in riff.txt/stdout so you can confirm what each label used.
LAST_LABEL_KINDS: Dict[str, str] = {}

# Vocals register (MIDI pitches)
VOCAL_LO_MIDI = 55  # ~G3
VOCAL_HI_MIDI = 76  # ~E5



# ----------------------------
# Data model
# ----------------------------

@dataclasses.dataclass(frozen=True)
class Event:
    start_beats: float
    dur_beats: float
    notes: List[Tuple[int, int]]  # (string, fret) per note (dyads allowed)
    velocity: int = 100


@dataclasses.dataclass(frozen=True)
class ScaleSpec:
    name: str
    intervals: Tuple[int, ...]
    degrees: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ChordDegree:
    degree_index: int
    root_pc: int
    triad_quality: str
    seventh_quality: str
    roman: str


@dataclasses.dataclass(frozen=True)
class MelodyContext:
    mode: str
    tonic_pc: int
    scale_pcs: Tuple[int, ...]
    degrees: Tuple[ChordDegree, ...]


@dataclasses.dataclass(frozen=True)
class VocalEvent:
    start_beats: float
    dur_beats: float
    pitch: int
    velocity: int = 80


@dataclasses.dataclass(frozen=True)
class PhraseData:
    kind: str
    events: List[Event]
    chords: List[str]  # chord labels per beat (len varies by section)
    ctx: MelodyContext
    tonic_midi: int



# ----------------------------
# Utilities
# ----------------------------

NOTE_TO_PC: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

PC_TO_NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_TO_NOTE_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def clampi(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def beats_to_seconds(bpm: int, beats: float) -> float:
    return float(beats) * 60.0 / float(bpm)


def parse_tone_to_pc(name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Empty tone")
    name = name[0].upper() + name[1:]
    if name not in NOTE_TO_PC:
        raise ValueError(f"Unknown tone: {name}")
    return int(NOTE_TO_PC[name])


def pc_to_note(pc: int, prefer_flats: bool = False) -> str:
    pc = int(pc) % 12
    return PC_TO_NOTE_FLAT[pc] if prefer_flats else PC_TO_NOTE_SHARP[pc]


def wrap_pitch(pitch: int, lo: int, hi: int) -> int:
    if lo > hi:
        lo, hi = hi, lo
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return int(clampi(pitch, lo, hi))


def string_fret_to_pitch(string: int, fret: int) -> int:
    return int(OPEN[int(string)] + int(fret))


def pitch_to_string_fret(pitch: int, prefer_low_strings: bool = True) -> Tuple[int, int]:
    candidates: List[Tuple[int, int]] = []
    for s in range(6, 0, -1):
        fret = int(pitch) - int(OPEN[s])
        if 0 <= fret <= 24:
            candidates.append((s, fret))
    if not candidates:
        s = 6 if prefer_low_strings else 1
        fret = clampi(int(pitch) - int(OPEN[s]), 0, 24)
        return (s, int(fret))

    # prefer low strings for "evil" feel
    if prefer_low_strings:
        candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    else:
        candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
    return candidates[0]


def choose_dyad_fingering(pitch_a: int, pitch_b: int) -> List[Tuple[int, int]]:
    pitch_a, pitch_b = int(pitch_a), int(pitch_b)
    if pitch_a > pitch_b:
        pitch_a, pitch_b = pitch_b, pitch_a

    best_cost = None
    best_pair: Optional[List[Tuple[int, int]]] = None

    for s1 in range(6, 0, -1):
        f1 = pitch_a - OPEN[s1]
        if not (0 <= f1 <= 24):
            continue
        for s2 in range(6, 0, -1):
            if s2 == s1:
                continue
            f2 = pitch_b - OPEN[s2]
            if not (0 <= f2 <= 24):
                continue
            spread = abs(int(f2) - int(f1))
            cost = spread + (6 - min(s1, s2)) * 0.9 + (f1 + f2) * 0.02
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pair = [(s1, int(f1)), (s2, int(f2))]

    if best_pair:
        best_pair.sort(key=lambda x: -x[0])
        return best_pair
    return [pitch_to_string_fret(pitch_a, True), pitch_to_string_fret(pitch_b, True)]


def powerchord_notes(root_pitch: int, inverted: bool) -> List[Tuple[int, int]]:
    root_pitch = int(root_pitch)
    lo, hi = int(PITCH_LO_MIDI), int(PITCH_HI_MIDI)
    fifth_up = root_pitch + 7
    fifth_dn = root_pitch - 5
    if fifth_up <= hi:
        fifth = fifth_up
    elif fifth_dn >= lo:
        fifth = fifth_dn
    else:
        fifth = fifth_up

    a, b = (fifth, root_pitch) if inverted else (root_pitch, fifth)
    return choose_dyad_fingering(a, b)



def _pitch_to_string_fret_options(pitch: int) -> List[Tuple[int, int]]:
    options: List[Tuple[int, int]] = []
    pitch = int(pitch)
    for s in range(6, 0, -1):
        fret = pitch - int(OPEN[s])
        if 0 <= fret <= 24:
            options.append((int(s), int(fret)))
    return options or [pitch_to_string_fret(pitch, True)]


def choose_poly_fingering(pitches: Sequence[int]) -> List[Tuple[int, int]]:
    """Pick a playable multi-note chord voicing (3–4 notes) on unique strings."""
    uniq = sorted(set(int(p) for p in pitches))
    if len(uniq) <= 2:
        if len(uniq) == 2:
            return choose_dyad_fingering(uniq[0], uniq[1])
        return [pitch_to_string_fret(uniq[0], True)] if uniq else []

    opts = [_pitch_to_string_fret_options(p) for p in uniq]
    best: Optional[List[Tuple[int, int]]] = None
    best_cost: Optional[float] = None

    def dfs(i: int, used: set, chosen: List[Tuple[int, int]]) -> None:
        nonlocal best, best_cost
        if i >= len(uniq):
            frets = [f for _, f in chosen]
            span = max(frets) - min(frets)
            low_bias = sum((6 - s) * (len(uniq) - idx) for idx, (s, _) in enumerate(chosen))
            cost = span * 1.0 + sum(frets) * 0.02 + low_bias * 0.25
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = list(chosen)
            return

        for s, f in opts[i]:
            if s in used:
                continue
            used.add(s)
            chosen.append((s, f))
            dfs(i + 1, used, chosen)
            chosen.pop()
            used.remove(s)

    dfs(0, set(), [])

    if best:
        best.sort(key=lambda x: -x[0])
        return best

    # fallback: allow collisions, then dedupe by keeping lowest fret on each string
    mapped = [pitch_to_string_fret(p, True) for p in uniq]
    by_string: Dict[int, int] = {}
    for s, f in mapped:
        by_string[int(s)] = min(int(f), by_string.get(int(s), 999))
    out = [(s, f) for s, f in by_string.items()]
    out.sort(key=lambda x: -x[0])
    return out


def _pc_positions_on_fretboard(pc: int) -> List[Tuple[int, int, int]]:
    """All (string, fret, pitch) positions for a pitch-class within 0..24 frets."""
    pc = int(pc) % 12
    out: List[Tuple[int, int, int]] = []
    for s in range(6, 0, -1):
        base = int(OPEN[s])
        for f in range(0, 25):
            p = base + int(f)
            if p % 12 == pc:
                out.append((int(s), int(f), int(p)))
    return out


def choose_poly_fingering_strict(pcs: Sequence[int], *, prefer_root_bass: bool = True) -> List[Tuple[int, int]]:
    """Return a triad/tetrad voicing (exactly len(pcs) notes) on unique strings.

    Unlike choose_poly_fingering(), this NEVER collapses to dyads/single-notes.
    It searches the whole 0..24 fretboard for each pitch-class and picks a low, compact voicing.
    """
    uniq_pcs = [int(p) % 12 for p in pcs]
    if len(uniq_pcs) < 3:
        raise ValueError("choose_poly_fingering_strict requires triads/tetrads (>=3 pcs)")

    root_pc = int(uniq_pcs[0]) % 12
    opts = [_pc_positions_on_fretboard(pc) for pc in uniq_pcs]

    best: Optional[List[Tuple[int, int, int]]] = None
    best_cost: Optional[float] = None

    def dfs(i: int, used: set, chosen: List[Tuple[int, int, int]]) -> None:
        nonlocal best, best_cost
        if i >= len(opts):
            pitches = [p for _, _, p in chosen]
            frets = [f for _, f, _ in chosen]
            strings = [s for s, _, _ in chosen]
            pitch_span = max(pitches) - min(pitches)
            fret_span = max(frets) - min(frets)
            low_bias = sum((6 - s) * (len(opts) - idx) for idx, s in enumerate(strings))
            inv_penalty = 0.0
            if prefer_root_bass:
                bass_pitch = min(pitches)
                bass_pc = int(bass_pitch) % 12
                inv_penalty = 6.0 if bass_pc != root_pc else 0.0
            cost = pitch_span * 0.55 + fret_span * 1.0 + sum(frets) * 0.02 + low_bias * 0.22 + inv_penalty
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = list(chosen)
            return

        # Prefer low strings + low frets
        cand = sorted(opts[i], key=lambda x: ((6 - x[0]) * 2.2 + x[1] * 0.55))
        for s, f, p in cand[:60]:
            if s in used:
                continue
            used.add(s)
            chosen.append((s, f, p))
            dfs(i + 1, used, chosen)
            chosen.pop()
            used.remove(s)

    dfs(0, set(), [])
    if not best:
        # Extremely defensive fallback: pick the lowest available unique-string position per pc.
        out: List[Tuple[int, int]] = []
        used = set()
        for pc in uniq_pcs:
            cand = sorted(_pc_positions_on_fretboard(pc), key=lambda x: ((6 - x[0]) * 2.2 + x[1] * 0.55))
            pick = None
            for s, f, _ in cand:
                if s not in used:
                    pick = (s, f)
                    used.add(s)
                    break
            if pick is None:
                # last resort: allow reuse of a string (still returns correct note count)
                s, f, _ = cand[0]
                pick = (s, f)
            out.append((int(pick[0]), int(pick[1])))
        out.sort(key=lambda x: -x[0])
        return out

    out2 = [(int(s), int(f)) for s, f, _ in best]
    out2.sort(key=lambda x: -x[0])
    return out2


def _pitches_in_range_for_pc(pc: int, lo: int, hi: int) -> List[int]:
    pc = int(pc) % 12
    return [p for p in range(int(lo), int(hi) + 1) if p % 12 == pc]


def _lowest_string6_pitch_for_pc(pc: int) -> int:
    pc = int(pc) % 12
    for fret in range(0, 25):
        p = int(OPEN[6] + fret)
        if p % 12 == pc:
            return int(wrap_pitch(p, int(PITCH_LO_MIDI), int(PITCH_HI_MIDI)))
    return int(PITCH_LO_MIDI)


# ----------------------------
# Scales / modes
# ----------------------------

MELODY_SCALES: Dict[str, ScaleSpec] = {
    # Diatonic
    "ionian": ScaleSpec("ionian", (0, 2, 4, 5, 7, 9, 11), ("I", "II", "III", "IV", "V", "VI", "VII")),
    "dorian": ScaleSpec("dorian", (0, 2, 3, 5, 7, 9, 10), ("I", "II", "bIII", "IV", "V", "VI", "bVII")),
    "phrygian": ScaleSpec("phrygian", (0, 1, 3, 5, 7, 8, 10), ("I", "bII", "bIII", "IV", "V", "bVI", "bVII")),
    "lydian": ScaleSpec("lydian", (0, 2, 4, 6, 7, 9, 11), ("I", "II", "III", "#IV", "V", "VI", "VII")),
    "mixolydian": ScaleSpec("mixolydian", (0, 2, 4, 5, 7, 9, 10), ("I", "II", "III", "IV", "V", "VI", "bVII")),
    "aeolian": ScaleSpec("aeolian", (0, 2, 3, 5, 7, 8, 10), ("I", "II", "bIII", "IV", "V", "bVI", "bVII")),
    "locrian": ScaleSpec("locrian", (0, 1, 3, 5, 6, 8, 10), ("I", "bII", "bIII", "IV", "bV", "bVI", "bVII")),

    # Harmonic/melodic minor & colors
    "harmonic_minor": ScaleSpec("harmonic_minor", (0, 2, 3, 5, 7, 8, 11), ("I", "II", "bIII", "IV", "V", "bVI", "VII")),
    "melodic_minor": ScaleSpec("melodic_minor", (0, 2, 3, 5, 7, 9, 11), ("I", "II", "bIII", "IV", "V", "VI", "VII")),
    "phrygian_dominant": ScaleSpec("phrygian_dominant", (0, 1, 4, 5, 7, 8, 10), ("I", "bII", "III", "IV", "V", "bVI", "bVII")),
    "hungarian_minor": ScaleSpec("hungarian_minor", (0, 2, 3, 6, 7, 8, 11), ("I", "II", "bIII", "#IV", "V", "bVI", "VII")),
    "double_harmonic": ScaleSpec("double_harmonic", (0, 1, 4, 5, 7, 8, 11), ("I", "bII", "III", "IV", "V", "bVI", "VII")),
    "persian": ScaleSpec("persian", (0, 1, 4, 5, 6, 8, 11), ("I", "bII", "III", "IV", "bV", "bVI", "VII")),
    "enigmatic": ScaleSpec("enigmatic", (0, 1, 4, 6, 8, 10, 11), ("I", "bII", "III", "#IV", "#V", "bVII", "VII")),
    "altered": ScaleSpec("altered", (0, 1, 3, 4, 6, 8, 10), ("I", "bII", "bIII", "III", "bV", "bVI", "bVII")),

    # Symmetric
    "diminished_hw": ScaleSpec("diminished_hw", (0, 1, 3, 4, 6, 7, 9, 10), ("I", "bII", "bIII", "III", "#IV", "V", "VI", "bVII")),
    "whole_tone": ScaleSpec("whole_tone", (0, 2, 4, 6, 8, 10), ("I", "II", "III", "#IV", "#V", "bVII")),
}

HAPPY_RANDOM_EXCLUDE = {"ionian", "mixolydian", "lydian"}


def random_dark_mode(rng: random.Random) -> str:
    choices = [m for m in MELODY_SCALES.keys() if m not in HAPPY_RANDOM_EXCLUDE]
    return rng.choice(choices)


# ----------------------------
# Diatonic chord derivation
# ----------------------------

def _triad_quality(third: int, fifth: int) -> str:
    third %= 12
    fifth %= 12
    if third == 4 and fifth == 7:
        return "maj"
    if third == 3 and fifth == 7:
        return "min"
    if third == 3 and fifth == 6:
        return "dim"
    if third == 4 and fifth == 8:
        return "aug"
    return "unk"


def _seventh_quality(triad: str, seventh: int) -> str:
    seventh %= 12
    if triad == "maj" and seventh == 11:
        return "maj7"
    if triad == "maj" and seventh == 10:
        return "7"
    if triad == "min" and seventh == 10:
        return "min7"
    if triad == "min" and seventh == 11:
        return "mMaj7"
    if triad == "dim" and seventh == 10:
        return "m7b5"
    if triad == "dim" and seventh == 9:
        return "dim7"
    if triad == "aug" and seventh == 11:
        return "maj7#5"
    if triad == "aug" and seventh == 10:
        return "7#5"
    return "7"


def _roman_for_degree(scale: ScaleSpec, degree: int, triad: str) -> str:
    lab = scale.degrees[degree]
    if triad in ("maj", "aug"):
        return lab
    if triad in ("min", "dim"):
        return lab.lower()
    return lab


def build_melody_context(mode: str, tonic_pc: int) -> MelodyContext:
    mode = str(mode)
    if mode not in MELODY_SCALES:
        raise ValueError(f"Unknown melody mode: {mode}")
    spec = MELODY_SCALES[mode]
    pcs = tuple((int(tonic_pc) + i) % 12 for i in spec.intervals)

    degrees: List[ChordDegree] = []
    n = len(pcs)
    for d in range(n):
        root = pcs[d]
        third = pcs[(d + 2) % n]
        fifth = pcs[(d + 4) % n]
        sev = pcs[(d + 6) % n] if n >= 7 else pcs[(d + 1) % n]
        triad = _triad_quality(third - root, fifth - root)
        seventh = _seventh_quality(triad, sev - root)
        roman = _roman_for_degree(spec, d, triad)
        degrees.append(ChordDegree(d, root, triad, seventh, roman))
    return MelodyContext(mode=mode, tonic_pc=int(tonic_pc), scale_pcs=pcs, degrees=tuple(degrees))


def _mode_characteristic_pcs(ctx: MelodyContext) -> Tuple[int, ...]:
    mode = str(ctx.mode).lower()
    tonic = int(ctx.tonic_pc) % 12
    pcs = tuple(int(p) % 12 for p in ctx.scale_pcs)

    char_intervals = {
        "dorian": {9},
        "phrygian": {1},
        "lydian": {6},
        "mixolydian": {10},
        "aeolian": {8},
        "locrian": {1, 6},
        "harmonic_minor": {11},
        "melodic_minor": {9, 11},
        "phrygian_dominant": {1, 4},
        "hungarian_minor": {6, 11},
        "double_harmonic": {1, 4, 11},
        "persian": {1, 6, 11},
        "enigmatic": {1, 6, 8, 11},
        "altered": {1, 3, 6, 8, 10},
        "diminished_hw": {1, 6, 10},
        "whole_tone": {6, 8},
    }.get(mode, set())

    if not char_intervals:
        return tuple()

    out = []
    for p in pcs:
        iv = (p - tonic) % 12
        if iv in char_intervals:
            out.append(p)
    return tuple(sorted(set(out)))


# ----------------------------
# Rhythm primitives
# ----------------------------

def build_melody_rhythm_slots(rng: random.Random) -> List[Tuple[float, float]]:
    """No rests; durations >= 0.5 beats; grid 0.5 beats.

    Biased toward 8ths to keep the melody moving.
    """
    slots: List[Tuple[float, float]] = []
    t = 0.0
    while t < 4.0 - 1e-9:
        rem = 4.0 - t
        choices: List[float] = []
        weights: List[float] = []
        if rem >= 4.0:
            choices.append(4.0); weights.append(0.03)
        if rem >= 3.0:
            choices.append(3.0); weights.append(0.03)
        if rem >= 2.0:
            choices.append(2.0); weights.append(0.14)
        if rem >= 1.0:
            choices.append(1.0); weights.append(0.30)
        if rem >= 0.5:
            choices.append(0.5); weights.append(0.50)  # more 8ths
        dur = float(rng.choices(choices, weights=weights, k=1)[0])
        dur = round(dur * 2.0) / 2.0
        slots.append((t, dur))
        t = round((t + dur) * 2.0) / 2.0

    if slots:
        s, d = slots[-1]
        if s + d > 4.0:
            slots[-1] = (s, max(0.5, 4.0 - s))
    return slots

def harmonic_windows_for_bar(rng: random.Random, *, prefer_long: bool) -> List[Tuple[float, float]]:
    t = 0.0
    out: List[Tuple[float, float]] = []
    while t < 4.0 - 1e-9:
        rem = 4.0 - t
        choices = [c for c in (4.0, 3.0, 2.0, 1.0, 0.5) if c <= rem + 1e-9]
        if not choices:
            break
        if prefer_long:
            w = {4.0: 0.22, 3.0: 0.14, 2.0: 0.46, 1.0: 0.16, 0.5: 0.02}
        else:
            w = {4.0: 0.06, 3.0: 0.08, 2.0: 0.20, 1.0: 0.40, 0.5: 0.26}
        weights = [w[float(c)] for c in choices]
        dur = float(rng.choices(choices, weights=weights, k=1)[0])
        dur = round(dur * 2.0) / 2.0
        out.append((t, dur))
        t = round((t + dur) * 2.0) / 2.0

    if out:
        s, d = out[-1]
        if s + d < 4.0 - 1e-9:
            out.append((s + d, round((4.0 - (s + d)) * 2.0) / 2.0))
    return out


def choose_degree_progression(rng: random.Random, ctx: MelodyContext, length: int) -> List[int]:
    n = len(ctx.scale_pcs)
    deg = rng.randrange(n)
    out = [deg]
    for _ in range(length - 1):
        r = rng.random()
        if r < 0.65:
            step = rng.choice([-1, 1])
        elif r < 0.90:
            step = rng.choice([-2, 2])
        else:
            step = rng.choice([-3, 3])
        deg = (deg + step) % n
        out.append(deg)
    return out


def degree_root_pitch(ctx: MelodyContext, degree: int, base_tonic_midi: int) -> int:
    root_pc = int(ctx.degrees[int(degree)].root_pc) % 12
    candidates = _pitches_in_range_for_pc(root_pc, int(PITCH_LO_MIDI), int(PITCH_HI_MIDI))
    if not candidates:
        return int(PITCH_LO_MIDI)
    base = int(base_tonic_midi)
    candidates.sort(key=lambda p: (p - int(PITCH_LO_MIDI)) * 0.8 + abs(int(p) - base) * 0.5)
    return int(candidates[0])


def choose_pitch_for_pc(rng: random.Random, pc: int, last_pitch: Optional[int]) -> int:
    candidates = _pitches_in_range_for_pc(int(pc), int(PITCH_LO_MIDI), int(PITCH_HI_MIDI))
    if not candidates:
        return int(PITCH_LO_MIDI)

    def cost(p: int) -> float:
        low_bias = (p - int(PITCH_LO_MIDI)) * 0.7
        if last_pitch is None:
            return low_bias
        return low_bias + abs(int(p) - int(last_pitch)) * 1.1

    candidates.sort(key=cost)
    top = candidates[: min(5, len(candidates))]
    weights = [1.0 / (1.0 + i) for i in range(len(top))]
    return int(rng.choices(top, weights=weights, k=1)[0])


# ----------------------------
# Motif helpers
# ----------------------------

def copy_bar(events: List[Event], src_bar: int, dst_bar: int) -> List[Event]:
    shift = float((dst_bar - src_bar) * 4)
    out: List[Event] = []
    for e in events:
        if int(e.start_beats // 4) != int(src_bar):
            continue
        out.append(Event(start_beats=float(e.start_beats) + shift, dur_beats=e.dur_beats, notes=list(e.notes), velocity=e.velocity))
    return out


def shift_events(events: List[Event], shift_beats: float) -> List[Event]:
    return [Event(start_beats=float(e.start_beats) + float(shift_beats), dur_beats=e.dur_beats, notes=list(e.notes), velocity=e.velocity) for e in events]


def seed_for_label(seed: int, label: str) -> int:
    h = 2166136261
    for ch in (str(seed) + "|" + label):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def trim_to_bars(events: List[Event], bar_sections: List[str], bars: int) -> Tuple[List[Event], List[str]]:
    limit = float(int(bars) * 4)
    out = [e for e in events if float(e.start_beats) < limit]
    out.sort(key=lambda e: e.start_beats)
    return out, bar_sections[: int(bars)]

def _apply_dim_interval_to_events(events: List[Event]) -> List[Event]:
    """Convert single-note events into diminished-interval dyads (tritone).

    Used by the 'justbursts' section to enforce diminished intervals on bar 4 and bar 8
    of the 8-bar phrase (contrast bars).
    """
    out: List[Event] = []
    for e in events:
        if not e.notes:
            out.append(e)
            continue
        s0, f0 = e.notes[0]
        p0 = string_fret_to_pitch(int(s0), int(f0))
        p1 = int(p0) + 6
        dyad = choose_dyad_fingering(int(p0), int(p1))
        out.append(Event(start_beats=e.start_beats, dur_beats=e.dur_beats, notes=dyad, velocity=e.velocity))
    return out


# ----------------------------
# Section generators
# ----------------------------

def generate_melodies_bar(
    *,
    bar_index: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    degree_plan: List[int],
    resolve: bool,
) -> Tuple[List[Event], List[str]]:
    rng = random.Random(int(seed) ^ (bar_index * 10007) ^ 0xC0FFEE)
    windows = harmonic_windows_for_bar(rng, prefer_long=False)
    n_deg = len(ctx.degrees)

    win_degs: List[int] = []
    prev = None
    for i, _ in enumerate(windows):
        deg = int(degree_plan[i % max(1, len(degree_plan))]) % n_deg
        if rng.random() < 0.55:
            deg = (deg + rng.choice([-3, -2, -1, 1, 2, 3])) % n_deg
        if prev is not None and deg == prev:
            deg = (deg + rng.choice([-1, 1])) % n_deg
        win_degs.append(deg)
        prev = deg
    if resolve and win_degs:
        win_degs[-1] = 0

    chords_per_beat: List[str] = [""] * 4
    for beat in range(4):
        t = float(beat)
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= t < ws + wd - 1e-9:
                wi = j
                break
        cd = ctx.degrees[win_degs[wi]]
        chords_per_beat[beat] = f"{cd.roman}:{pc_to_note(cd.root_pc, True)}"

    slots = build_melody_rhythm_slots(rng)
    events: List[Event] = []
    last_pitch: Optional[int] = None
    char_pcs = list(_mode_characteristic_pcs(ctx))
    scale_pcs = set(int(p) % 12 for p in ctx.scale_pcs)

    for (s, d) in slots:
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= s < ws + wd - 1e-9:
                wi = j
                break
        deg = int(win_degs[wi])
        cd = ctx.degrees[deg]

        # chord pcs
        chord_pcs = {int(cd.root_pc) % 12}
        chord_pcs.add((int(cd.root_pc) + (4 if cd.triad_quality in ("maj", "aug") else 3)) % 12)
        if cd.triad_quality == "aug":
            chord_pcs.add((int(cd.root_pc) + 8) % 12)
        elif cd.triad_quality == "dim":
            chord_pcs.add((int(cd.root_pc) + 6) % 12)
        else:
            chord_pcs.add((int(cd.root_pc) + 7) % 12)

        r = rng.random()
        if char_pcs and r < 0.28:
            pc = int(rng.choice(char_pcs)) % 12
        elif r < 0.62:
            pc = int(rng.choice(sorted(chord_pcs))) % 12
        else:
            pc = int(rng.choice(sorted(scale_pcs))) % 12

        pitch = choose_pitch_for_pc(rng, pc, last_pitch)
        last_pitch = int(pitch)

        want_dyad = (d >= 1.0) and (rng.random() < (0.45 if d >= 2.0 else 0.30))
        if want_dyad:
            inverted = rng.random() < 0.40
            notes = powerchord_notes(int(pitch), inverted=inverted)
        else:
            notes = [pitch_to_string_fret(int(pitch), prefer_low_strings=True)]

        events.append(Event(start_beats=float(bar_index * 4 + s), dur_beats=float(d), notes=notes, velocity=int(vel)))

    if resolve:
        tonic_pitch = degree_root_pitch(ctx, 0, base_tonic_midi)
        events.append(Event(start_beats=float(bar_index * 4 + 3.0), dur_beats=1.0, notes=powerchord_notes(int(tonic_pitch), inverted=False), velocity=int(vel)))

    return events, chords_per_beat


def generate_rhythm_bar(
    *,
    section: str,
    bar_index: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    degree_plan: List[int],
    power_bias: float,
    resolve: bool,
) -> Tuple[List[Event], List[str]]:
    """Generate one 4/4 rhythm bar.

    section:
      - gallops:      111x (3 sixteenths + rest), powerchords are palm-muted (velocity=45)
      - bursts:       1111 (4 sixteenths),        powerchords are palm-muted (velocity=45)
      - offbeatgallops: 1x11 (note, rest, note, note), 16ths are palm-muted (velocity=45),
                        and the 1x11 pattern always appears in runs of >= 2 consecutive beats.
      - gallopsopen:  same as gallops, but powerchords use normal velocity (vel)
      - burstsopen:   same as bursts,  but powerchords use normal velocity (vel)
    """
    section = str(section)
    rng = random.Random(int(seed) ^ (bar_index * 7331) ^ 0xBADA55)

    windows = harmonic_windows_for_bar(rng, prefer_long=False)
    n_deg = len(ctx.degrees)
    tonic_pitch = degree_root_pitch(ctx, 0, base_tonic_midi)

    palm_vel = int(PALM_MUTE_VELOCITY)
    normal_vel = int(vel)

    if section in ("gallops", "bursts"):
        chord_vel = palm_vel
        single_vel = palm_vel
    elif section in ("gallopsopen", "burstsopen"):
        chord_vel = normal_vel
        single_vel = palm_vel
    elif section == "offbeatgallops":
        chord_vel = normal_vel
        single_vel = palm_vel
    else:
        chord_vel = normal_vel
        single_vel = normal_vel

    rhythm_sections = {"gallops", "bursts", "offbeatgallops", "gallopsopen", "burstsopen"}

    # Slightly less tonic-overuse than before, but still biased.
    tonic_boost = 0.14 if section in rhythm_sections else 0.10
    if section in ("gallops", "gallopsopen"):
        root_bias = 0.62
    elif section in ("bursts", "burstsopen"):
        root_bias = 0.60
    elif section == "offbeatgallops":
        root_bias = 0.56
    else:
        root_bias = 0.45

    win_degs: List[int] = []
    prev: Optional[int] = None
    for i, _ in enumerate(windows):
        deg = int(degree_plan[i % max(1, len(degree_plan))]) % n_deg
        if rng.random() < 0.45:
            deg = (deg + rng.choice([-2, -1, 1, 2])) % n_deg
        if rng.random() < float(tonic_boost):
            deg = 0
        if prev is not None and deg == prev:
            deg = (deg + rng.choice([-1, 1])) % n_deg
        win_degs.append(int(deg))
        prev = int(deg)
    if resolve and win_degs:
        win_degs[-1] = 0

    chords_per_beat: List[str] = ["" for _ in range(4)]
    for beat in range(4):
        t = float(beat)
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= t < ws + wd - 1e-9:
                wi = j
                break
        cd = ctx.degrees[win_degs[wi]]
        chords_per_beat[beat] = f"{cd.roman}:{pc_to_note(cd.root_pc, True)}"

    events: List[Event] = []
    cluster = 0
    offbeat_run_remaining = 0

    for beat in range(4):
        t = float(beat)
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= t < ws + wd - 1e-9:
                wi = j
                break

        deg = int(win_degs[wi])
        if resolve and beat >= 3:
            deg = 0

        base_pitch = degree_root_pitch(ctx, deg, base_tonic_midi)

        # Offbeat-gallops must be >=2 consecutive beats of 1x11.
        beats_left = 4 - beat
        force_pattern = section == "offbeatgallops" and offbeat_run_remaining > 0
        if force_pattern:
            use_power = False
            cluster = 0
            offbeat_run_remaining -= 1
        else:
            use_power = (rng.random() < float(power_bias)) or (cluster > 0)
            if section == "offbeatgallops" and beat == 3:
                # Avoid starting a 1-beat run on the last beat.
                use_power = True

        if use_power:
            if rng.random() < 0.80:
                cluster = clampi(cluster + 1, 0, 6)
            else:
                cluster = clampi(cluster - 1, 0, 6)

            inverted = rng.random() < 0.45
            r = rng.random()
            if r < 0.78:
                dur = 1.0
            elif r < 0.95:
                dur = 2.0
            else:
                dur = 4.0 if beat == 0 else 2.0

            power_tonic_prob = 0.22 if section in rhythm_sections else 0.0
            pitch = tonic_pitch if (rng.random() < power_tonic_prob) else base_pitch

            events.append(
                Event(
                    start_beats=float(bar_index * 4 + beat),
                    dur_beats=float(dur),
                    notes=powerchord_notes(int(pitch), inverted=inverted),
                    velocity=int(chord_vel),
                )
            )
            continue

        # 16th patterns
        if section == "offbeatgallops":
            # 1x11 (rest is 2nd 16th)
            starts = [beat + 0.00, beat + 0.50, beat + 0.75]
        elif section in ("bursts", "burstsopen"):
            starts = [beat + 0.00, beat + 0.25, beat + 0.50, beat + 0.75]
        else:  # gallops / gallopsopen
            starts = [beat + 0.00, beat + 0.25, beat + 0.50]

        pitch = tonic_pitch if rng.random() < float(root_bias) else base_pitch
        for st in starts:
            events.append(
                Event(
                    start_beats=float(bar_index * 4 + st),
                    dur_beats=0.25,
                    notes=[pitch_to_string_fret(int(pitch), True)],
                    velocity=int(single_vel),
                )
            )

        if section == "offbeatgallops" and not force_pattern:
            # We started a new run: total run length 2..3 beats, clipped to bar end.
            total = int(rng.choice([2, 3]))
            offbeat_run_remaining = max(0, min(total - 1, beats_left - 1))

    return events, chords_per_beat

def evil_progression_plan(rng: random.Random, ctx: MelodyContext) -> List[int]:
    n = len(ctx.scale_pcs)
    if n < 7:
        plan = choose_degree_progression(rng, ctx, 8)
        plan[3] = 0
        plan[7] = 0
        return plan

    candidates = [
        [0, 6, 5, 0, 0, 6, 5, 0],  # andalusian-ish with resolve
        [0, 1, 0, 0, 0, 1, 6, 0],  # phrygian vamp-ish
        [0, 3, 5, 0, 0, 3, 5, 0],  # i-iv-bVI-V-ish -> resolve
        [0, 5, 3, 0, 0, 5, 6, 0],
        [0, 2, 5, 0, 0, 2, 6, 0],
    ]
    plan = list(rng.choice(candidates))
    plan[3] = 0
    plan[7] = 0
    if ctx.mode == "locrian" and rng.random() < 0.6:
        plan[1] = 1
        plan[5] = 1
    # make bar7 lean to V for "prepare resolve"
    plan[6] = 4 if n >= 5 else plan[6]
    return [int(x) % n for x in plan]


def generate_chords_bar(
    *,
    bar_index: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    bar_degree: int,
    resolve: bool,
) -> Tuple[List[Event], List[str]]:
    rng = random.Random(int(seed) ^ (bar_index * 9001) ^ 0x1234)
    windows = harmonic_windows_for_bar(rng, prefer_long=True)
    n_deg = len(ctx.degrees)

    win_degs: List[int] = []
    prev = None
    deg0 = int(bar_degree) % n_deg
    for _ws, _wd in windows:
        deg = deg0
        if rng.random() < 0.78:
            deg = (deg + rng.choice([-3, -2, -1, 1, 2, 3])) % n_deg
        if prev is not None and deg == prev:
            deg = (deg + rng.choice([-2, -1, 1, 2])) % n_deg
        win_degs.append(deg)
        prev = deg
        deg0 = deg

    if resolve and win_degs:
        win_degs[-1] = 0

    chords_per_beat: List[str] = [""] * 4
    for beat in range(4):
        t = float(beat)
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= t < ws + wd - 1e-9:
                wi = j
                break
        cd = ctx.degrees[win_degs[wi]]
        chords_per_beat[beat] = f"{cd.roman}:{pc_to_note(cd.root_pc, True)}"

    events: List[Event] = []
    last_deg: Optional[int] = None

    for (ws, wd), deg in zip(windows, win_degs):
        deg = int(deg) % n_deg
        if last_deg is not None and deg == last_deg and rng.random() < 0.88:
            deg = (deg + rng.choice([-2, -1, 1, 2])) % n_deg
        if resolve and (ws + wd) >= 4.0 - 1e-9:
            deg = 0

        root_pitch = degree_root_pitch(ctx, deg, base_tonic_midi)
        inverted = rng.random() < 0.50

        # rare tritone dyad
        if rng.random() < (0.10 if resolve else 0.06) and wd >= 1.0:
            notes = choose_dyad_fingering(int(root_pitch), int(root_pitch + 6))
        else:
            notes = powerchord_notes(int(root_pitch), inverted=inverted)

        events.append(Event(start_beats=float(bar_index * 4 + ws), dur_beats=float(wd), notes=notes, velocity=int(vel)))
        last_deg = deg

    return events, chords_per_beat


def generate_downpicking_bar(
    *,
    bar_index: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    degree_plan: List[int],
    resolve: bool,
) -> Tuple[List[Event], List[str]]:
    """Generate one downpicking bar (only downpicks + powerchords).

    - Default texture: 2 palm-muted eighths (velocity=45) per beat.
    - Occasionally replaces a beat with a powerchord hit at normal velocity (vel).
    """
    rng = random.Random(int(seed) ^ (bar_index * 9001) ^ 0xD00DCAFE)
    windows = harmonic_windows_for_bar(rng, prefer_long=False)
    n_deg = len(ctx.degrees)

    tonic_pitch = degree_root_pitch(ctx, 0, base_tonic_midi)
    palm_vel = int(PALM_MUTE_VELOCITY)
    normal_vel = int(vel)

    powerchord_prob = 0.30  # per beat

    def pick_deg(i: int) -> int:
        if not degree_plan:
            return 0
        d = int(degree_plan[i % len(degree_plan)]) % n_deg
        # Slightly less tonic-overuse than before.
        if rng.random() < 0.18:
            d = 0
        elif rng.random() < 0.35:
            d = (d + rng.choice([-2, -1, 1, 2])) % n_deg
        return int(d) % n_deg

    win_degs: List[int] = []
    prev: Optional[int] = None
    for i, _ in enumerate(windows):
        d = pick_deg(i)
        if prev is not None and d == prev:
            d = (d + rng.choice([-1, 1])) % n_deg
        win_degs.append(int(d))
        prev = int(d)
    if resolve and win_degs:
        win_degs[-1] = 0

    chords_per_beat: List[str] = ["" for _ in range(4)]
    for beat in range(4):
        t = float(beat)
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= t < ws + wd - 1e-9:
                wi = j
                break
        cd = ctx.degrees[win_degs[wi]]
        chords_per_beat[beat] = f"{cd.roman}:{pc_to_note(cd.root_pc, True)}"

    events: List[Event] = []

    for beat in range(4):
        t0 = float(bar_index * 4 + beat)

        t = float(beat)
        wi = 0
        for j, (ws, wd) in enumerate(windows):
            if ws <= t < ws + wd - 1e-9:
                wi = j
                break
        deg = int(win_degs[wi])
        if resolve and beat >= 3:
            deg = 0

        base_pitch = degree_root_pitch(ctx, deg, base_tonic_midi)

        # Reduce "tone note" dominance slightly, but keep it present.
        single_pitch = tonic_pitch if rng.random() < 0.50 else base_pitch

        if rng.random() < float(powerchord_prob):
            inverted = rng.random() < 0.40
            r = rng.random()
            if r < 0.80:
                dur = 1.0
            elif r < 0.95:
                dur = 2.0
            else:
                dur = 4.0 if beat == 0 else 2.0

            events.append(
                Event(
                    start_beats=t0,
                    dur_beats=float(dur),
                    notes=powerchord_notes(int(base_pitch), inverted=inverted),
                    velocity=int(normal_vel),
                )
            )
            continue

        # 2 palm-muted eighths (downpicked)
        events.append(
            Event(
                start_beats=t0,
                dur_beats=0.5,
                notes=[pitch_to_string_fret(int(single_pitch), True)],
                velocity=int(palm_vel),
            )
        )
        events.append(
            Event(
                start_beats=t0 + 0.5,
                dur_beats=0.5,
                notes=[pitch_to_string_fret(int(single_pitch), True)],
                velocity=int(palm_vel),
            )
        )

    if resolve:
        events.append(
            Event(
                start_beats=float(bar_index * 4 + 3.0),
                dur_beats=1.0,
                notes=powerchord_notes(int(tonic_pitch), inverted=False),
                velocity=int(normal_vel),
            )
        )

    events.sort(key=lambda e: e.start_beats)
    return events, chords_per_beat

def generate_pedalpoint_phrase_8(
    *,
    bar_start: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
) -> Tuple[List[Event], List[str]]:
    """Pedal point phrase with alternating 8ths.

    The pedal root varies inside each bar (blocks of 1/2/3/4 beats) so the pedal note
    isn't dominated by the tonic across the whole phrase. Pedal strokes are palm-muted
    (velocity=45); the alternating "other" notes are normal velocity.
    """
    rng = random.Random(int(seed) ^ 0xC0FFEE99)
    n_deg = len(ctx.degrees)

    pedal_candidates = [d for d in (0, 4, 5, 3, 2) if d < n_deg]
    if not pedal_candidates:
        pedal_candidates = [0]

    tonic_w = float(PEDALPOINT_TONIC_WEIGHT)

    def pick_pedal_deg(exclude: Optional[int] = None) -> int:
        choices = list(pedal_candidates)
        if exclude is not None and exclude in choices and len(choices) > 1:
            choices = [d for d in choices if d != exclude] or list(pedal_candidates)
        if len(choices) <= 1:
            return int(choices[0])
        rest = max(0.0, 1.0 - tonic_w)
        weights = []
        for d in choices:
            weights.append(tonic_w if d == 0 else rest / float(len(choices) - (1 if 0 in choices else 0)) if (0 in choices) else 1.0)
        # If tonic isn't present in choices, normalize to equal weights.
        if 0 not in choices:
            weights = [1.0 for _ in choices]
        return int(rng.choices(choices, weights=weights, k=1)[0])

    # Default "metal pedal" color degrees; optionally add a bit more.
    base_other = [d for d in (1, 3, 4, 5) if d < n_deg]
    extra_other = [d for d in (2, 6) if d < n_deg]
    candidate_other_degs = list(base_other)
    if extra_other and rng.random() < 0.35:
        candidate_other_degs.extend(extra_other)
    if not candidate_other_degs:
        candidate_other_degs = [d for d in range(n_deg)]

    events: List[Event] = []
    chord_grid: List[str] = []

    start_with_other = rng.random() < 0.45
    palm_vel = int(PALM_MUTE_VELOCITY)
    normal_vel = int(vel)
    lo, hi = int(PITCH_LO_MIDI), int(PITCH_HI_MIDI)

    for b in range(8):
        # Build a per-beat pedal root map for this bar (varies by 1/2/3/4 beat blocks).
        block_beats = int(rng.choice([1, 2, 3, 4]))
        pedal_deg_per_beat: List[int] = []
        beat = 0
        last_deg: Optional[int] = None
        while beat < 4:
            avoid = last_deg if (last_deg is not None and rng.random() < 0.70) else None
            deg = pick_pedal_deg(exclude=avoid)
            for _ in range(block_beats):
                if beat >= 4:
                    break
                pedal_deg_per_beat.append(int(deg))
                beat += 1
            last_deg = int(deg)

        if len(set(pedal_deg_per_beat)) == 1 and block_beats < 4 and len(pedal_candidates) > 1:
            pedal_deg_per_beat[-1] = pick_pedal_deg(exclude=pedal_deg_per_beat[-1])

        for beat_i in range(4):
            cd = ctx.degrees[int(pedal_deg_per_beat[beat_i])]
            chord_grid.append(f"{cd.roman}:{pc_to_note(cd.root_pc, True)}")

        bar0 = float((bar_start + b) * 4)
        for i in range(8):  # 8 eighth-notes
            t = bar0 + i * 0.5
            beat_i = int(i * 0.5)  # 0..3
            pedal_deg = int(pedal_deg_per_beat[beat_i])
            pedal_pc = int(ctx.degrees[pedal_deg].root_pc) % 12
            pedal_pitch = int(degree_root_pitch(ctx, pedal_deg, base_tonic_midi))

            is_other = ((i % 2) == 0 and start_with_other) or ((i % 2) == 1 and not start_with_other)
            if not is_other:
                events.append(
                    Event(
                        start_beats=t,
                        dur_beats=0.5,
                        notes=[pitch_to_string_fret(int(pedal_pitch), True)],
                        velocity=int(palm_vel),
                    )
                )
                continue

            # Pick a non-pedal color degree for the "other" hit.
            other_pool = [d for d in candidate_other_degs if d != pedal_deg] or [d for d in range(n_deg) if d != pedal_deg] or [pedal_deg]
            other_deg = int(rng.choice(other_pool)) % n_deg
            other_pc = int(ctx.degrees[other_deg].root_pc) % 12
            if other_pc == pedal_pc and n_deg > 1:
                other_deg = int((other_deg + 1) % n_deg)
                other_pc = int(ctx.degrees[other_deg].root_pc) % 12

            candidates = [p for p in _pitches_in_range_for_pc(other_pc, lo, hi) if int(p) >= int(pedal_pitch)]
            if not candidates:
                candidates = _pitches_in_range_for_pc(other_pc, lo, hi)
            if not candidates:
                pitch = int(pedal_pitch)
            else:
                candidates.sort(key=lambda p: abs(int(p) - int(pedal_pitch)))
                pitch = int(candidates[0])

            events.append(
                Event(
                    start_beats=t,
                    dur_beats=0.5,
                    notes=[pitch_to_string_fret(int(pitch), True)],
                    velocity=int(normal_vel),
                )
            )

    events.sort(key=lambda e: e.start_beats)
    return events, chord_grid
def generate_pedalpoint_octave_phrase_8(
    *,
    bar_start: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
) -> Tuple[List[Event], List[str]]:
    """Pedal point phrase alternating pedal and its octave (note↔octave).

    The pedal root varies inside each bar (blocks of 1/2/3/4 beats). The pattern is
    either pedal→octave→pedal→octave... or octave→pedal→octave→pedal...
    Pedal strokes are palm-muted (velocity=45); octave strokes are normal velocity.
    """
    rng = random.Random(int(seed) ^ 0xC0FFEE99 ^ 0x0C7A0B1E)
    n_deg = len(ctx.degrees)

    pedal_candidates = [d for d in (0, 4, 5, 3, 2) if d < n_deg]
    if not pedal_candidates:
        pedal_candidates = [0]

    tonic_w = float(PEDALPOINT_TONIC_WEIGHT)

    def pick_pedal_deg(exclude: Optional[int] = None) -> int:
        choices = list(pedal_candidates)
        if exclude is not None and exclude in choices and len(choices) > 1:
            choices = [d for d in choices if d != exclude] or list(pedal_candidates)
        if len(choices) <= 1:
            return int(choices[0])
        rest = max(0.0, 1.0 - tonic_w)
        weights = []
        for d in choices:
            weights.append(tonic_w if d == 0 else rest / float(len(choices) - (1 if 0 in choices else 0)) if (0 in choices) else 1.0)
        if 0 not in choices:
            weights = [1.0 for _ in choices]
        return int(rng.choices(choices, weights=weights, k=1)[0])

    events: List[Event] = []
    chord_grid: List[str] = []

    start_with_octave = rng.random() < 0.50
    palm_vel = int(PALM_MUTE_VELOCITY)
    normal_vel = int(vel)
    lo, hi = int(PITCH_LO_MIDI), int(PITCH_HI_MIDI)

    for b in range(8):
        block_beats = int(rng.choice([1, 2, 3, 4]))
        pedal_deg_per_beat: List[int] = []
        beat = 0
        last_deg: Optional[int] = None
        while beat < 4:
            avoid = last_deg if (last_deg is not None and rng.random() < 0.70) else None
            deg = pick_pedal_deg(exclude=avoid)
            for _ in range(block_beats):
                if beat >= 4:
                    break
                pedal_deg_per_beat.append(int(deg))
                beat += 1
            last_deg = int(deg)

        if len(set(pedal_deg_per_beat)) == 1 and block_beats < 4 and len(pedal_candidates) > 1:
            pedal_deg_per_beat[-1] = pick_pedal_deg(exclude=pedal_deg_per_beat[-1])

        for beat_i in range(4):
            cd = ctx.degrees[int(pedal_deg_per_beat[beat_i])]
            chord_grid.append(f"{cd.roman}:{pc_to_note(cd.root_pc, True)}")

        bar0 = float((bar_start + b) * 4)
        for i in range(8):  # 8 eighth-notes
            t = bar0 + i * 0.5
            beat_i = int(i * 0.5)  # 0..3
            pedal_deg = int(pedal_deg_per_beat[beat_i])

            pedal_pitch = int(degree_root_pitch(ctx, pedal_deg, base_tonic_midi))
            up = pedal_pitch + 12
            dn = pedal_pitch - 12
            if up <= hi:
                octave_pitch = up
            elif dn >= lo:
                octave_pitch = dn
            else:
                octave_pitch = clampi(up, lo, hi)
            if int(octave_pitch) == int(pedal_pitch):
                octave_pitch = clampi((pedal_pitch + 12) if (pedal_pitch + 12 <= hi) else (pedal_pitch - 12), lo, hi)

            is_octave = ((i % 2) == 0 and start_with_octave) or ((i % 2) == 1 and not start_with_octave)
            pitch = int(octave_pitch) if is_octave else int(pedal_pitch)
            velocity = int(normal_vel) if is_octave else int(palm_vel)

            events.append(
                Event(
                    start_beats=t,
                    dur_beats=0.5,
                    notes=[pitch_to_string_fret(int(pitch), True)],
                    velocity=int(velocity),
                )
            )

    events.sort(key=lambda e: e.start_beats)
    return events, chord_grid

def _is_minorish(ctx: MelodyContext) -> bool:
    return bool(ctx.degrees) and str(ctx.degrees[0].triad_quality) == "min"


def _nearest_pitch_for_pc(target: int, pc: int, lo: int, hi: int) -> int:
    candidates = _pitches_in_range_for_pc(int(pc), int(lo), int(hi))
    if not candidates:
        return int(clampi(int(target), int(lo), int(hi)))
    return int(min(candidates, key=lambda p: abs(int(p) - int(target))))


def _tonality_hint(ctx: MelodyContext) -> str:
    return "minor" if _is_minorish(ctx) else "major"


def _degree_offsets_for_tonality(tonality: str) -> Tuple[int, ...]:
    if str(tonality) == "minor":
        return (0, 2, 3, 5, 7, 8, 10)  # natural minor degrees
    return (0, 2, 4, 5, 7, 9, 11)      # major degrees


def _roman_degree_index(numeral: str) -> int:
    n = str(numeral).upper()
    mapping = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}
    if n not in mapping:
        raise ValueError(f"Bad roman numeral: {numeral}")
    return int(mapping[n])


def _parse_roman_token(
    token: str,
    *,
    tonic_pc: int,
    tonality: str,
) -> Tuple[int, List[int], int, str]:
    """Return (root_pc, chord_pcs, chord_size, label)."""
    tok = str(token).strip()
    if not tok:
        return int(tonic_pc) % 12, [int(tonic_pc) % 12, (int(tonic_pc) + 7) % 12, (int(tonic_pc) + 0) % 12], 3, ""

    degree_offs = _degree_offsets_for_tonality(tonality)

    def degree_root_offset(numeral: str) -> int:
        acc2 = 0
        rest2 = str(numeral).strip()
        while rest2 and rest2[0] in ("b", "#"):
            acc2 += -1 if rest2[0] == "b" else 1
            rest2 = rest2[1:]
        idx = _roman_degree_index(rest2)
        return int(degree_offs[idx]) + int(acc2)

    def quality_intervals(is_minor: bool, dim: bool, halfdim: bool, aug: bool, seventh: str) -> List[int]:
        if dim:
            triad = [0, 3, 6]
        elif halfdim:
            triad = [0, 3, 6]
        elif aug:
            triad = [0, 4, 8]
        elif is_minor:
            triad = [0, 3, 7]
        else:
            triad = [0, 4, 7]

        if not seventh:
            return triad

        if seventh == "maj7":
            sev = 11
        elif halfdim:
            sev = 10
        elif dim and seventh == "dim7":
            sev = 9
        else:
            # default 7 => minor 7th (dominant/minor seventh)
            sev = 10
        return triad + [sev]

    # Tritone substitute dominant
    if tok.startswith("subV"):
        base = tok
        sec = None
        if "/" in tok:
            base, sec = tok.split("/", 1)
        has_maj7 = "maj7" in base
        has_7 = ("7" in base) or has_maj7
        target = sec if sec else "I"
        target_off = degree_root_offset(target)
        dom_root_off = target_off + 7
        sub_root_off = dom_root_off + 6
        root_pc = (int(tonic_pc) + int(sub_root_off)) % 12
        intervals = quality_intervals(False, False, False, False, "maj7" if has_maj7 else ("7" if has_7 else ""))
        pcs = [(root_pc + i) % 12 for i in intervals]
        return root_pc, pcs, len(intervals), f"{tok}:{pc_to_note(root_pc, True)}"

    # Secondary chords like V/V, V7/ii, vii°/V
    if "/" in tok:
        base, sec = tok.split("/", 1)
        target_off = degree_root_offset(sec)

        base_clean = base.replace("°", "o")
        dim = ("o" in base_clean) or ("dim" in base_clean)
        halfdim = ("ø" in base) or ("ø" in base_clean)
        aug = ("+" in base_clean)
        has_maj7 = "maj7" in base_clean
        has_7 = ("7" in base_clean) or has_maj7

        # Determine secondary root by function (dominant or leading-tone)
        if base_clean.lower().startswith("vii"):
            root_off = target_off - 1  # leading-tone to target
        else:
            root_off = target_off + 7  # dominant of target

        # Secondary dominants are major/dominant by default unless explicitly dim/halfdim/aug
        is_minor = base_clean and base_clean[0].islower()
        seventh = "maj7" if has_maj7 else ("7" if has_7 else "")
        if (not dim) and (not halfdim) and (not aug) and base_clean.upper().startswith("V"):
            is_minor = False
            seventh = "7" if has_7 else ""
        if dim and has_7:
            seventh = "dim7"

        root_pc = (int(tonic_pc) + int(root_off)) % 12
        intervals = quality_intervals(is_minor, dim, halfdim, aug, seventh)
        pcs = [(root_pc + i) % 12 for i in intervals]
        return root_pc, pcs, len(intervals), f"{tok}:{pc_to_note(root_pc, True)}"

    # Plain roman/borrrowed chords: optional accidentals then numeral then quality markers
    acc = 0
    rest = tok
    while rest and rest[0] in ("b", "#"):
        acc += -1 if rest[0] == "b" else 1
        rest = rest[1:]

    rest_clean = rest.replace("°", "o")
    halfdim = "ø" in rest_clean
    dim = ("o" in rest_clean) or ("dim" in rest_clean)
    aug = "+" in rest_clean

    has_maj7 = "maj7" in rest_clean
    has_7 = ("7" in rest_clean) or has_maj7

    # numeral is leading run of I/V chars
    m_num = re.match(r"(?i)^(vii|vi|iv|iii|ii|v|i)", rest_clean)
    if not m_num:
        raise ValueError(f"Bad roman token: {tok}")
    numeral = m_num.group(0)
    base_off = degree_root_offset(numeral)
    root_off = base_off + acc
    root_pc = (int(tonic_pc) + int(root_off)) % 12

    is_minor = numeral.islower()
    seventh = "maj7" if has_maj7 else ("7" if has_7 else "")
    if dim and has_7:
        seventh = "dim7"

    intervals = quality_intervals(is_minor, dim, halfdim, aug, seventh)
    pcs = [(root_pc + i) % 12 for i in intervals]
    return root_pc, pcs, len(intervals), f"{tok}:{pc_to_note(root_pc, True)}"


def _select_chordprogression_template(
    rng: random.Random,
    ctx: MelodyContext,
    forced: Optional[str],
) -> Tuple[str, ChordProgressionTemplate]:
    if forced:
        name = str(forced)
        if name not in CHORDPROGRESSION_POOL:
            raise ValueError(f"Unknown chordprogression: {name}")
        return name, CHORDPROGRESSION_POOL[name]

    minorish = _is_minorish(ctx)
    names = list(CHORDPROGRESSION_CHOICES)
    weights: List[float] = []
    for n in names:
        t = CHORDPROGRESSION_POOL[n]
        w = float(t.weight)
        tags = set(t.tags)
        if minorish:
            if "minor" in tags or "sad" in tags:
                w *= 2.0
            if "major" in tags and "sad" not in tags:
                w *= 0.75
        else:
            if "major" in tags:
                w *= 1.35
            if "minor" in tags:
                w *= 0.75
        if "chromatic" in tags:
            w *= 1.05
        weights.append(max(0.01, w))
    name = str(rng.choices(names, weights=weights, k=1)[0])
    return name, CHORDPROGRESSION_POOL[name]


def _next_multiple_of_4(n: int) -> int:
    n = int(n)
    return int(((n + 3) // 4) * 4)


def _build_cycle_tokens_and_durations(
    tokens: Sequence[str],
    *,
    repeats: int,
    min_bars: int,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    base = list(tokens) * int(repeats)
    durs = [1] * len(base)
    base_bars = sum(durs)
    target = max(int(min_bars), _next_multiple_of_4(base_bars))
    extra = int(target - base_bars)

    # Extend from the end (cadential hold), then sprinkle if needed.
    idx = len(durs) - 1
    while extra > 0 and idx >= 0:
        durs[idx] += 1
        extra -= 1
        idx -= 1
        if idx < 0:
            idx = len(durs) - 1
            if rng.random() < 0.50:
                break
    while extra > 0:
        j = rng.randrange(len(durs))
        durs[j] += 1
        extra -= 1

    return base, durs


def generate_chordprogression_segment_16(
    *,
    bar_start: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    chordprogression: Optional[str],
) -> Tuple[List[Event], List[str]]:
    """A 16-bar tonal progression segment, with strict triads/tetrads only.

    Unlike other sections, this does NOT force an 8-bar phrase that is repeated verbatim.
    It builds a full 16-bar harmonic plan so longer progressions (>=5 chords) can repeat cleanly.
    """
    rng = random.Random(int(seed) ^ 0xC0BD1234)
    name, tmpl = _select_chordprogression_template(rng, ctx, chordprogression if chordprogression else None)

    tonic_pc = int(ctx.tonic_pc) % 12
    tonality = _tonality_hint(ctx)

    tokens = list(tmpl.tokens)
    if len(tokens) > 8:
        tokens = tokens[:8]  # keep segments sane/playable

    # Build a "cycle" (progression repeats twice if longer than 4 chords).
    base_repeats = 2 if len(tokens) > 4 else 2
    cycle_tokens, cycle_durs = _build_cycle_tokens_and_durations(tokens, repeats=base_repeats, min_bars=8, rng=rng)
    cycle_bars = int(sum(cycle_durs))

    # Fill the 16-bar segment by repeating whole cycles if possible; otherwise extend last chord(s).
    seg_tokens: List[str] = []
    seg_durs: List[int] = []
    reps = max(1, int(SEGMENT_BARS // max(1, cycle_bars)))
    reps = min(reps, 2)  # keep variety; 8-bar cycles will repeat twice.
    for _ in range(reps):
        seg_tokens.extend(cycle_tokens)
        seg_durs.extend(cycle_durs)

    bars_now = int(sum(seg_durs))
    remain = int(SEGMENT_BARS - bars_now)
    if remain > 0 and seg_durs:
        # Extend cadence (end-hold) across remaining bars.
        k = len(seg_durs) - 1
        while remain > 0 and k >= 0:
            seg_durs[k] += 1
            remain -= 1
            k -= 1

    # Hard safety.
    total_bars = int(sum(seg_durs))
    if total_bars != int(SEGMENT_BARS):
        # Normalize to exactly 16 by trimming/adding to last chord.
        diff = int(SEGMENT_BARS - total_bars)
        if diff > 0:
            seg_durs[-1] += diff
        elif diff < 0:
            # trim from the end (never removes chords)
            take = -diff
            while take > 0 and seg_durs:
                dec = min(take, max(0, seg_durs[-1] - 1))
                seg_durs[-1] -= dec
                take -= dec
                if seg_durs[-1] <= 0:
                    seg_durs[-1] = 1
                    break

    events: List[Event] = []
    chord_grid: List[str] = []

    bar_i = 0
    for tok, dur_bars in zip(seg_tokens, seg_durs):
        root_pc, chord_pcs, chord_size, label = _parse_roman_token(tok, tonic_pc=tonic_pc, tonality=tonality)
        chord_pcs = chord_pcs[: chord_size]  # defensive

        # Ensure triads/tetrads only
        if chord_size < 3:
            chord_pcs = chord_pcs + [(root_pc + 7) % 12, (root_pc + 0) % 12]
            chord_pcs = chord_pcs[:3]
            chord_size = 3
        if chord_size > 4:
            chord_pcs = chord_pcs[:4]
            chord_size = 4

        voicing = choose_poly_fingering_strict(chord_pcs, prefer_root_bass=True)

        events.append(
            Event(
                start_beats=float((bar_start + bar_i) * 4),
                dur_beats=float(int(dur_bars) * 4),
                notes=voicing,
                velocity=int(vel),
            )
        )

        # chord labels per beat
        chord_grid.extend([label] * int(dur_bars) * 4)
        bar_i += int(dur_bars)
        if bar_i >= int(SEGMENT_BARS):
            break

    chord_grid = (chord_grid + [""] * (SEGMENT_BARS * 4))[: (SEGMENT_BARS * 4)]
    events.sort(key=lambda e: e.start_beats)
    return events, chord_grid

def _chordsprog_strip_target_roman(token: str) -> str:
    """Return a safe roman target for secondary notation.

    Examples:
        "V7/ii" -> "ii"
        "subV7/V" -> "V"
        "bVI" -> "bVI"
        "iv6" -> "iv"
    """
    t = str(token).strip()
    if "/" in t:
        return t.split("/", 1)[1].strip()

    m = re.match(r"^([b#]*)(N|[ivIV]+)", t)
    if m:
        acc = m.group(1) or ""
        core = m.group(2)
        return f"{acc}{core}"
    return t


def _chordsprog_pitch_for_pc(*, pc: int, prefer: int, lo: int, hi: int) -> int:
    """Pick a pitch in [lo, hi] with pitch-class pc, closest to prefer."""
    pc = int(pc) % 12
    prefer = int(max(lo, min(hi, prefer)))
    best = None
    best_dist = None
    for d in range(-5, 6):
        cand = int(prefer) + int(d) * 12
        if lo <= cand <= hi and (cand % 12) == pc:
            dist = abs(cand - prefer)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = cand
    if best is None:
        best = choose_pitch_for_pc_in_range(random.Random(pc * 99991 + prefer), pc, lo, lo, hi)
    return int(best)


def _chordsprog_make_prep_token(rng, next_token: str) -> str:
    """Choose a cadential 'prep' token that resolves into `next_token`."""
    tgt = _chordsprog_strip_target_roman(next_token)
    r = float(rng.random())
    if r < 0.45:
        return f"vii°7/{tgt}"
    if r < 0.75:
        return f"subV7/{tgt}"
    return f"V7/{tgt}"


def _chordsprog_is_majmin_triad(root_pc: int, chord_pcs) -> bool:
    """True only for plain major/minor triads (not dim/aug)."""
    root_pc = int(root_pc) % 12
    pcs = [int(p) % 12 for p in chord_pcs]
    has_m3 = ((root_pc + 3) % 12) in pcs
    has_M3 = ((root_pc + 4) % 12) in pcs
    has_5 = ((root_pc + 7) % 12) in pcs
    return bool(has_5 and (has_m3 ^ has_M3))


def generate_chordsprog_segment_16(
    *,
    bar_start: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    chordprogression: Optional[str],
) -> Tuple[List[Event], List[str]]:
    """A 16-bar tonal progression segment rendered as powerchords/dyads.

    No prep runs / mini-progressions: one chord per planned duration.
    - Major/minor triads -> powerchord (root+5), sometimes inverted (5th in bass).
    - Dim/aug 5th triads -> dyad (root + dim5/aug5).
    - Tetrads -> full tetrads (4 notes).
    """
    rng = random.Random(int(seed) ^ 0xB16B00B5)
    _, tmpl = _select_chordprogression_template(rng, ctx, chordprogression if chordprogression else None)

    tonic_pc = int(base_tonic_midi) % 12
    tonality = _tonality_hint(ctx)

    tokens = list(tmpl.tokens)
    if len(tokens) > 8:
        tokens = tokens[:8]

    cycle_tokens, cycle_durs = _build_cycle_tokens_and_durations(tokens, repeats=2, min_bars=8, rng=rng)

    seg_tokens: List[str] = []
    seg_durs: List[int] = []
    total = 0
    i = 0
    while total < int(SEGMENT_BARS):
        tok = str(cycle_tokens[i % len(cycle_tokens)])
        dur = int(cycle_durs[i % len(cycle_durs)])
        if total + dur > int(SEGMENT_BARS):
            dur = int(SEGMENT_BARS) - int(total)
        if dur <= 0:
            break
        seg_tokens.append(tok)
        seg_durs.append(dur)
        total += dur
        i += 1

    if int(total) < int(SEGMENT_BARS) and seg_durs:
        seg_durs[-1] += int(SEGMENT_BARS) - int(total)

    def fifth_interval_for_triad(root_pc: int, chord_pcs: List[int]) -> int:
        pcs = {int(p) % 12 for p in chord_pcs}
        root_pc = int(root_pc) % 12
        if (root_pc + 6) % 12 in pcs and (root_pc + 7) % 12 not in pcs:
            return 6
        if (root_pc + 8) % 12 in pcs and (root_pc + 7) % 12 not in pcs:
            return 8
        return 7

    def dyad_notes_for_interval(root_pc: int, interval: int, inverted: bool) -> List[Tuple[int, int]]:
        root_pc = int(root_pc) % 12
        interval = int(interval) % 12 or 7
        lo, hi = int(PITCH_LO_MIDI), int(PITCH_HI_MIDI)
        comp = 12 - interval

        root_pitch = _lowest_string6_pitch_for_pc(root_pc)

        if inverted:
            root_hi = int(root_pitch)
            bass = int(root_hi - comp)
            if bass < lo:
                root_hi += 12
                bass = int(root_hi - comp)
            if root_hi <= hi and bass >= lo:
                return choose_dyad_fingering(bass, root_hi)

        bass = int(root_pitch)
        up = bass + interval
        dn = bass - comp
        top = up if up <= hi else (dn if dn >= lo else up)
        return choose_dyad_fingering(bass, top)

    events: List[Event] = []
    chord_grid: List[str] = []

    bar_i = 0
    for tok, dur_bars in zip(seg_tokens, seg_durs):
        root_pc, chord_pcs, chord_size, label = _parse_roman_token(tok, tonic_pc=tonic_pc, tonality=tonality)
        chord_pcs = chord_pcs[: chord_size]

        if chord_size >= 4:
            voicing = choose_poly_fingering_strict(chord_pcs[:4], prefer_root_bass=True)
        else:
            interval = fifth_interval_for_triad(root_pc, chord_pcs)
            invert = bool(interval == 7 and rng.random() < 0.35)
            voicing = dyad_notes_for_interval(root_pc, interval, invert)

        events.append(
            Event(
                start_beats=float((bar_start + bar_i) * 4),
                dur_beats=float(int(dur_bars) * 4),
                notes=voicing,
                velocity=int(vel),
            )
        )
        chord_grid.extend([f"{label}:{pc_to_note(root_pc, True)}"] * (int(dur_bars) * 4))
        bar_i += int(dur_bars)

    chord_grid = chord_grid[: int(SEGMENT_BARS) * 4]
    return events, chord_grid
def generate_classical_phrase_8(
    *,
    bar_start: int,
    vel: int,
    seed: int,
) -> Tuple[List[Event], List[str]]:
    """Classical/neoclassical: stepwise runs, sequences, arpeggios. No 16ths; leaps <= octave."""
    rng = random.Random(int(seed) ^ 0xABCD1234)

    vel = FIXED_VELOCITY_45

    classical_modes = ["harmonic_minor", "melodic_minor", "hungarian_minor", "phrygian_dominant", "aeolian"]
    mode = rng.choice(classical_modes)
    tonic_pc = rng.randrange(12)
    ctx = build_melody_context(mode, tonic_pc)

    lo, hi = int(PITCH_LO_MIDI), int(CLASSICAL_HI_MIDI)

    def pick_pitch(pc: int, last: Optional[int]) -> int:
        pcs = _pitches_in_range_for_pc(pc, lo, hi)
        if not pcs:
            return lo
        if last is not None:
            pcs = [p for p in pcs if abs(int(p) - int(last)) <= 12] or pcs
        pcs.sort(key=lambda p: (p - lo) * 0.12 + (0 if last is None else abs(int(p) - int(last)) * 0.85))
        return int(pcs[0])

    def triplet_group(start: float, deg: int, direction: int, last: Optional[int]) -> Tuple[List[Event], int, int]:
        # 3 consecutive notes in 1 beat
        events: List[Event] = []
        for j in range(3):
            pc = int(ctx.scale_pcs[deg]) % 12
            p = pick_pitch(pc, last)
            last = p
            events.append(Event(start_beats=start + (j / 3.0), dur_beats=1.0 / 3.0, notes=[pitch_to_string_fret(int(p), prefer_low_strings=False)], velocity=int(vel)))
            deg = (deg + direction) % len(ctx.scale_pcs)
        return events, deg, int(last) if last is not None else lo

    events: List[Event] = []
    chord_grid: List[str] = []
    last_pitch: Optional[int] = None
    n = len(ctx.scale_pcs)

    motif_steps = rng.choice([[1, 1, -2, 1], [1, -1, 1, -1], [2, -1, -1, 2]])

    for bar in range(8):
        chord_grid.extend([f"CLA:{mode}:{pc_to_note(tonic_pc, True)}"] * 4)
        bar0 = float((bar_start + bar) * 4)
        kind = rng.choices(["run8", "arp8", "motif", "triplets"], weights=[0.52, 0.28, 0.17, 0.03], k=1)[0]

        if kind == "triplets":
            deg = rng.randrange(n)
            direction = rng.choice([1, -1])
            # 1 beat of triplets, then 8ths
            evs, deg, lastp = triplet_group(bar0, deg, direction, last_pitch)
            last_pitch = lastp
            events.extend(evs)
            t = 1.0
            for i in range(6):  # remaining 3 beats => 6 eighths
                pc = int(ctx.scale_pcs[deg]) % 12
                p = pick_pitch(pc, last_pitch)
                last_pitch = p
                events.append(Event(start_beats=bar0 + t + i * 0.5, dur_beats=0.5, notes=[pitch_to_string_fret(int(p), False)], velocity=int(vel)))
                deg = (deg + direction) % n

        elif kind == "run8":
            deg = rng.randrange(n)
            direction = rng.choice([1, -1])
            for i in range(8):
                pc = int(ctx.scale_pcs[deg]) % 12
                p = pick_pitch(pc, last_pitch)
                last_pitch = p
                events.append(Event(start_beats=bar0 + i * 0.5, dur_beats=0.5, notes=[pitch_to_string_fret(int(p), False)], velocity=int(vel)))
                if rng.random() < 0.85:
                    deg = (deg + direction) % n
                else:
                    deg = (deg + rng.choice([-1, 1])) % n

        elif kind == "motif":
            deg = rng.randrange(n)
            for i in range(8):
                deg = (deg + motif_steps[i % len(motif_steps)]) % n
                pc = int(ctx.scale_pcs[deg]) % 12
                p = pick_pitch(pc, last_pitch)
                last_pitch = p
                events.append(Event(start_beats=bar0 + i * 0.5, dur_beats=0.5, notes=[pitch_to_string_fret(int(p), False)], velocity=int(vel)))

        else:  # arp8
            deg = rng.randrange(n)
            pcs = [
                int(ctx.scale_pcs[deg]) % 12,
                int(ctx.scale_pcs[(deg + 2) % n]) % 12,
                int(ctx.scale_pcs[(deg + 4) % n]) % 12,
                int(ctx.scale_pcs[(deg + 2) % n]) % 12,
            ]
            pat = rng.choice([[0, 1, 2, 3, 2, 1, 0, 1], [0, 2, 1, 2, 0, 2, 1, 3]])
            for i in range(8):
                pc = pcs[pat[i] % len(pcs)]
                p = pick_pitch(pc, last_pitch)
                last_pitch = p
                events.append(Event(start_beats=bar0 + i * 0.5, dur_beats=0.5, notes=[pitch_to_string_fret(int(p), False)], velocity=int(vel)))

    events.sort(key=lambda e: e.start_beats)
    return events, chord_grid


# ----------------------------
# Phrase builders (8 bars motif logic)
# ----------------------------

def phrase_motif_8(
    *,
    kind: str,
    bar_start: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    chordprogression: Optional[str] = None,
    ctx: MelodyContext,
) -> Tuple[List[Event], List[str]]:
    rng = random.Random(int(seed) ^ 0xAA55AA55)

    if kind == "classical":
        return generate_classical_phrase_8(bar_start=bar_start, vel=vel, seed=seed)

    if kind == "chordprogression":
        return generate_chordprogression_segment_16(
            bar_start=bar_start,
            base_tonic_midi=base_tonic_midi,
            vel=vel,
            seed=seed,
            ctx=ctx,
            chordprogression=chordprogression,
        )

    if kind == "chordsprog":
        return generate_chordsprog_segment_16(
            bar_start=bar_start,
            base_tonic_midi=base_tonic_midi,
            vel=vel,
            seed=seed,
            ctx=ctx,
            chordprogression=chordprogression,
        )

    # per bar degree plans
    plans = [choose_degree_progression(rng, ctx, length=4) for _ in range(8)]

    def gen_bar(i: int, resolve: bool) -> Tuple[List[Event], List[str]]:
        b = bar_start + i
        if kind == "melodies":
            return generate_melodies_bar(
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 100 + i,
                ctx=ctx,
                degree_plan=plans[i],
                resolve=resolve,
            )
        if kind == "chords":
            plan = evil_progression_plan(rng, ctx)
            deg = plan[i]
            return generate_chords_bar(
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 200 + i,
                ctx=ctx,
                bar_degree=deg,
                resolve=resolve or (i in (3, 7)),
            )
        if kind == "downpicking":
            return generate_downpicking_bar(
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 300 + i,
                ctx=ctx,
                degree_plan=plans[i],
                resolve=resolve,
            )
        if kind == "gallops":
            return generate_rhythm_bar(
                section="gallops",
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 400 + i,
                ctx=ctx,
                degree_plan=plans[i],
                power_bias=0.34,
                resolve=resolve,
            )
        if kind == "gallopsopen":
            return generate_rhythm_bar(
                section="gallopsopen",
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 420 + i,
                ctx=ctx,
                degree_plan=plans[i],
                power_bias=0.34,
                resolve=resolve,
            )
        if kind == "bursts":
            return generate_rhythm_bar(
                section="bursts",
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 450 + i,
                ctx=ctx,
                degree_plan=plans[i],
                power_bias=0.34,
                resolve=resolve,
            )
        if kind == "justbursts":
            evs, ch = generate_rhythm_bar(
                section="bursts",
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 460 + i,
                ctx=ctx,
                degree_plan=plans[i],
                power_bias=0.0,
                resolve=resolve,
            )
            if i in (3, 7):
                evs = _apply_dim_interval_to_events(evs)
            return evs, ch

        if kind == "burstsopen":
            return generate_rhythm_bar(
                section="burstsopen",
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 470 + i,
                ctx=ctx,
                degree_plan=plans[i],
                power_bias=0.34,
                resolve=resolve,
            )
        if kind == "offbeatgallops":
            return generate_rhythm_bar(
                section="offbeatgallops",
                bar_index=b,
                base_tonic_midi=base_tonic_midi,
                vel=vel,
                seed=seed + 500 + i,
                ctx=ctx,
                degree_plan=plans[i],
                power_bias=0.38,
                resolve=resolve,
            )
        if kind in ("pedalpoint", "pedalpointoctave"):
            raise RuntimeError("pedalpoint handled at phrase level")
        raise ValueError(f"Unknown kind: {kind}")

    if kind == "pedalpoint":
        return generate_pedalpoint_phrase_8(bar_start=bar_start, base_tonic_midi=base_tonic_midi, vel=vel, seed=seed, ctx=ctx)

    if kind == "pedalpointoctave":
        return generate_pedalpoint_octave_phrase_8(bar_start=bar_start, base_tonic_midi=base_tonic_midi, vel=vel, seed=seed, ctx=ctx)

    # bars 1,2,4,8 then copy
    ev1, ch1 = gen_bar(0, False)
    ev2, ch2 = gen_bar(1, False)
    ev4, ch4 = gen_bar(3, False)
    ev8, ch8 = gen_bar(7, True)

    full: List[Event] = []
    chords_by_bar: Dict[int, List[str]] = {}

    for i, (evs, ch) in {0: (ev1, ch1), 1: (ev2, ch2), 3: (ev4, ch4), 7: (ev8, ch8)}.items():
        full.extend(evs)
        chords_by_bar[i] = ch

    # bar3 copy bar1, bar5 copy bar1, bar6 copy bar2, bar7 copy bar1
    for src, dst in [(0, 2), (0, 4), (1, 5), (0, 6)]:
        src_bar = bar_start + src
        dst_bar = bar_start + dst
        full.extend(shift_events([e for e in full if int(e.start_beats // 4) == src_bar], float((dst_bar - src_bar) * 4)))
        chords_by_bar[dst] = list(chords_by_bar[src])

    full.sort(key=lambda e: e.start_beats)
    chord_grid: List[str] = []
    for b in range(8):
        chord_grid.extend(chords_by_bar.get(b, [""] * 4))
    return full, chord_grid

def generate_song(
    *,
    total_bars: int,
    base_tonic_midi: int,
    vel: int,
    seed: int,
    ctx: MelodyContext,
    tonechange: int,
    chordprogression: Optional[str] = None,
    section: Optional[str] = None,
) -> Tuple[List[Event], List[str], List[str], Dict[str, MelodyContext], int]:
    """Generate the guitar song plus a beat-level chord guide.

    Form (16-bar segments; each segment = 8-bar phrase + verbatim repeat):
      intro - verse - bridge - chorus - intro - verse - bridge - chorus - instrumental - bridge - chorus - outro

    Repeated labels are verbatim because phrases are cached per label.
    """
    global LAST_TONECHANGES
    LAST_TONECHANGES = []

    total_bars = max(int(total_bars), SONG_MIN_TOTAL_BARS)
    seg_needed = int(math.ceil(total_bars / float(SEGMENT_BARS)))
    rng = random.Random(int(seed) ^ 0x5A17F00D)

    prefix = ["intro", "verse", "bridge", "chorus", "intro", "verse", "bridge", "chorus"]
    suffix = ["instrumental", "bridge", "chorus", "outro"]

    labels: List[str] = []
    labels.extend(prefix)
    while len(labels) + len(suffix) < seg_needed:
        labels.extend(["verse", "bridge", "chorus"])
    labels.extend(suffix)
    labels = labels[:seg_needed]

    chord_slot = rng.choice(["chorus", "bridge"]) if rng.random() < 0.50 else None
    # Decide kind per *label* (labels repeat later).
    label_kind: Dict[str, str] = {}

    def pick_kind() -> str:
        weights: List[float] = []
        for k in SECTION_KINDS_RANDOM:
            if k == "downpicking":
                weights.append(float(SECTION_DOWNPICK_WEIGHT))
            elif k == "melodies":
                weights.append(float(SECTION_MELODIES_WEIGHT))
            else:
                weights.append(1.0)
        return str(rng.choices(SECTION_KINDS_RANDOM, weights=weights, k=1)[0])

    for lbl in sorted(set(labels)):
        label_kind[lbl] = str(section) if section else pick_kind()

    global LAST_LABEL_KINDS
    LAST_LABEL_KINDS = dict(label_kind)

    candidates = sorted(set(labels))
    tc = clampi(int(tonechange), 0, 9999)
    tc = min(tc, len(candidates)) if candidates else 0
    changed = set(rng.sample(candidates, k=tc)) if tc > 0 else set()

    def alt_ctx_for(lbl: str) -> Tuple[MelodyContext, int]:
        alt_mode = random_dark_mode(rng)
        alt_tonic = rng.randrange(12)
        alt_ctx = build_melody_context(alt_mode, alt_tonic)
        cand = _pitches_in_range_for_pc(int(alt_tonic), int(PITCH_LO_MIDI), int(PITCH_HI_MIDI))
        tonic_midi = int(cand[0]) if cand else int(base_tonic_midi)
        LAST_TONECHANGES.append(
            f"{lbl}: {pc_to_note(ctx.tonic_pc, True)} {ctx.mode} -> {pc_to_note(alt_tonic, True)} {alt_mode}"
        )
        return alt_ctx, tonic_midi

    phrase_cache: Dict[str, PhraseData] = {}
    label_to_ctx: Dict[str, MelodyContext] = {}

    for lbl in sorted(set(labels)):
        kind = label_kind[lbl]
        use_ctx = ctx
        use_tonic = base_tonic_midi
        if lbl in changed:
            use_ctx, use_tonic = alt_ctx_for(lbl)

        phrase_seed = seed_for_label(seed, lbl)
        evs, chords = phrase_motif_8(
            kind=kind,
            bar_start=0,
            base_tonic_midi=use_tonic,
            vel=vel,
            seed=phrase_seed,
            chordprogression=chordprogression,
            ctx=use_ctx,
        )
        phrase_cache[lbl] = PhraseData(kind=kind, events=evs, chords=chords, ctx=use_ctx, tonic_midi=use_tonic)
        label_to_ctx[lbl] = use_ctx

    events: List[Event] = []
    bar_sections: List[str] = []
    beat_chords: List[str] = []

    bar = 0
    for lbl in labels:
        ph = phrase_cache[lbl]

        if ph.kind in ("chordprogression", "chordsprog"):
            # This section owns the whole 16-bar segment (no forced 8-bar repeat).
            events.extend(shift_events(ph.events, float(bar * 4)))
            bar_sections.extend([lbl] * SEGMENT_BARS)
            beat_chords.extend(list(ph.chords)[: SEGMENT_BARS * 4])
            bar += SEGMENT_BARS
            continue

        events.extend(shift_events(ph.events, float(bar * 4)))
        bar_sections.extend([lbl] * PHRASE_BARS)
        beat_chords.extend(list(ph.chords)[: PHRASE_BARS * 4])

        events.extend(shift_events(ph.events, float((bar + PHRASE_BARS) * 4)))
        bar_sections.extend([lbl] * PHRASE_BARS)
        beat_chords.extend(list(ph.chords)[: PHRASE_BARS * 4])

        bar += SEGMENT_BARS

    events.sort(key=lambda e: e.start_beats)
    events, bar_sections = trim_to_bars(events, bar_sections, total_bars)
    bars_out = int(len(bar_sections))
    beat_chords = beat_chords[: bars_out * 4]
    return events, bar_sections, beat_chords, label_to_ctx, bars_out
def _vocal_style_for_label(lbl: str) -> Optional[str]:
    if lbl == "verse":
        return "verse"
    if lbl == "bridge":
        return "bridge"
    if lbl == "chorus":
        return "chorus"
    return None
def _parse_root_pc_from_chord_label(label: str) -> Optional[int]:
    if not label:
        return None
    note = str(label).split(":")[-1].strip()
    return int(NOTE_TO_PC[note]) if note in NOTE_TO_PC else None


def choose_pitch_for_pc_in_range(
    rng: random.Random,
    pc: int,
    last_pitch: Optional[int],
    lo: int,
    hi: int,
) -> int:
    candidates = _pitches_in_range_for_pc(int(pc), int(lo), int(hi))
    if not candidates:
        return int(clampi(int(pc), int(lo), int(hi)))

    def cost(p: int) -> float:
        low_bias = (int(p) - int(lo)) * 0.85  # always prefer low
        if last_pitch is None:
            return low_bias
        return low_bias + abs(int(p) - int(last_pitch)) * 1.05

    candidates.sort(key=cost)
    top = candidates[: min(7, len(candidates))]
    weights = [1.0 / (1.0 + i) for i in range(len(top))]
    return int(rng.choices(top, weights=weights, k=1)[0])
def build_vocals_rhythm_slots(rng: random.Random, style: str) -> List[Tuple[float, float]]:
    """Return (start, dur) slots for a 4-beat bar. Grid = 0.5 beats."""
    style = str(style)
    if style == "verse":
        w = {0.5: 0.58, 1.0: 0.23, 2.0: 0.12, 3.0: 0.03, 4.0: 0.04}
    elif style == "bridge":
        w = {0.5: 0.22, 1.0: 0.50, 2.0: 0.16, 3.0: 0.05, 4.0: 0.07}
    else:  # chorus
        w = {0.5: 0.10, 1.0: 0.16, 2.0: 0.22, 3.0: 0.04, 4.0: 0.48}

    slots: List[Tuple[float, float]] = []
    t = 0.0
    while t < 4.0 - 1e-9:
        rem = 4.0 - t
        choices = [c for c in (4.0, 3.0, 2.0, 1.0, 0.5) if c <= rem + 1e-9]
        if not choices:
            break
        # Whole-note only makes musical sense at bar start.
        if t > 1e-9 and 4.0 in choices:
            choices.remove(4.0)
        weights = [w[float(c)] for c in choices]
        dur = float(rng.choices(choices, weights=weights, k=1)[0])
        dur = round(dur * 2.0) / 2.0
        slots.append((t, dur))
        t = round((t + dur) * 2.0) / 2.0
    return slots


def _choose_vocal_range_for_label(seed: int, lbl: str) -> Tuple[int, int]:
    """Pick a 1-octave (12 semitone) range per label; biased toward low notes."""
    rng = random.Random(seed_for_label(seed, f"vocal_range|{lbl}") ^ 0xFACEB00C)

    span = 12
    lo_min = int(VOCAL_LO_MIDI)
    lo_max = int(VOCAL_HI_MIDI) - span
    lo_max = max(lo_min, lo_max)

    candidates = list(range(lo_min, lo_max + 1))
    weights = [1.0 / (1.0 + (p - lo_min)) for p in candidates]  # heavier weight for lower lo
    lo = int(rng.choices(candidates, weights=weights, k=1)[0])
    return lo, int(lo + span)


def _pad_chords(chords: List[str], n: int) -> List[str]:
    if len(chords) >= n:
        return list(chords[:n])
    return list(chords) + [""] * (n - len(chords))


def _generate_vocal_phrase_8(
    *,
    lbl: str,
    beat_chords_phrase: List[str],  # 8 bars * 4 beats = 32
    ctx: MelodyContext,
    seed: int,
    velocity: int,
) -> List[VocalEvent]:
    style = _vocal_style_for_label(lbl)
    if style is None:
        return []

    beat_chords_phrase = _pad_chords(list(beat_chords_phrase), 32)
    velocity = clampi(int(velocity), 1, 127)

    lo, hi = _choose_vocal_range_for_label(seed, lbl)

    out: List[VocalEvent] = []
    last_pitch: Optional[int] = None

    for bar_in_phrase in range(8):
        rng = random.Random(seed_for_label(seed, f"vocals_phrase|{lbl}|{bar_in_phrase}") ^ 0xC0DE123)
        slots = build_vocals_rhythm_slots(rng, style)

        for s, d in slots:
            if rng.random() < (0.12 if style != "chorus" else 0.08):
                continue

            beat_idx = min(3, max(0, int(math.floor(float(s) + 1e-9))))
            chord_label = beat_chords_phrase[bar_in_phrase * 4 + beat_idx]
            root_pc = _parse_root_pc_from_chord_label(chord_label)
            if root_pc is None:
                root_pc = int(ctx.tonic_pc) % 12

            scale = [int(p) % 12 for p in ctx.scale_pcs]
            if root_pc in scale:
                root_deg = scale.index(int(root_pc))
            else:
                root_deg = min(range(len(scale)), key=lambda i: abs(int(scale[i]) - int(root_pc)))

            interval_deg = 3 if rng.random() < 0.52 else 4
            target_pc = int(scale[(root_deg + interval_deg) % len(scale)]) % 12

            if last_pitch is not None and rng.random() < 0.18:
                pitch = int(last_pitch)
            else:
                pitch = choose_pitch_for_pc_in_range(rng, target_pc, last_pitch, lo, hi)

            if last_pitch is not None and pitch == last_pitch and rng.random() > 0.35:
                # bias nudges downward to keep the register low and tight
                pitch = choose_pitch_for_pc_in_range(rng, target_pc, int(last_pitch) - 12, lo, hi)

            last_pitch = int(pitch)
            out.append(
                VocalEvent(
                    start_beats=float(bar_in_phrase * 4) + float(s),
                    dur_beats=float(d),
                    pitch=int(pitch),
                    velocity=int(velocity),
                )
            )

    out.sort(key=lambda e: e.start_beats)
    return out


def _choose_vocal_range_for_song(seed: int) -> Tuple[int, int]:
    """Pick a 1-octave (12 semitone) range for the whole vocal track; biased toward low notes."""
    rng = random.Random(seed_for_label(seed, "vocal_range|song") ^ 0xFACEB00C)

    span = 12
    lo_min = int(VOCAL_LO_MIDI)
    lo_max = int(VOCAL_HI_MIDI) - span
    lo_max = max(lo_min, lo_max)

    candidates = list(range(lo_min, lo_max + 1))
    weights = [1.0 / (1.0 + (p - lo_min)) for p in candidates]
    lo = int(rng.choices(candidates, weights=weights, k=1)[0])
    return lo, int(lo + span)


def _pad_chords(chords: List[str], n: int) -> List[str]:
    if len(chords) >= n:
        return list(chords[:n])
    return list(chords) + [""] * (n - len(chords))


def _generate_vocal_phrase_8(
    *,
    lbl: str,
    beat_chords_phrase: List[str],  # 8 bars * 4 beats = 32
    ctx: MelodyContext,
    seed: int,
    velocity: int,
    lo: int,
    hi: int,
) -> List[VocalEvent]:
    style = _vocal_style_for_label(lbl)
    if style is None:
        return []

    beat_chords_phrase = _pad_chords(list(beat_chords_phrase), 32)
    velocity = clampi(int(velocity), 1, 127)

    out: List[VocalEvent] = []
    last_pitch: Optional[int] = None

    for bar_in_phrase in range(8):
        rng = random.Random(seed_for_label(seed, f"vocals_phrase|{lbl}|{bar_in_phrase}") ^ 0xC0DE123)
        slots = build_vocals_rhythm_slots(rng, style)

        for s, d in slots:
            if rng.random() < (0.12 if style != "chorus" else 0.08):
                continue

            beat_idx = min(3, max(0, int(math.floor(float(s) + 1e-9))))
            chord_label = beat_chords_phrase[bar_in_phrase * 4 + beat_idx]
            root_pc = _parse_root_pc_from_chord_label(chord_label)
            if root_pc is None:
                root_pc = int(ctx.tonic_pc) % 12

            scale = [int(p) % 12 for p in ctx.scale_pcs]
            if root_pc in scale:
                root_deg = scale.index(int(root_pc))
            else:
                root_deg = min(range(len(scale)), key=lambda i: abs(int(scale[i]) - int(root_pc)))

            interval_deg = 3 if rng.random() < 0.52 else 4
            target_pc = int(scale[(root_deg + interval_deg) % len(scale)]) % 12

            if last_pitch is not None and rng.random() < 0.18:
                pitch = int(last_pitch)
            else:
                pitch = choose_pitch_for_pc_in_range(rng, target_pc, last_pitch, lo, hi)

            if last_pitch is not None and pitch == last_pitch and rng.random() > 0.35:
                pitch = choose_pitch_for_pc_in_range(rng, target_pc, int(last_pitch) - 12, lo, hi)

            last_pitch = int(pitch)
            out.append(
                VocalEvent(
                    start_beats=float(bar_in_phrase * 4) + float(s),
                    dur_beats=float(d),
                    pitch=int(pitch),
                    velocity=int(velocity),
                )
            )

    out.sort(key=lambda e: e.start_beats)
    return out


def generate_vocals_for_song(
    *,
    bar_sections: List[str],
    beat_chords: List[str],
    label_to_ctx: Dict[str, MelodyContext],
    seed: int,
    velocity: int = VOCAL_DEFAULT_VELOCITY,
) -> List[VocalEvent]:
    """Generate vocals that repeat exactly for repeated verse/bridge/chorus labels."""
    velocity = clampi(int(velocity), 1, 127)
    lo, hi = _choose_vocal_range_for_song(seed)

    phrase_cache: Dict[str, List[VocalEvent]] = {}
    out: List[VocalEvent] = []

    bar_i = 0
    bars_total = len(bar_sections)

    while bar_i < bars_total:
        lbl = bar_sections[bar_i]
        run = 1
        while (bar_i + run) < bars_total and bar_sections[bar_i + run] == lbl:
            run += 1

        style = _vocal_style_for_label(lbl)
        if style is None:
            bar_i += run
            continue

        ctx = label_to_ctx.get(lbl)
        if ctx is None:
            bar_i += run
            continue

        if lbl not in phrase_cache:
            phrase_chords = beat_chords[bar_i * 4 : (bar_i + 8) * 4]
            phrase_cache[lbl] = _generate_vocal_phrase_8(
                lbl=lbl,
                beat_chords_phrase=phrase_chords,
                ctx=ctx,
                seed=seed,
                velocity=velocity,
                lo=lo,
                hi=hi,
            )

        phrase = phrase_cache[lbl]

        seg_start_beats = float(bar_i * 4)
        seg_end_beats = float((bar_i + run) * 4)

        for ev in phrase:
            t = seg_start_beats + float(ev.start_beats)
            if t < seg_end_beats - 1e-9:
                out.append(VocalEvent(start_beats=t, dur_beats=ev.dur_beats, pitch=ev.pitch, velocity=ev.velocity))

        if run >= 16:
            rep_shift = float(8 * 4)
            for ev in phrase:
                t = seg_start_beats + rep_shift + float(ev.start_beats)
                if t < seg_end_beats - 1e-9:
                    out.append(VocalEvent(start_beats=t, dur_beats=ev.dur_beats, pitch=ev.pitch, velocity=ev.velocity))

        bar_i += run

    out.sort(key=lambda e: e.start_beats)
    return out
def events_to_tab(events: List[Event], bars: int, bar_sections: List[str]) -> str:
    sub = 16  # characters per bar
    total = int(bars) * sub
    grid: Dict[int, List[List[str]]] = {s: [list("-" * total)] for s in range(1, 7)}

    def put(string: int, beat: float, text: str) -> None:
        col = int(round(beat * 4))  # 16th columns (display only)
        col = clampi(col, 0, total - 1)
        row = grid[string][0]
        for i, ch in enumerate(text):
            if col + i < total:
                row[col + i] = ch

    for ev in events:
        bar_i = int(ev.start_beats // 4)
        if bar_i >= bars:
            continue
        col_beat = float(ev.start_beats)
        for (s, f) in ev.notes:
            put(s, col_beat, str(int(f)))

    # header: sections per bar
    section_line = []
    for b in range(bars):
        section_line.append(f"{bar_sections[b]:<8}"[:8])
    out_lines = ["SECTIONS: " + " | ".join(section_line), ""]
    for s in range(1, 7):
        name = STRING_NAMES[s]
        row = "".join(grid[s][0])
        chunks = [row[i * sub : (i + 1) * sub] for i in range(bars)]
        out_lines.append(f"{name}| " + " | ".join(chunks))
    return "\n".join(out_lines)



def _vlq(n: int) -> bytes:
    """Variable-length quantity encoding for MIDI."""
    n = int(n)
    if n < 0:
        n = 0
    out = bytearray()
    out.append(n & 0x7F)
    n >>= 7
    while n:
        out.insert(0, 0x80 | (n & 0x7F))
        n >>= 7
    return bytes(out)


def _u16(n: int) -> bytes:
    return int(n).to_bytes(2, byteorder="big", signed=False)


def _u32(n: int) -> bytes:
    return int(n).to_bytes(4, byteorder="big", signed=False)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return bytes(tag) + _u32(len(payload)) + bytes(payload)


def _write_midi_fallback(
    *,
    bpm: int,
    notes: List[Tuple[int, int, int, int]],  # (pitch, start_tick, end_tick, velocity)
    out_mid: str,
    program: int,
) -> None:
    """Write a simple format-0 MIDI with tempo + program change + notes."""
    ppq = 480
    tempo_us = int(round(60_000_000 / float(clampi(int(bpm), 20, 400))))

    msgs: List[Tuple[int, int, bytes]] = []

    # tempo meta
    tempo = tempo_us.to_bytes(3, byteorder="big", signed=False)
    msgs.append((0, 0, bytes([0xFF, 0x51, 0x03]) + tempo))

    # program change (channel 0)
    msgs.append((0, 1, bytes([0xC0, clampi(int(program), 0, 127)])))

    for pitch, st, en, vel in notes:
        pitch = clampi(int(pitch), 0, 127)
        st = max(0, int(st))
        en = max(st + 1, int(en))
        vel = clampi(int(vel), 0, 127)
        # note off first at same tick
        msgs.append((en, 0, bytes([0x80, pitch, 0])))
        msgs.append((st, 1, bytes([0x90, pitch, vel])))

    msgs.sort(key=lambda x: (x[0], x[1]))

    track = bytearray()
    last_tick = 0
    for tick, _order, payload in msgs:
        delta = int(tick) - int(last_tick)
        last_tick = int(tick)
        track += _vlq(delta)
        track += payload

    # end of track
    track += _vlq(0) + bytes([0xFF, 0x2F, 0x00])

    header = _chunk(b"MThd", _u16(0) + _u16(1) + _u16(ppq))
    body = _chunk(b"MTrk", bytes(track))
    with open(out_mid, "wb") as f:
        f.write(header)
        f.write(body)




def write_midi(*, bpm: int, events: List[Event], out_mid: str) -> None:
    """Write the guitar MIDI. Uses pretty_midi if available; otherwise a tiny fallback writer."""
    if pretty_midi is None:
        ppq = 480
        notes: List[Tuple[int, int, int, int]] = []
        for ev in events:
            st = int(round(float(ev.start_beats) * ppq))
            en = int(round(float(ev.start_beats + ev.dur_beats) * ppq))
            for (s, f) in ev.notes:
                pitch = string_fret_to_pitch(s, f)
                notes.append((int(pitch), st, en, int(ev.velocity)))
        _write_midi_fallback(bpm=bpm, notes=notes, out_mid=out_mid, program=30)
        return

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=30)

    for ev in events:
        start = beats_to_seconds(bpm, ev.start_beats)
        end = beats_to_seconds(bpm, ev.start_beats + ev.dur_beats)
        for (s, f) in ev.notes:
            pitch = string_fret_to_pitch(s, f)
            inst.notes.append(
                pretty_midi.Note(
                    velocity=int(ev.velocity),
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                )
            )

    pm.instruments.append(inst)
    pm.write(out_mid)


def write_vocals_midi(*, bpm: int, events: List[VocalEvent], out_mid: str) -> None:
    """Write the vocals MIDI. Uses pretty_midi if available; otherwise a tiny fallback writer."""
    if pretty_midi is None:
        ppq = 480
        notes: List[Tuple[int, int, int, int]] = []
        for ev in events:
            st = int(round(float(ev.start_beats) * ppq))
            en = int(round(float(ev.start_beats + ev.dur_beats) * ppq))
            notes.append((int(ev.pitch), st, en, int(ev.velocity)))
        _write_midi_fallback(bpm=bpm, notes=notes, out_mid=out_mid, program=52)
        return

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=52, name="Vocals")

    for ev in events:
        start = beats_to_seconds(bpm, ev.start_beats)
        end = beats_to_seconds(bpm, ev.start_beats + ev.dur_beats)
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(ev.velocity),
                pitch=int(ev.pitch),
                start=float(start),
                end=float(end),
            )
        )

    pm.instruments.append(inst)
    pm.write(out_mid)



# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bpm", type=int, default=140)
    p.add_argument(
        "--bars",
        type=int,
        default=SONG_DEFAULT_TOTAL_BARS,
        help="Total bars (song mode). Minimum 192 enforced; if not divisible by 16, tail is trimmed.",
    )
    p.add_argument("--seed", type=int, default=None, help="Omit for random each run; set for reproducible output.")
    p.add_argument("--out", type=str, default="riff")
    p.add_argument("--root_fret", type=int, default=0, help="0 = low open D base tonic")
    p.add_argument("--velocity", type=int, default=80)

    p.add_argument("--tone", type=str, default=None, help="Tonic note (e.g. D, Eb, F#). Random if omitted.")
    p.add_argument("--melody_mode", type=str, default=None, choices=sorted(MELODY_SCALES.keys()), help="Mode/scale. Random dark if omitted.")
    p.add_argument("--tonechange", type=int, default=0, help="How many structure labels regenerate in a different key+mode (repeats remain verbatim).")


    p.add_argument(
        "--chordprogression",
        nargs="?",
        const="random",
        default=None,
        choices=("random",) + CHORDPROGRESSION_CHOICES,
        help=(
            "Force section=chordsprog. Optionally pick a named classical progression "
            "template from the pool; omit the value for random."
        ),
    )

    p.add_argument(
        "--section",
        type=str,
        default=None,
        choices=SECTION_KINDS,
        help="Force the entire song to use a single section kind (song structure + repeats remain).",
    )

    p.add_argument("--no_midi", action="store_true")
    return p.parse_args()



def main() -> None:
    a = parse_args()
    bpm = clampi(int(a.bpm), 40, 260)

    seed = int(a.seed) if a.seed is not None else secrets.randbits(31)
    rng = random.Random(seed ^ 0x13579BDF)

    tonic_pc = parse_tone_to_pc(a.tone) if a.tone else rng.randrange(12)
    mode = str(a.melody_mode) if a.melody_mode else random_dark_mode(rng)
    if (not a.melody_mode) and mode in HAPPY_RANDOM_EXCLUDE:
        mode = random_dark_mode(rng)

    ctx = build_melody_context(mode, tonic_pc)

    bars = int(a.bars)
    bars = max(bars, SONG_MIN_TOTAL_BARS)
    bars = (bars // SEGMENT_BARS) * SEGMENT_BARS  # trim to 16s

    base_tonic_midi = int(OPEN[6] + int(a.root_fret))
    vel = clampi(int(a.velocity), 1, 127)

    forced_section = str(a.section) if a.section else None
    if forced_section == "chordprogression":
        forced_section = "chordsprog"
    chordprog = None
    if a.chordprogression is not None:
        forced_section = "chordsprog"
        chordprog = None if str(a.chordprogression) == "random" else str(a.chordprogression)

    events, bar_sections, beat_chords, label_to_ctx, bars_out = generate_song(
        total_bars=bars,
        base_tonic_midi=base_tonic_midi,
        vel=vel,
        seed=seed,
        ctx=ctx,
        tonechange=int(a.tonechange),
        chordprogression=chordprog,
        section=forced_section,
    )

    out_mid = f"{a.out}.mid"
    out_voc_mid = f"{a.out}_vocals.mid"
    out_txt = f"{a.out}.txt"

    title = f"{a.out} | mode={ctx.mode} tone={pc_to_note(ctx.tonic_pc, True)} bars={bars_out} seed={seed}"
    tone_line = f"Tone changes: {len(LAST_TONECHANGES)}"
    tone_details = "\n".join(LAST_TONECHANGES)

    if not a.no_midi:
        write_midi(bpm=bpm, events=events, out_mid=out_mid)
        vocal_events = generate_vocals_for_song(
            bar_sections=bar_sections,
            beat_chords=beat_chords,
            label_to_ctx=label_to_ctx,
            seed=seed,
            velocity=VOCAL_DEFAULT_VELOCITY,
        )
        write_vocals_midi(bpm=bpm, events=vocal_events, out_mid=out_voc_mid)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write(tone_line + "\n")
        if tone_details:
            f.write(tone_details + "\n")
        f.write("\n")
        if LAST_LABEL_KINDS:
            f.write("LABEL→KIND: " + ", ".join(f"{k}={v}" for k, v in sorted(LAST_LABEL_KINDS.items())) + "\n\n")
        f.write(events_to_tab(events, bars_out, bar_sections))
        f.write("\n")

    print(title)
    print(tone_line)
    if tone_details:
        print(tone_details)
    if LAST_LABEL_KINDS:
        print("LABEL→KIND: " + ", ".join(f"{k}={v}" for k, v in sorted(LAST_LABEL_KINDS.items())))
    print(f"Wrote: {out_txt}")
    if not a.no_midi:
        print(f"Wrote: {out_mid}")
        print(f"Wrote: {out_voc_mid}")


if __name__ == "__main__":
    main()

