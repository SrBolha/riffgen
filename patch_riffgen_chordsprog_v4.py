# patch_riffgen_chordsprog_v3.py
"""
Patches riffgen.py so chordsprog becomes a chordprogression-like section
(no prep runs), rendered as powerchords/inversions for maj/min triads,
dyads for dim/aug 5th triads, and full tetrads for 7ths.

Usage:
  python patch_riffgen_chordsprog_v3.py riffgen.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def patch_section_kinds(text: str) -> str:
    pat = r"^SECTION_KINDS:\s*List\[str\]\s*=\s*\[\s*$"
    m = re.search(pat, text, flags=re.MULTILINE)
    if not m:
        die("Could not find SECTION_KINDS List[str] definition to patch.")

    after = text[m.end() :]
    close = re.search(r"^\]\s*$", after, flags=re.MULTILINE)
    if not close:
        die("Could not find closing ']' for SECTION_KINDS.")

    start = m.start()
    end = m.end() + close.end()

    replacement = (
        'SECTION_KINDS: List[str] = [\n'
        '    "gallops",\n'
        '    "gallopsopen",\n'
        '    "bursts",\n'
        '    "justbursts",\n'
        '    "burstsopen",\n'
        '    "offbeatgallops",\n'
        '    "chords",\n'
        '    "chordsprog",\n'
        '    "chordprogression",\n'
        '    "pedalpoint",\n'
        '    "pedalpointoctave",\n'
        '    "classical",\n'
        '    "downpicking",\n'
        '    "melodies",\n'
        ']\n'
        "\n"
        '# Random selection pool: chordprogression is superseded by chordsprog.\n'
        'SECTION_KINDS_RANDOM: Tuple[str, ...] = tuple(k for k in SECTION_KINDS if k != "chordprogression")\n'
    )
    return text[:start] + replacement + text[end:]


def patch_pick_kind(text: str) -> str:
    text2 = re.sub(
        r"for k in SECTION_KINDS:\s*",
        "for k in SECTION_KINDS_RANDOM:\n",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if text2 == text:
        die("Could not patch pick_kind loop (for k in SECTION_KINDS).")

    text3 = re.sub(
        r"rng\.choices\(\s*SECTION_KINDS\s*,\s*weights=weights\s*,\s*k=1\s*\)\[0\]",
        "rng.choices(SECTION_KINDS_RANDOM, weights=weights, k=1)[0]",
        text2,
        count=1,
        flags=re.MULTILINE,
    )
    if text3 == text2:
        die("Could not patch pick_kind selection (rng.choices(SECTION_KINDS,...)).")

    return text3


def patch_chordprogression_flag(text: str) -> str:
    # Force chordsprog in main() when --chordprogression is used.
    text2 = re.sub(
        r'forced_section\s*=\s*"chordprogression"\s*',
        'forced_section = "chordsprog"\n',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if text2 == text:
        die('Could not patch main() forced_section = "chordprogression".')
    return text2


def patch_generate_chordsprog_segment_16(text: str) -> str:
    start_pat = r"^def generate_chordsprog_segment_16\(\s*$"
    m_start = re.search(start_pat, text, flags=re.MULTILINE)
    if not m_start:
        die("Could not find def generate_chordsprog_segment_16 to patch.")

    # next top-level def
    end_pat = r"^def [a-zA-Z_][a-zA-Z0-9_]*\(\s*$"
    m_end = re.search(end_pat, text[m_start.end() :], flags=re.MULTILINE)
    if not m_end:
        die("Could not find the next def after generate_chordsprog_segment_16.")

    start = m_start.start()
    end = m_start.end() + m_end.start()

    replacement = '''def generate_chordsprog_segment_16(
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
'''
    return text[:start] + replacement + text[end:]


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("riffgen.py")
    if not target.exists():
        die(f"File not found: {target}")

    text = target.read_text(encoding="utf-8")
    text = patch_section_kinds(text)
    text = patch_pick_kind(text)
    text = patch_chordprogression_flag(text)
    text = patch_generate_chordsprog_segment_16(text)

    target.write_text(text, encoding="utf-8")
    print(f"Patched {target} successfully.")


if __name__ == "__main__":
    main()
