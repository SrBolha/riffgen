# riffgen — Chordprogression & chordsprog work log

This README summarizes **everything changed / attempted during this chat** to improve the **chord progression sections** in `riffgen.py`.

> Files referenced in this workspace:
> - `riffgen.py` (the generator)
> - `riff.txt` (a generated output example)
> - `patch_riffgen_chordsprog.py` (early patch script)
> - `patch_riffgen_chordsprog_v2.py` (second patch script)

---

## 1) Original problem statement

You reported that the **`chordprogression`** section was sometimes emitting **single notes**, and you needed:

- **Only triads or tetrads** (3–4 notes), never 1-note output.
- The progression logic should be **tonal** and **ignore `--melody_mode`** (so it can use secondary dominants, tritone subs, minor dominants, etc.).
- Add a **large pool** of **classical + sad** progressions (vast variety).
- The chord progression section should **not be constrained** by the same 8-bar repeat mechanics as other sections:
  - It can be **longer** (e.g., repeat the progression twice if it’s > 4 chords).
  - If progression chord-count is odd (e.g., `I V I` / `ii V I`), extend some chords so the **total bars** is **divisible by 4**.

---

## 2) Changes introduced / discussed

### 2.1 “Roman-token” progression pool (tonal, not melody_mode)
We moved away from a purely diatonic / degree-index pool and toward **roman-token templates** that can express:

- `V7/ii`, `vii°7/V` (secondary functions)
- `subV7`, `subV7/ii` (tritone substitutions)
- borrowed chords like `bIImaj7`, `bVI`, `bVII`, etc.
- minor dominants (`v`, `v7`)
- diminished/half-diminished (`vii°7`, `iiø7`)

In the current `riffgen.py`, you can see a big roman-token pool under `CHORDPROGRESSION_POOL` (around the early part of the file).

### 2.2 Auto-expansion + weighting (Option A)
You asked for “Option A”: automatically generate **many more** progressions (cadential variants, inserted applied dominants, tritone subs, etc.) and weight selection toward **sad/classical** choices.

Multiple patch iterations attempted to do this robustly across different `riffgen.py` versions.

### 2.3 Fixing “single notes” in chordprogression
The requirement: chordprogression should **never collapse to 1 note**.

The approach used:
- enforce strict 3–4 note selection for each chord (triads/tetrads only),
- prefer “playable” guitar voicings,
- fallback logic must still return 3–4 notes.

---

## 3) Introducing `chordsprog`

### 3.1 First design (rhythmic “prep” inside bar)
A new section kind **`chordsprog`** was introduced, based on the same progression pool as chordprogression, but able to play **half notes** and **quarter notes** as small “prep” motions toward the next chord.

You later rejected this behavior because it sounded **too jazzy**.

### 3.2 Revised design request (final direction)
You asked for a simpler approach:

- **No more “little progressions” / prep runs**.
- Make `chordsprog` behave **just like chordprogression** (harmonic rhythm, chord durations) but render:
  - **powerchords** (root + perfect 5th) and **inverted powerchords**
  - if chord has **diminished 5th** (tritone) or **augmented 5th**:
    - generate **2-note dyad**: root + (dim5/aug5)
  - (previously you allowed tetrads for 7ths, but final request moved toward dyads-only; this was actively iterated)

- Additionally: when the song structure would use **`chordprogression`**, it should use **`chordsprog` instead**.

This last “replacement behavior” was proposed as a patch, but whether it’s active depends on which patch you applied locally (see troubleshooting below).

---

## 4) Patch scripts & why some failed on Windows

You ran patch scripts on Windows and hit errors. These were caused by **version mismatches** between the patcher’s expectations and your local `riffgen.py`.

### 4.1 `patch_riffgen_chordsprog.py`
Error you saw:
- `Could not find def generate_chordprogression_phrase_8`

Reason:
- Your `riffgen.py` had already moved to `generate_chordprogression_segment_16(...)` (no `...phrase_8`).

### 4.2 `patch_riffgen_chordsprog_v2.py`
Error you saw:
- `Could not find SECTION_KINDS tuple to patch.`

Reason:
- Your file uses `SECTION_KINDS: List[str] = [...]` (a list), not a tuple.

### 4.3 Next step scripts (v3 / poweronly / simplified chordsprog)
Later patches were adjusted to:
- handle `SECTION_KINDS` being a list or tuple,
- patch `generate_chordsprog_segment_16` directly,
- optionally rewrite selection so chordprogression is replaced by chordsprog in random structures.

Depending on what you ran locally, you may still need the “final simplification” patch that removes the jazzy prep behavior.

---

## 5) How to run (typical)

Generate only the chordsprog section:

```bash
python riffgen.py --section chordsprog
```

Generate a random song structure:

```bash
python riffgen.py
```

If your build supports forcing a progression name:

```bash
python riffgen.py --section chordsprog --chordprogression <template_name>
```

---

## 6) Troubleshooting checklist

### 6.1 Patch script “could not find …”
This almost always means your local `riffgen.py` differs from the script’s expected layout.

Quick checks:

- Search for the generator name:
  - `generate_chordprogression_phrase_8` (older)
  - `generate_chordprogression_segment_16` (newer)
  - `generate_chordsprog_segment_16`

- Search for section kinds:
  - `SECTION_KINDS: List[str] = [` (list)
  - `SECTION_KINDS = (` (tuple)

Once you know which version you’re on, use a patcher built for that version (or patch manually).

### 6.2 “chordsprog is too jazzy”
That means your `generate_chordsprog_segment_16` is using patterns like:
- `cur / prep / next` subdivisions per bar.

The final requested behavior is:
- **no subdivisions**
- one chord event per chord-duration (like chordprogression),
- dyads as described (P5 / dim5 / aug5).

---

## 7) What to change next (recommended final patch)
To match your latest request precisely, implement:

1. Rewrite `generate_chordsprog_segment_16`:
   - use same duration plan as chordprogression
   - emit dyads only:
     - perfect-5 powerchord or inverted
     - dim5/aug5 dyad when the chord implies it

2. Replace chordprogression selection with chordsprog in random song structure:
   - remove `"chordprogression"` from random pool OR map it to `"chordsprog"` when assembling.

If you want, I can produce a single **final** patch script tailored to your exact local file by matching function boundaries in your uploaded `riffgen.py`.

---

## 8) Conversation timeline (condensed)
- Identified chordprogression outputting single notes.
- Spec’d triad/tetrad-only harmonic sections, ignoring `--melody_mode`.
- Added/expanded classical + sad progressions, including secondary dominants and tritone subs.
- Added new `chordsprog` section with rhythmic prep (rejected as too jazzy).
- Iterated patch scripts (v1/v2) and debugged Windows errors due to mismatched file layouts.
- Final request: make chordsprog behave like chordprogression but voiced as dyads/powerchords, and replace chordprogression with chordsprog in song structure.

---
