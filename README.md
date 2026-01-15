[README.md](https://github.com/user-attachments/files/24630875/README.md)
# riffgen

Metal guitar riff + progression generator for D‑standard tuning. Outputs ASCII tab and MIDI (riff, vocals, and optional solo).

## Quick start

```bash
python3 riffgen.py
```

Outputs:
- `riff.txt` (ASCII tab + chord grid)
- `riff.mid` (guitar)
- `riff_vocals.mid` (vocal melody)
- `riff_solo.mid` (solo melody, only when a solo section is chosen)

## Core options

```bash
python3 riffgen.py --section gallopsprog
python3 riffgen.py --section chordsprog happy
python3 riffgen.py --prog
python3 riffgen.py --nonprog
python3 riffgen.py --mood happy
```

## Progression modes

- Default chord mood is **sad**.
- Use `--mood happy` (or `--chord_mood happy`) to force happy progressions.
- `--prog` forces all sections to use prog variants where possible.
- `--nonprog` forces non‑prog variants.

## Section forcing

```bash
python3 riffgen.py --section gallops
python3 riffgen.py --section gallopsprog
python3 riffgen.py --section chordsprog sad
```

## Notes

- Song form is 16‑bar segments; bridge is 8 bars.
- The 2nd 8 bars of a 16‑bar section are transposed ±3 semitones (except bridge).
- `solomelody` is generated for exactly one prog section per song and follows that section’s progression.

