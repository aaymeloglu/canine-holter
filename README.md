# canine-holter

Tools for recording and interpreting ambulatory ECG (Holter monitor) data from a dog, from NorthEast Monitoring DR200 and DR400 recorders.

## Background

Started after our Doberman (Teeny) had a seizure-like episode with no clear diagnosis and a month+ wait for a cardiology referral. Rather than wait on the clinic's loaner Holter program every time, we bought our own DR200 recorder + canine vest. This repo reads and interprets the raw recordings ourselves - screening for PVC burden (the standard Doberman occult DCM metric) and other arrhythmias.

**This is a screening/triage aid, not a diagnostic tool.** It doesn't replace a cardiologist's read.

## Status

v1: rules-based PVC detection, tested against MIT-BIH (human) and PhysioZoo (canine, no PVC labels) data. Native DR200/DR400 `flash.dat` ingestion supports the recorders' three-channel, 180 Hz `SampleStorageFormat=1` data, stops at stale blocks left by earlier recordings on a reused card, and trims a long off-body tail (a recorder mailed back running). Vendor-extracted `flashc0.dat` through `flashc2.dat` channels are also supported. See [DR200/DR400 format research](docs/dr200-format.md) for the format evidence and current limits.

## Usage

```bash
pip install -e ".[dev]"
canine-holter /Volumes/DR200/flash.dat --out report/
```

The input can also be a renamed native file (any `.dat` that starts with the recorder metadata), a WFDB record, or a vendor-extracted channel such as `flashc0.dat`.

The primary output is `report/report.pdf`. Page 1 is four panels (recording, heart rate, ventricular ectopy, pauses); each value sits beside the ESVC Doberman screening band it is compared with and is colored green / amber / red by where it falls. The recording panel states how much of the recording was analyzed: the first and last minute and any off-body, saturated, or flat stretches are excluded before anything is counted, and the timeline shades them. Page 2 is the whole-recording timeline (heart rate plus PVC / pause / brady / tachy lanes) with an hourly table. Then, after a one-page primer on reading them, the rhythm strips: the heart-rate extremes and longest pause first, then each flagged event (couplets, triplets, VT runs), then each isolated PVC. Every strip shows all three recorder channels on standard ECG paper at 25 mm/s and 10 mm/mV, labels every beat (N / V / ?), shades the beats it is about, prints the intervals around them in seconds, and says in plain English what it shows - with the measurements behind the software's call - and whether it matters, coloured against the same bands as page 1. Each strip section is capped at 24 strips spread evenly through the recording, and the heading says when the cap applied. The PDF is the only file written. Events are labelled by time of day using the start clock in the recording header. If the recorder's clock was wrong, override it with `--start-time 15:36` (or `HH:MM:SS`, or `"YYYY-MM-DD HH:MM"`); a time-only value keeps the recording's own date.

Or launch the GUI: `python -m canine_holter.gui.app`, or download it from [Releases](https://github.com/aaymeloglu/canine-holter/releases): `canine-holter.dmg` for macOS (signed and notarized) or `canine-holter-windows.zip` for Windows (unzip, run `canine-holter.exe`; the Windows build is not code-signed, so SmartScreen will warn the first time - click *More info*, then *Run anyway*). It is three steps: choose the recording, choose the folder to write `report.pdf` into, then Run; the status line shows progress and the result, and the report opens when it is done.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/superpowers/specs/` for design rationale and rejected alternatives; implementation history remains in git.
