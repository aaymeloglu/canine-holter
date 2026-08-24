# canine-holter

Tools for recording and interpreting ambulatory ECG (Holter monitor) data from a dog, starting with an ALBA Medical DR200 unit.

## Background

Started after our Doberman (Teeny) had a seizure-like episode with no clear diagnosis and a month+ wait for a cardiology referral. Rather than wait on the clinic's loaner Holter program every time, we bought our own DR200 recorder + canine vest. This repo reads and interprets the raw recordings ourselves - screening for PVC burden (the standard Doberman occult DCM metric) and other arrhythmias.

**This is a screening/triage aid, not a diagnostic tool.** It doesn't replace a cardiologist's read.

## Status

v1: rules-based PVC detection, tested against MIT-BIH (human) and PhysioZoo (canine, no PVC labels) data. Native DR200 `flash.dat` ingestion supports the recorder's three-channel, 180 Hz `SampleStorageFormat=1` data, and vendor-extracted `flashc0.dat` through `flashc2.dat` channels are also supported. See [DR200 format research](docs/dr200-format.md) for the format evidence and current limits.

## Usage

```bash
pip install -e ".[dev]"
canine-holter /Volumes/DR200/flash.dat --out report/
```

The input can also be a WFDB record or a vendor-extracted DR200 channel such as `flashc0.dat`.

The primary output is `report/report.pdf`: summary stats and a whole-recording timeline (heart rate plus PVC / pause / brady / tachy lanes) on the first page, then a rhythm strip for each flagged event (couplets, triplets, VT runs) and for each isolated PVC, with the flagged beats marked. Each strip section is capped at 24 strips spread evenly through the recording, and the heading says when the cap applied. Page 1 also prints the ESVC Doberman screening reference ranges (PVCs per 24 h, complex ectopy, pauses) next to the numbers, so a reader can see how the counts compare - it never calls a recording normal or abnormal. `report.md`, `timeline.png`, and the strip PNGs are written alongside. Events are labelled by time of day using the start clock in the recording header. If the recorder's clock was wrong, override it with `--start-time 15:36` (or `HH:MM:SS`, or `"YYYY-MM-DD HH:MM"`); a time-only value keeps the recording's own date.

Or launch the GUI: `python -m canine_holter.gui.app` (or download the signed `.app` from [Releases](https://github.com/aaymeloglu/canine-holter/releases)).

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for design history.
