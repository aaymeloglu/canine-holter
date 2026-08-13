# canine-holter

Tools for recording and interpreting ambulatory ECG (Holter monitor) data from a dog, starting with an ALBA Medical DR200 unit.

## Background

Started after our Doberman (Teeny) had a seizure-like episode with no clear diagnosis and a month+ wait for a cardiology referral. Rather than wait on the clinic's loaner Holter program every time, we bought our own DR200 recorder + canine vest. This repo reads and interprets the raw recordings ourselves - screening for PVC burden (the standard Doberman occult DCM metric) and other arrhythmias.

**This is a screening/triage aid, not a diagnostic tool.** It doesn't replace a cardiologist's read.

## Status

v1: rules-based PVC detection, tested against MIT-BIH (human) and PhysioZoo (canine, no PVC labels) data. DR200-native file support isn't built yet - see `docs/superpowers/specs/2026-08-13-pvc-detection-design.md` for why and what's blocking it.

## Usage

```bash
pip install -e ".[dev]"
canine-holter path/to/recording --out report/
```

Or launch the GUI: `python -m canine_holter.gui.app` (or download the signed `.app` from [Releases](https://github.com/aaymeloglu/canine-holter/releases)).

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for design history.
