# CLAUDE.md

Guidance for agents working in this repository.

## Purpose and safety boundary

This tool reads ambulatory canine ECG recordings and screens for PVC burden, PVC patterns, bradycardia, tachycardia, and pauses. It targets the NorthEast Monitoring DR200 and DR400 recorders, which share one native format.

**It is a screening/triage aid, not a diagnostic tool, and does not replace a cardiologist's read.** Every generated report must retain that disclaimer. Comparing measurements with published reference bands, including green/amber/red presentation, is expected; the disclaimer carries the not-a-diagnosis framing.

Thresholds are provisional and are not clinically calibrated. Do not describe their output as a diagnosis.

## Architecture

The pipeline is linear:

```
ingest → quality → detection → classify → arrhythmia → report
```

`pipeline.py` wires the stages; the CLI and GUI drive it. Stages exchange only the public data structures in `types.py` and `arrhythmia/burden.py`. Keep detection, quality, classification, and aggregation format-agnostic, and do not import another stage's private helpers.

- `ingest/`: supported files become a `Recording`. `loader.load_recording()` routes WFDB records and native DR200/DR400 `flash.dat` files.
- `quality/gate.py`: returns recording duration, excluded artifact spans, and the length of any trimmed off-body tail. `exclude_beats()` drops beats inside those spans and resets the first RR afterward so artifact cannot become a pause, run, or rate event. The lead-off rule catches the DR400's open-electrode tone (a square wave at half the sample rate) and the amplitude median ignores tone and flat windows. Kurtosis and spectral-noise rules were rejected because they excluded ventricular flutter/VT; read the signal-quality spec before adding a noise rule.
- `detection/detect.py`: NeuroKit2's QRS-burst detection per lead, reimplemented in `_qrs_fiducials` so that a burst with no local maximum (a QRS that is a small r and a deep S on that lead) yields a fallback fiducial instead of nothing. A fallback can only corroborate a beat another lead resolved, never make one alone. Then lead agreement (a beat needs two leads within 150 ms, at least one resolved; its QRS width is the median of the per-lead widths, each measured at that lead's own peak), plus custom QRS width measurement. Its amplitude window must cover the whole ±150 ms QRS search range. Rate-gated search-back and T-wave rejection address separate, locally gated failure modes. Extend the hand-counted Teeny fixtures before retuning thresholds.
- `classify/rules.py`: causal rolling-baseline rules. A PVC (`V`) must be both premature and wide; a ventricular escape beat (`E`) is wide and at least 1.5x late instead. Neither feeds the baseline. The sequential loop is intentional; future beats must never influence the current beat.
- `arrhythmia/burden.py`: produces `ArrhythmiaSummary`, including burden, runs (consecutive `V` only; an `E` ends a run), escape-beat times, pauses (at 2.5 s and 5 s), rate events and rate shares, heart-rate statistics, heart-rate variability (SDNN, RMSSD, pNN50 over NN intervals), quality accounting, and hourly rows, which align to clock hours when the recording start is known.
- `report/`: builds plain `ReportContent`, then renders `report.pdf`. Reports include the disclaimer, six summary panels with reference bands, quality accounting, timeline, hourly table, and reviewable ECG strips, ending with one strip per hour. Strip caps must always be stated. The cardiologist's HE/LX report is the external oracle; `docs/superpowers/specs/2026-09-02-cardiologist-report-parity-design.md` records how its numbers map to ours.
- `gui/app.py`: a small Tkinter wrapper around `run_analysis()`. Analysis runs on a worker thread.

Detailed rationale, measurements, rejected approaches, and acceptance evidence live in `docs/superpowers/specs/`. Completed implementation steps belong in git history, not duplicated plans.

## Data contracts

`Recording(samples, sample_rate, start_time, source, channels=None, channel_names=())` is frozen with `eq=False`. `samples` is the one-dimensional lead that quality gating judges. `channels`, when present, contains every lead with shape `(n_channels, n_samples)`; detection runs on all of them and keeps the beats at least two leads agree on, and the report draws them. A single-lead input is detected on `samples`. Keep the channel array aligned with `samples`.

`Beat(time, rr_interval, qrs_duration, label)` is frozen. Labels are `None` before classification, `"N"`, `"V"`, `"E"` (ventricular escape beat), or `"U"`.

The downstream pipeline is sample-rate-agnostic and has been exercised at 180, 360, and 500 Hz. Never hardcode 180 Hz outside DR200 ingestion.

## Classifier scope

Version 1 is rules-based: causal timing and width rules with a rolling baseline. That is a consequence of the ground truth we have, not a preference. As of 2026-09-02 the only external truth is the cardiologist's HE/LX report for the 2026-08-27 recording: hourly counts and a handful of strip times, not a label per beat. Any rule that goes beyond timing and width (beat shape against a template, or anything trained) needs beat-level truth to be tuned and judged; `docs/superpowers/specs/2026-09-02-beat-shape-design.md` shows a shape rule over-calling PVCs 6-7x without it, and lists the stretches whose beat-by-beat read would unblock it. Human ECG data is not a substitute for canine validation.

## Native DR200/DR400 format

Do not re-reverse-engineer `flash.dat`; the evidence and full specification are in `docs/dr200-format.md`, and the parser is `ingest/dr200.py`. Both recorders write the same format.

- Blocks are 512 bytes: length, type, source position, 456 data bytes, reserved bytes, and checksum.
- Every active block satisfies `sum(block[:508]) + u32(block[508:]) == 0x4CB31`.
- `SampleStorageFormat=1` uses low-nibble-first, three-channel, nonlinear four-bit deltas accumulated continuously across blocks.
- The delta table is `0, +1, +3, +6, +12, +21, +38, +70, pace, -70, -38, -21, -12, -6, -3, -1`. Nibble 8 is a simultaneous pacemaker marker on all three channels, not a voltage delta.
- Scaling is 12.5 µV per count.
- Bytes 466..472 of an ECG block are the recording sequence number and recorder serial; the parser stops at the first ECG block whose pair differs from the first block's, because a reused card keeps older recordings' blocks with valid checksums and sometimes contiguous source positions.
- The table and decode loop were validated byte-for-byte against NorthEast's `unpackdc` output. Revalidate against that independent oracle after any decoder change; never validate an encoder and decoder built from the same table against each other.
- Native support is intentionally limited to 180 Hz, three-channel format 1. Reject other modes rather than guessing.
- A native file is recognized by the name `flash.dat` or by its first block's content, so renamed copies load.

## Testing

Use TDD for behavior changes: write the failing test, see it fail, implement, then see it pass.

Parsers must fail closed with specific, actionable errors. Every malformed-input guard needs a test. Codec tests must use literal values or an independent oracle, not circular encode/decode fixtures.

Run:

```bash
pip install -e ".[dev]"
pytest
pytest --cov=canine_holter --cov-report=term-missing
```

CI runs the suite on every push to `main` and every pull request. Do not chase 100% coverage by testing Tk event loops; test GUI controller behavior at its boundaries.

## Workflow and style

- Work on a branch and open a PR against `main`; CI must be green.
- PRs are squash-merged, using the PR title and body as the commit message.
- Write a dated design spec in `docs/superpowers/specs/` before a non-trivial feature. Specs record why; git records implementation history.
- Use Python 3.11+ and NumPy for signal work. Vectorize where practical, except for stateful/causal loops.
- Use frozen dataclasses for boundary data.
- Prefix module-private helpers with `_` and keep them module-local.
- Prefer specific errors such as `NativeDR200FormatError`, naming the bad block or field.
- Comments should preserve constraints and rationale the code cannot express, not narrate implementation history.

## Running and packaging

```bash
canine-holter <input> --out report/ [--start-time HH:MM]
python -m canine_holter.gui.app
```

The GUI ships through GitHub Releases as a signed/notarized macOS app and an unsigned Windows onedir build. `canine-holter.spec` builds both. Icons come from the attributed Microsoft Fluent Emoji source in `assets/`; do not replace them with unlicensed or hand-drawn variants.

`.github/workflows/release.yml` builds both platforms for `v*` tags and creates the release. Apple signing secrets live in Actions. Two Developer ID certificates have the same display name; distinguish this repository's certificate from the `hedgehog` certificate by expiration date.

## Known limits

- The pipeline analyzes channel 0; CLI and GUI do not yet expose channel selection.
- Supraventricular ectopy, AV block, and junctional escape beats are not assessed; they need P-wave analysis. The report says so for SVPBs rather than printing zero.
- Pacemaker handling is format-faithful but only synthetically tested.
- Detection/classification and brady/tachy/pause thresholds remain provisional.
- Quality gating catches severe artifact and the first/last minute, but not all moderate or normal-amplitude hash noise. Do not add kurtosis or template-correlation SQI without new evidence; existing specs document why they failed.
- Lead agreement removes one lead's T-wave and P-wave detections and its posture-shifted widths, but a noise burst or a T wave that every lead detects as a beat still reads as a PVC.
- Escape beats are undercounted: the width measurement calls few of the night-time escape beats wide (8 of the cardiologist's 32 on 2026-08-27); the ones it misses differ by shape, not width. Runs of escape beats (idioventricular rhythm) are not counted. See the beat-shape spec for what unblocks both.
