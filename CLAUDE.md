# CLAUDE.md

Guidance for agents working in this repo.

## What this is

A tool that reads ambulatory ECG (Holter monitor) recordings from a dog and screens for **PVC burden** - the count of premature ventricular complexes, which is the standard metric cardiologists use to screen Dobermans for occult dilated cardiomyopathy. It also flags secondary arrhythmias (brady/tachycardia, pauses) and PVC patterns (couplets, triplets, VT runs). Built around an ALBA Medical DR200 recorder (a rebadged NorthEast Monitoring unit).

**This is a screening/triage aid, not a diagnostic tool. It does not replace a cardiologist's read.** That framing is load-bearing: every generated report carries the disclaimer, and it must stay. Comparing numbers with published reference bands - including color-coding them green / amber / red - is expected; the disclaimer is what carries the not-a-diagnosis framing.

## Architecture

The pipeline is a linear flow, and each stage talks to the next **only through plain data structures**, never by reaching into another module's internals:

```
ingest → quality → detection → classify → arrhythmia → report
         (loader picks the format)  (exclude_beats)   (+ pipeline wires it, cli/gui drive it)
```

- `ingest/` - raw file to a `Recording` (samples + sample_rate + start_time + source). `loader.load_recording()` sniffs the input and dispatches: WFDB records, native DR200 `flash.dat`, or vendor-extracted `flashcN.dat` channels. Adding a new recorder format means adding a loader here and teaching the router about it - nothing downstream changes.
- `quality/gate.py` - `assess_quality()`: samples in, `SignalQuality` (duration + excluded artifact spans) out. Amplitude rules per 5 s window against the recording's own median peak-to-peak (>4x: off-body swings / saturation; <0.1x: lead-off), a flat-line rule (>90% zero deltas), and the first and last minute unconditionally (hookup/removal, the HE/LX vendor convention); excluded windows within 30 s are bridged and padded 2 s. Kurtosis and spectral noise rules were tested and **rejected because they exclude ventricular flutter/VT** (100% of MIT-BIH 207's VFL windows) - read the 2026-08-26 spec's evidence table before adding any noise rule. `exclude_beats()` drops beats inside spans and resets the RR of the first beat after each, so a span can never read as a pause, run, or brady/tachy event. `summarize()` takes the `SignalQuality` and reports `duration_sec` / `analyzed_sec` / `excluded`; PVCs per 24 h scale by analyzed time.
- `detection/detect.py` - `detect_beats()`: R-peak detection (NeuroKit2) + QRS width via a custom energy-envelope method, returns a list of unlabeled `Beat`s. Peaks under 20% of the recording's median R amplitude are dropped first: at slow resting rates NeuroKit invents beats inside long RR gaps, and those phantoms cascade into false PVCs (54 of 65 on Teeny's first recording).
- `classify/rules.py` - `classify_beats()`: rules-based PVC labeling. A beat is "V" only if BOTH premature (short RR vs. a causal rolling baseline) AND wide (long QRS vs. baseline). This module is deliberately isolated behind `Beat in, labeled Beat out` so a learned model could replace it later without touching anything else.
- `arrhythmia/burden.py` - `summarize()`: aggregates labeled beats into an `ArrhythmiaSummary` (PVC burden %, couplet/triplet/VT-run counts, sustained brady/tachy events, pauses, `heart_rate` min/mean/max with times - min/max are 5-beat medians so one phantom beat can't set them - the `longest_run`/`fastest_run` of 3+ PVCs, rated from the RRs inside the run, `duration_sec` / `analyzed_sec` / `excluded` from the `SignalQuality`, and `hourly` rows - per-hour analyzed seconds/beats/rates/PVCs/couplets/runs/pauses counted from the recording start to its end, last partial hour included).
- `report/` - `generate.build_content()` turns beats + summary into a `ReportContent`: four `SummaryGroup`s of `SummaryRow`s (Recording, Heart rate, Ventricular ectopy, Pauses; each row is label + value + the short reference band printed beside it + an `ok`/`caution`/`alert` status from `reference.py`, or no status where no published band exists), two footer lines (color legend, source), the strip sections - heart-rate extremes first (fastest/slowest HR, longest pause, fastest run - the strips a reader checks first, and what would have exposed the off-body "pauses" on the first real recording), then flagged multi-beat runs, then isolated PVCs; the latter two capped at `MAX_STRIPS_PER_SECTION` chosen evenly through the recording with the cap stated in the heading; never cap silently - and the hourly table. `generate.write_report()` renders it to `report.pdf`, the only file written (its path is what `run_analysis` returns). Tests assert on `ReportContent`, not the PDF - matplotlib writes glyph codes, so the PDF is not greppable; end-to-end tests use the `report_text` fixture in `tests/conftest.py`, which spies on what reaches `write_pdf` (rows become `label: value (reference)` lines). `pdf.py` renders with matplotlib `PdfPages` (no extra dependency): the summary panels in a 2x2 grid on page 1 (values colored by status), timeline page with the hourly table beneath it (continued on its own pages past 26 rows), then strip pages; `timeline.py` draws the heart-rate trend over PVC/pause/brady/tachy event lanes from beats + summary only, with excluded spans as hatched grey bands; `strip.py` draws every lead stacked on a standard ECG grid at true clinical scale (25 mm/s, 10 mm/mV, and the scale actually used is always printed) with an N/V/? label over each beat, RR intervals around the flagged beats (every RR when the window is sparse), the flagged beats shaded, and a bracket on a pause; each strip carries a `StripCaption` (title, what it shows with the measured RR/QRS against the classifier's own baseline, whether it is significant, status) built in `generate.py`, and a one-page primer (`common.HOW_TO_READ_STRIPS`) precedes the first strip - the strips are written for a non-expert to check the software's calls, keep them that way; `common.py` holds the text/event helpers; `reference.py` holds the published bands (ESVC Doberman DCM guidelines, Wess et al. 2017) as short strings, the status rules, and the 24-h PVC scaling that is only computed for 20 h or more of analyzed time. Event times are wall-clock labels when `Recording.start_time` is known.
- `pipeline.py` / `cli.py` / `gui/app.py` - `run_analysis()` wires the stages; the CLI and the Tkinter GUI both call it. The GUI is the artifact non-technical users run (see Packaging). Its logic is a frozen `AppState` plus module-level transitions (`choose_recording`, `choose_output`, `run`, the label/status text functions) that call the tkinter dialogs by module-level name so tests can monkeypatch them; `AnalyzerWindow` is the only widget code and runs the analysis on a worker thread, polling with `after()`.

**The boundary contract is the most important design property here.** The classifier is swappable and `ingest` grows new formats precisely because modules only exchange `Recording`/`Beat`/`ArrhythmiaSummary`. Keep it that way: no module should import another's private helpers, and `detect`/`classify`/etc. must stay format-agnostic (they only ever see samples + a sample rate, then time + RR + QRS).

### Data contracts (`types.py`)

- `Recording(samples: np.ndarray, sample_rate: float, start_time, source, channels=None, channel_names=())` - `frozen=True, eq=False` (the `eq=False` matters: the default dataclass `__eq__` raises on the numpy array field). `samples` is the 1-D analysis lead and is all that detection/quality/classification ever see; `channels` is every recorded lead, `(n_channels, n_samples)`, for the report's strips only (DR200 native: all three, `Ch 1`-`Ch 3`; WFDB: every signal; vendor-extracted `flashcN.dat`: `None`). `__post_init__` rejects a channel array that does not line up with `samples`.
- `Beat(time, rr_interval, qrs_duration, label)` - `frozen=True`. `label` is `None` (undetected), `"N"`, `"V"`, or `"U"`.

The pipeline is **sample-rate-agnostic** on purpose - it is validated at 180 Hz (DR200), 360 Hz (MIT-BIH), and 500 Hz (PhysioZoo). Do not hardcode a rate downstream of ingest.

## v1 is rules-based, not ML - on purpose

There is no public canine-labeled PVC ECG dataset (checked exhaustively). Human data (MIT-BIH) can't stand in for canine ground truth - dogs have faster rates and different QRS morphology. So v1 uses causal, per-recording rolling-baseline thresholds rather than a trained model. The path to v2 is accumulating real Teeny recordings cross-checked against a cardiologist's read; the isolated `classify` module is the seam where a learned model would drop in. Don't add an ML classifier without real labeled canine data to validate it against.

## DR200 / `flash.dat` format (do not re-reverse-engineer this)

The DR200 writes NorthEast Monitoring's proprietary `flash.dat` to the SD card. This format was reverse-engineered for this project; the full spec and evidence are in **`docs/dr200-format.md`**, and the parser is `ingest/dr200.py`. Key facts so you don't rediscover them:

- 512-byte blocks: `[u32 length=512][u8 type=0x1e][u8 0][u32 source_position][456 data bytes][42 reserved][u32 checksum]`. The first block is plaintext INI metadata (`SampleRate=180`, `SampleStorageFormat=1`, start date/time, serial).
- Checksum invariant: `sum(block[:508]) + u32(block[508:]) == 0x4CB31` on every active block.
- Data is a **non-linear 4-bit companding delta**, low-nibble-first, three channels interleaved per timepoint, accumulated continuously across blocks. The nibble→delta table is `0, +1, +3, +6, +12, +21, +38, +70, pace, -70, -38, -21, -12, -6, -3, -1`. Nibble 8 is a simultaneous pacemaker marker on all three channels (interpolated, not emitted as a voltage spike). A naive "signed 4-bit delta" model is WRONG - nibble 7 is +70, not +7, which is what lets a real QRS decode.
- Scaling: 12.5 µV per count.
- NorthEast's own tooling: `procfl.exe` (flash.dat → datacard) is **gated behind a HASP hardware dongle**. `unpackdc.exe` (datacard → 16-bit samples) is **public-domain and runs without the dongle**; it's a useful oracle for validating the decoder but is not shipped. The pure-Python parser exists specifically so the packaged app needs no vendor software or Wine.
- The delta table was cross-validated byte-for-byte against `unpackdc` output on NorthEast's bundled demo recording. If you touch the table or decode loop, re-validate against that oracle, not against the parser's own encoder (that would be circular).

## Testing conventions

- **TDD.** Write the failing test, watch it fail, implement, watch it pass. Commit in small steps.
- **Fail closed.** Parsers (especially `dr200.py`) must reject malformed/unsupported input with a clear error rather than silently producing a wrong signal. This is a medical-adjacent tool; a wrong number is worse than a loud failure. Every guard should have a test proving it fires.
- **Tests must not be circular.** For codec/table logic, don't encode-with-table-T then decode-with-table-T and call it verified - a wrong T passes. Assert against literal expected values derived from an independent source (e.g. the `unpackdc` cross-check, or hand-computed physical values).
- **Run the suite:** `pytest`. CI (`.github/workflows/ci.yml`) runs it on every push to `main` and every PR, headless under `xvfb` (the GUI module imports tkinter). Keep CI green.
- **Coverage** is broad, not decoder-only. Genuinely hard-to-test surfaces (real Tk window construction in `gui/app.py::main`) are left uncovered deliberately - the testable core of the GUI (`analyze_and_report`, the file-picker logic) is monkeypatched instead. Don't chase 100% by testing Tk event loops.

## Workflow

- Work on a branch and open a PR; CI must be green. PRs are **squash-merged** with the PR title and body as the commit message, so write the PR body as the commit message you want in `main`'s history. Merged branches are deleted automatically.
- Design specs live in `docs/superpowers/specs/` and implementation plans in `docs/superpowers/plans/`, dated `YYYY-MM-DD-<topic>`. Write the spec before a non-trivial feature; it is the record of *why*.

## Dev commands

```bash
pip install -e ".[dev]"          # install with test/build deps
pytest                            # full suite
pytest --cov=canine_holter --cov-report=term-missing   # coverage (needs pytest-cov)
canine-holter <input> --out report/ [--start-time HH:MM]  # CLI
python -m canine_holter.gui.app                         # GUI
```

Input to the CLI/GUI can be a native DR200 `flash.dat`, a WFDB record (base path or `.hea`), or a vendor-extracted `flashcN.dat`/`.raw` channel.

## Coding conventions

- Python 3.11+, numpy for signal work (vectorize; the decode loop is per-block but vectorized within a block).
- Frozen dataclasses for data contracts.
- Private helpers are `_prefixed` and stay module-local.
- Raise specific, actionable errors (`NativeDR200FormatError` for flash.dat problems, naming the block/field at fault).
- Match the surrounding style; keep comments to constraints the code can't express (e.g. why a loop must stay sequential/causal), not narration.

## Packaging & release

The GUI ships via GitHub Releases so a non-technical user can download and run it with no Python: a signed, notarized macOS `.app` in `canine-holter.dmg`, and an **unsigned** Windows onedir build in `canine-holter-windows.zip` (SmartScreen warns on first run; the README tells the user what to click - signing needs a certificate or Azure Trusted Signing, not yet set up). `canine-holter.spec` (PyInstaller, onedir) builds both from the same file: the macOS `BUNDLE` takes `assets/icon.icns`, the Windows `EXE` takes `assets/icon.ico`; both are resized from the same Microsoft Fluent Emoji "Dog face" 3D PNG (MIT - see `assets/ATTRIBUTION.md`; regenerate with `sips`/`iconutil` and Pillow only, never hand-drawn), and the bundle version is read from the installed package metadata. `.github/workflows/release.yml` (triggered on `v*` tags) runs `build-macos` (Homebrew Python for a working tkinter, codesign, notarize) and `build-windows` (setup-python) in parallel, then a `release` job creates the GitHub release with both files. Tagging `vX.Y.Z` and pushing the tag cuts a release.

Signing secrets live in the repo's Actions secrets (`APPLE_*`). Note there are two distinct "Andrew Aymeloglu" Developer ID certs in the Apple account (this repo's and the `hedgehog` repo's) that are indistinguishable by name in Apple's portal - tell them apart by expiration date if you ever need to.

## Known limits / open items

- Native `flash.dat` support is `SampleStorageFormat=1`, 180 Hz, three-channel only; other modes are rejected rather than guessed.
- The pipeline currently analyzes channel 0; the native parser accepts a channel index in Python but the CLI/GUI don't expose it yet.
- Pacemaker-marker handling is faithful to the documented format but only tested synthetically (no real pacemaker recording available; dogs rarely have them).
- Thresholds (PVC prematurity/width ratios, phantom-peak amplitude fraction, brady/tachy/pause limits) are provisional and per-recording, not clinically calibrated. Calibration waits on real cardiologist-reviewed Teeny recordings.
- Quality gating catches severe artifact (off-body swings, saturation, flat line, lead-off) and the edge minutes. Moderate noise with readable QRS complexes and mid-recording hash noise at normal amplitude are not excluded (the VT-safe option to evaluate for that is a template-correlation SQI, not kurtosis).
- Beat detection misses beats during tachycardia: on Teeny's 2026-08-23 recording `detect_beats` agrees with a Pan-Tompkins second opinion within 0.2% everywhere except 130-150 min (~150 bpm, clean ECG), where it misses 22% and 49% of beats. The misses surface as false "pauses". Open item in `detection/`; the two-detector comparison in the 2026-08-26 spec is a ready-made oracle.
