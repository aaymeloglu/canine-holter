# Phantom-beat rejection, per-PVC strips, reference ranges — Design

Date: 2026-08-24

## Motivation

Teeny's first real recording (2026-08-23, 2 h 28 m) reported 65 PVCs, 2
couplets, and 64 pauses. Inspecting the strips showed that 54 of the 59
PVCs flagged before the vest slipped sit on flat baseline: peak-to-peak
signal of 0.03–0.11 mV at the flagged instant, against ~2.9 mV for a real
R-wave. At Teeny's resting rate (~45–60 bpm, strong sinus arrhythmia) the
R-peak detector invents a beat inside long RR gaps; the width estimator
then measures noise as "wide", the RR to the previous real beat looks
"premature", and the rules call it a PVC. Both "couplets" were also false
(one is sinus tachycardia during play, the other is the off-body
artifact).

Beyond the wrong count, the report gives a non-specialist no way to tell
whether the numbers matter: isolated PVCs - where the errors live - are
never shown, and nothing says what a normal count looks like.

Three changes, in the order they matter:

1. Reject phantom beats in detection.
2. Plot a strip for every flagged PVC, not only multi-beat runs.
3. Print the published Doberman screening reference ranges beside the
   numbers.

## Non-goals

- Signal-quality / off-body gating and "analyzable duration" (separate
  work; the artifact tail still produces junk).
- Recalibrating the brady/tachy thresholds or the sustained-event window.
- ~~Any wording that implies a diagnosis. Reference ranges are printed for
  comparison; the report never says "normal" or "abnormal" about the dog.~~
  Superseded 2026-08-26: values are colour-coded against the published
  bands; see `2026-08-26-signal-quality-and-summary-page-design.md`.
- Adding amplitude to the `Beat` contract. Rejection happens entirely
  inside detection, which is the only stage that sees samples.

## 1. Phantom-beat rejection (`detection/detect.py`)

After NeuroKit returns R-peak indices, measure each peak's amplitude as
the peak-to-peak range of the cleaned signal within ±60 ms
(`R_AMPLITUDE_WINDOW_SEC = 0.06`) of the index. The reference amplitude is
the median over all detected peaks in the recording. A peak whose
amplitude is below `MIN_R_AMPLITUDE_FRACTION = 0.2` of that median is
dropped before RR intervals and QRS widths are computed, so a phantom in
a gap becomes a long RR (which is what was really there) rather than two
short ones.

Why 0.2: Teeny's phantoms are all under 4 % of the median; real PVCs can
be smaller than sinus beats but not five times smaller on a working
electrode. The threshold is relative to the recording's own R-waves, so
it is sample-rate- and gain-agnostic.

Edge cases: fewer than two surviving peaks → `[]` (existing contract). A
zero median (every peak on flat signal) → `[]`; there is no signal to
reason about, and failing closed beats emitting beats we cannot vouch
for.

The helper `_reject_low_amplitude_peaks(cleaned, r_peaks, sample_rate)`
is module-local and tested directly with literal arrays; `detect_beats`
is tested through a monkeypatched `nk.ecg_peaks` that returns a known
peak set containing a phantom index. MIT-BIH and PhysioZoo fixture tests
must keep passing (real beats must not be rejected).

## 2. A strip for every PVC (`report/`)

`common.py` gains `isolated_pvcs(beats)` (runs of exactly one "V") beside
the existing `flagged_runs` (runs of 2+). The report gets a second
section, **Isolated PVCs**, after Flagged events, with one strip per PVC,
labelled `PVC 3: isolated PVC at ~17:50:20 (t=8232.8s)` and written to
`pvc_N_strip.png`.

Both sections are capped at `MAX_STRIPS_PER_SECTION = 24` strips (8 PDF
pages each). When a section exceeds the cap, the strips shown are evenly
spaced through the recording (`select_evenly(items, max_n)`), and the
section heading says so: `Isolated PVCs (24 of 312 shown, evenly spaced
through the recording)`. The cap is never silent.

`draw_strip` takes an optional `mark_times` and draws a faint vertical
line at each flagged beat so a reader can find the beat in question
without knowing what a PVC looks like. Both sections pass their beats'
times.

PDF: strip pages for both sections, each page headed with its section
title. In report-only mode (no samples) the flagged events and isolated
PVCs are listed as text on pages after page 1 instead of on page 1, so
page 1 always holds summary, reference ranges, and timeline regardless of
how many events there are. (Previously three events fit on page 1 and a
long list would have collided with the timeline.)

## 3. Reference ranges (`report/reference.py`, new)

Source: ESVC screening guidelines for DCM in Doberman Pinschers (Wess et
al., J Vet Cardiol 2017) - Holter bands per 24 h: under 50 PVCs normal
(any PVCs merit attention); 50–300 equivocal, repeat within the year (two
such recordings within a year are diagnostic); over 300 diagnostic of
occult DCM. Pause context from canine Holter studies: pauses over 2.5 s
are common in healthy dogs with sinus arrhythmia; pauses over ~5 s, or
with collapse/fainting, warrant review.

Summary lines gain:

- `Longest pause: 2.97 s` - the longest RR interval in the recording.
  `ArrhythmiaSummary` gains `longest_pause_sec: float | None = None`
  (None when no beat has an RR). Without a duration the pause reference
  is meaningless.
- `PVCs per 24 h: 12 (scaled from 22h 10m)` when the recording is at
  least `MIN_HOURS_FOR_24H_SCALING = 20` h; otherwise `PVCs per 24 h: not
  computed (recording is 2h 28m; needs >= 20 h)`. PVC frequency varies
  across a day, so a short recording is not scaled.

A **Reference ranges** section follows the Summary in both markdown and
PDF page 1: one line each for PVCs/24 h, complex ectopy (couplets,
triplets, VT: any is worth a cardiologist's review), pauses, and the
source. When the recording is under 20 h a final line says the PVC band
does not apply to it. The existing disclaimer stays at the top.

## Testing

- `tests/detection/test_detect.py`: the helper drops a 0.05-amplitude
  peak among 2.0-amplitude peaks and keeps uniform peaks; a zero median
  yields no peaks; `detect_beats` with a monkeypatched peak set drops the
  phantom and reports the gap as one long RR. Fixture tests unchanged.
- `tests/arrhythmia/test_burden.py`: `longest_pause_sec` is the max RR;
  None with no RR.
- `tests/report/test_reference.py`: `pvc_per_24h` is None under 20 h,
  equals the count at 24 h, halves at 48 h; the lines contain the 50/300
  bands and the short-recording note only when applicable.
- `tests/report/test_generate.py`: an isolated PVC appears under
  "Isolated PVCs" with `pvc_1_strip.png` (and still not under "Flagged
  events"); 30 isolated PVCs yield 24 strips and a "24 of 30" heading;
  the reference section and longest-pause line are in the markdown.
- `tests/report/test_pdf.py`: isolated-PVC strips add pages; no-samples
  mode lists events on a text page (page count updated).
- `tests/report/test_strip.py`: `mark_times` adds a vertical line per
  time.
- Manual: re-run Teeny's `flash.dat` and record the before/after counts
  in the PR.
