# Signal-quality gating, time analyzed, and a denser summary page

**Date:** 2026-08-26
**Status:** approved design; supersedes the "never says normal or abnormal"
non-goal in `2026-08-24-phantom-beats-pvc-strips-reference-ranges-design.md`.

## Problem

Every clinical Holter report separates *recording duration* from *time
analyzed* and prints the artifact fraction, per recording and per hour
(QRS Diagnostics sample report; NorthEast Monitoring HE/LX "General
Profile"; Welch Allyn HScribe "Profile"). Ours does not. On Teeny's first
recording the last ~5 minutes are off-body (the vest coming off): 25 mV
swings, then hash noise at normal amplitude, then flat line. The detector
invents sparse beats in that stretch and the report counts 31 of its 69
"pauses" there, and scales PVCs per 24 h by wall-clock time.

Two presentation problems on the same page: the Summary is a 17-line
single column that fills most of page 1, and the Reference ranges block
below it is a paragraph a reader has to cross-reference by hand.

## Goals

1. Exclude signal that is not analyzable ECG, report how much was
   excluded, and never let an excluded stretch produce a pause, a
   brady/tachy event, a run, or a beat.
2. Summary page as four titled panels in a 2x2 grid, each value beside
   the published band it is compared with, coloured by where it falls.
3. Keep the pipeline's boundary contract: quality talks to the rest of the
   pipeline through one frozen dataclass.

## Non-goals

- Moderate noise that still contains readable QRS complexes (the MIT-BIH
  Noise Stress Test at 6 dB SNR). Vendors tolerate it ("Excellent" setting
  in HE/LX) and the beat detector is validated on it; excluding it is a
  calibration question for later.
- Off-body hash noise at normal amplitude in the *middle* of a recording.
  See "Rejected rules" for why the noise measures that would catch it are
  unsafe here.
- Beat-detection misses during tachycardia (found during this work, see
  "Follow-ups"). Separate change to `detection/`.
- Multi-channel quality (channel switching as HScribe does). The pipeline
  analyzes channel 0.

## 1. Quality stage (`quality/gate.py`)

```
ingest -> quality ---------------------------------+
                                                   v
          detection -> exclude_beats -> classify -> arrhythmia -> report
```

`assess_quality(samples, sample_rate) -> SignalQuality` sees only samples
and a rate (format-agnostic, like detection). It splits the recording into
non-overlapping `WINDOW_SEC = 5` s windows and excludes a window when any
rule fires:

| Rule | Constant | Fires when | What it is for |
|---|---|---|---|
| High amplitude | `MAX_AMPLITUDE_RATIO = 4.0` | window peak-to-peak > 4x the median window peak-to-peak of the recording | off-body swings, saturation, gross motion |
| Low amplitude | `MIN_AMPLITUDE_RATIO = 0.1` | window peak-to-peak < 0.1x the median | lead-off, flat line at a rail |
| Flat line | `MAX_FLAT_FRACTION = 0.9` | more than 90% of the consecutive-sample differences in the window are exactly zero | flat line anywhere (Clifford 2012 flat-line SQI; CinC 2011 rule). A true flat line is ~100%; a quiet DR200 baseline at 12.5 uV/count reaches ~50% on Teeny's resting stretches, so 0.5 would have excluded real ECG |
| Recording edges | `EDGE_SEC = 60` | first and last 60 s of the recording | hookup and removal; the HE/LX convention ("none of the signal except the first minute and the last minute is called artifact" even with its artifact detector off) |

Then: bridge gaps of `BRIDGE_SEC = 30` s or less between excluded windows
(the off-body tail has quiet stretches that are not ECG either), pad each
span by `PAD_SEC = 2` s on both sides (edge beats are half-buried in
noise), clip to the recording. The remainder after the last full window
needs no rule of its own: the last-minute edge span always covers it. A
median peak-to-peak of zero means no window has a signal at all: the whole
recording is excluded (fail closed).

Ratios are relative to the recording's own median because DR200 samples
carry a decoder DC offset (the delta stream accumulates from zero) and
gain varies by recorder and lead; nothing downstream of ingest may assume
an absolute scale.

### Evidence (5 s or 10 s windows; identical conclusions at both)

| Data | Windows the rules exclude | Notes |
|---|---|---|
| Teeny 2026-08-23, on-body part (0-142.9 min) | 0.1% (the 2 s hookup) | play/tachycardia burst at 17:49 has ratio 1.1 |
| Teeny, off-body tail (142.9-146.7 min) | 64% of windows before bridging; one span after | 31 of 69 report "pauses" were here |
| Teeny, hash noise (146.7-148.1 min) | caught by the last-minute edge rule + bridging | amplitude alone misses it (ratio 1-1.5) |
| MIT-BIH 119 (large PVCs) | 0% | max ratio 1.11 |
| PhysioZoo Dog_01 | 0% | max ratio 1.54 |
| MIT-BIH NST 118e24 / 118e06 (24 / 6 dB SNR), clean and noisy minutes | 0% / 0% | moderate noise is tolerated by design |
| MIT-BIH NST 118e_6 (-6 dB SNR) | 97% of noisy minutes, 0% of clean minutes | 0 of 1291 annotated clean-segment beats lost |
| MIT-BIH 207 ventricular flutter / VT episodes, both leads | 0% | max ratio 3.02 (V1, flutter); 25% margin to the threshold |

### Rejected rules

- **Kurtosis (kSQI < 4 or < 5; Zhao 2018, Clifford 2012).** Catches
  moderate noise well (67-92% of 6 dB NST windows) but excludes 100% of
  the ventricular-flutter and VT windows in MIT-BIH 207 and 4.5% of its
  normal rhythm: a near-sinusoidal signal has low kurtosis whether it is
  noise or VT. Deleting the events the tool exists to find is the worst
  possible failure, so no kurtosis rule.
- **Two-detector agreement (bSQI; Li & Clifford 2008).** On Teeny's
  recording the windows it flags in the on-body part (135-139 min) are
  clean ECG at ~150 bpm where NeuroKit's detector misses most beats, not
  artifact; and on 118e_6 it flags 59% of the *clean* minutes because the
  -6 dB segments derail the detectors' adaptive thresholds. It measures
  detector health, not signal quality. Kept as a follow-up test oracle.
- **High-frequency power ratio (P > 40 Hz / P > 1 Hz).** Safe for VT
  (0.000 on 207) but catches only 25% of the hash-noise windows at a 0.3
  threshold and none of the electrode-motion noise. Not worth a rule.

## 2. Data contracts

```python
@dataclass(frozen=True)
class SignalQuality:
    duration_sec: float
    excluded: tuple[tuple[float, float], ...]   # (start, end) seconds, sorted, non-overlapping
    @property
    def analyzed_sec(self) -> float
    def analyzed_within(self, start: float, end: float) -> float   # seconds of [start, end) not excluded
    def contains(self, t: float) -> bool
```

`exclude_beats(beats, quality) -> list[Beat]` drops every beat inside an
excluded span and returns the first beat after each span with
`rr_interval=None` (the contract already means "no previous beat"). No
other stage looks at spans to decide anything about beats.

`ArrhythmiaSummary` gains `duration_sec`, `analyzed_sec`, and `excluded`;
`HourRow` gains `analyzed_sec`. `summarize(beats, dog_weight_class,
quality=None)`: without a `SignalQuality` (the report-only path and the
existing tests) duration is the last beat's time, analyzed equals
duration, nothing is excluded. Hours run to `duration_sec`, so a recording
whose last minutes are excluded still lists them.

`run_analysis` wires `assess_quality` before detection and passes the
result to `exclude_beats` and `summarize`. No new CLI or GUI options.

## 3. Reference scaling

`pvc_per_24h(pvc_count, analyzed_sec)` scales by analyzed time, and the
`MIN_HOURS_FOR_24H_SCALING = 20` rule applies to analyzed hours. The
report says "needs >= 20 h analyzed" when it declines.

## 4. Report

### Summary page

`ReportContent.summary_lines` and `reference_lines` are replaced by
`summary_groups: list[SummaryGroup]`:

```python
@dataclass(frozen=True)
class SummaryRow:
    label: str
    value: str
    reference: str = ""          # short band text printed beside the value, e.g. "<50 | 50-300 | >300"
    status: str | None = None    # "ok" | "caution" | "alert" | None (uncoloured)

@dataclass(frozen=True)
class SummaryGroup:
    title: str
    rows: list[SummaryRow]
```

Four groups, laid out 2x2 (Recording, Heart rate / Ventricular ectopy,
Pauses):

| Group | Rows (label: value [reference]) | Status |
|---|---|---|
| Recording | Start; Duration; Analyzed (h m, % of duration); Excluded (h m, "artifact / off-body"); Beats | Analyzed: ok >= 20 h, caution below (the 24 h bands need it) |
| Heart rate | Mean; Slowest (bpm at time); Fastest (bpm at time); Brady events; Tachy events | none (no published Doberman band; brady/tachy thresholds are provisional) |
| Ventricular ectopy | PVCs (count, burden %); per 24 h [<50 \| 50-300 \| >300]; Couplets [0]; Triplets [0]; VT runs 4+ [0]; Longest run; Fastest run [<180 bpm] | per 24 h: ok <50, caution 50-300, alert >300, uncoloured "n/a" when analyzed < 20 h. Couplets/Triplets/VT runs: ok 0, alert >= 1. Fastest run: ok none, caution < 180 bpm, alert >= 180 |
| Pauses | Pauses >= 2.5 s (count); Longest [<2.5 \| 2.5-5 \| >5 s] | Longest: ok < 2.5, caution 2.5-5, alert > 5; count uncoloured |

Colours: ok `#2e7d32`, caution `#b26a00`, alert `#c62828`; reference text
8 pt grey. Below the panels, two lines replace the old block: a legend
("Colours compare each value with the band printed beside it (ESVC
Doberman DCM screening guidelines, Wess et al. 2017). They are not a
diagnosis.") and the disclaimer already at the top stays where it is.

The "never normal or abnormal" wording is removed from `reference.py`, the
README, and the earlier spec's non-goals; CLAUDE.md gets a sentence saying
colour-coding against published bands is expected and the disclaimer is
what carries the not-a-diagnosis framing.

### Timeline and hourly table

Excluded spans are drawn as grey hatched bands across both timeline
panels. The hourly table gains an `Analyzed (min)` column after `Hour`.

### Test fixture

`report_text` joins each row as `label: value (reference)` plus the group
titles, so end-to-end tests keep grepping plain text.

## 5. Tests

- `tests/quality/test_gate.py`: synthetic 1 mV beats at 180 Hz with (a) a
  10 mV noise burst, (b) a flat stretch, (c) a 0.05 mV stretch; literal
  expected spans including edge minutes, bridging of a 20 s gap, no
  bridging of a 40 s gap, 2 s padding, remainder window, all-zero
  recording fully excluded, a 30 s recording fully excluded by the edge
  rule. `exclude_beats`: beats inside a span dropped, first beat after it
  has `rr_interval=None`, others untouched.
- `tests/arrhythmia/test_burden.py`: `duration_sec`/`analyzed_sec`/`excluded`
  with and without quality; hours extend to duration; `HourRow.analyzed_sec`.
- `tests/report/test_reference.py`: scaling uses analyzed seconds; the 20 h
  rule on analyzed time.
- `tests/report/test_generate.py`: group titles, row values, each status
  rule at its boundaries (49/50/300/301 per 24 h; 179/180 bpm; 2.49/2.5/5/5.01 s).
- `tests/report/test_timeline.py`: one band per excluded span.
- `tests/test_pipeline.py`: end-to-end text contains `Analyzed:` and the
  native-flash fixture reports its edge-minute exclusion.

Manual validation (not CI: the recording is gitignored): run Teeny's
2026-08-23 flash.dat and report excluded spans, beats kept, pauses before
and after, PVC count before and after.

## Follow-ups found while researching this

1. **Beat detection misses during tachycardia.** Against a Pan-Tompkins
   second opinion, `detect_beats` agrees within 0.2% everywhere on Teeny's
   recording except 130-150 min, where it misses 22% and 49% of beats on
   clean ECG at ~150 bpm. Those misses become false "pauses" (three of
   the 42 that survive gating are exactly there). Needs its own
   investigation in `detection/`; the bSQI computation is a ready-made
   test oracle.
2. Mid-recording hash noise at normal amplitude is not excluded (see
   Non-goals). If it shows up on real recordings, a template-correlation
   SQI (HScribe's per-beat noise property; NeuroKit's `averageQRS`) is the
   VT-safe option to evaluate, not kurtosis.

## Sources

- QRS Diagnostics sample Holter report: https://www.vectracor.com/wp-content/uploads/2020/03/ML962-Holter-Sample-Report.pdf
- NorthEast Monitoring HE/LX Analysis operator's manual (Scanning Criteria, Signal quality; Tables): https://nemon.com/supportfiles/NEMM027-Rev-P%20HE-LX%20Analysis.pdf
- Welch Allyn HScribe clinician's guide (lead-fail status, per-beat quality): https://www.hillrom.com/content/dam/hillrom-aem/us/en/sap-documents/LIT/9515-/9515-213-71-ENGLITPDF.pdf
- Clifford et al. 2012, Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms, Physiol Meas 33:1419
- Zhao & Zhang 2018, SQI quality evaluation mechanism of single-lead ECG signal, Front Physiol 9:727: https://www.frontiersin.org/articles/10.3389/fphys.2018.00727/full
- Li, Mark & Clifford 2008, Robust heart rate estimation ... signal quality indices, Physiol Meas 29:15: https://pmc.ncbi.nlm.nih.gov/articles/PMC2259026/
- Tat et al., PhysioNet/CinC Challenge 2011 (flat-line rule): https://www.cinc.org/archives/2011/pdf/0441.pdf
- MIT-BIH Noise Stress Test Database (nstdb) and MIT-BIH Arrhythmia record 207 (PhysioNet), used for the evidence table.
