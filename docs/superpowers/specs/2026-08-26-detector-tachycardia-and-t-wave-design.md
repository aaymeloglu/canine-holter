# Beat detection: tachycardia misses and T-wave double detections

**Date:** 2026-08-26
**Status:** approved

## Problem

Teeny's 2026-08-25 recording (24.35 h, DR200, v0.3.1 + PR #21) reported a
4-beat "VT run at 150 bpm" and, once the negative-QRS amplitude fix landed,
96 PVCs of which 38 sat in the single hour 14:25-15:25. Both are beat
detection failures, in opposite directions:

1. **Tachycardia misses.** NeuroKit's `neurokit` method thresholds a
   0.1 s-smoothed absolute gradient at 1.5x its 0.75 s boxcar *mean*. At
   ~150 bpm the mean rises until the threshold sits on the QRS gradient
   itself; each beat crosses it in 6-17 ms fragments that the method's
   region-length floor (0.4x the recording-wide mean region, 37 ms here)
   discards. On the hand-counted window 08:47:52 +14 s it finds 21 of 36
   beats and nothing at all for 3.4 s. The classifier's RR baseline
   inflates to 1.27 s, four consecutive sinus beats at 0.40 s read as
   premature, and the report prints a VT run. The same failure produced
   the 130-150 min misses on the 2026-08-23 recording.
2. **T waves detected as beats.** Lying down, Teeny's analysis lead shows
   the QRS as a ~0.3 mV narrow spike and the T wave as a ~0.7 mV broad
   trough 0.2-0.35 s later. When the QT interval clears NeuroKit's 300 ms
   minimum peak spacing, the T wave is detected as a second beat: early
   (coupling 0.23-0.44x the local RR), "wide" (117-172 ms by the energy
   envelope vs 72 ms), and therefore labelled V. This is the 15:12:50
   couplet and most of the 38 PVCs in 14:25-15:25. Until PR #21 the +/-60 ms
   amplitude filter was rejecting these T-wave detections by accident.

## What was tried and rejected

All measurements on Teeny 2026-08-25 (three hand-counted windows: tachy
08:47:52 +14 s = 36 beats; lying-down 15:22:53 +20 s = 15 beats; quiet
17:06:18 +16 s = 7 beats with a real 4.67 s gap) and MIT-BIH 119 (first
60 s, 65 annotated beats, sensitivity/precision at 150 ms).

| Detector | tachy / lying / quiet | MIT-119 | verdict |
|---|---|---|---|
| `neurokit` (current) | 21 / 12 / 7 | 1.000 / 1.000 | misses tachycardia |
| `pantompkins1985` | 33 / 24 / 7 | 0.985 / 0.970 | double-counts the T waves |
| `hamilton2002` | 32 / 24 / 7 | 0.985 / 0.970 | same |
| `kalidas2017` | 28 / 19 / 7 | 0.985 / 0.970 | both, partly |
| `elgendi2010`, `nabian2018`, `rodrigues2021`, `manikandan2012`, `engzeemod2012` | worse on at least one | - | no |
| `martinez2004`, `christov2004` | 13k / 284k beats in 24 h | - | unusable here |
| `neurokit` parameter grid (36 combos of avgwindow, gradthreshweight, minlenweight, mindelay) | every combo that reaches >= 33 on tachy adds 3,000-17,000 beats elsewhere | precision falls to 0.5-0.75 with shorter windows | not tunable |
| Own Pan-Tompkins decision rules (adaptive thresholds, refractory, slope-based T rule, search-back) on a bandpass/derivative/square/MWI feature | 36-40 / 22-25 / 7-8 | 1.000 / 1.000 | admits the T waves; the sharper 10-35 Hz band that suppresses them drops MIT sensitivity to 0.89 (wide PVCs are low-frequency too) |
| Own gradient detector with a rate-robust threshold (rolling median floor + fraction of rolling peak) | 30-36 / 12-25 / 7-9 | 1.000 / 1.000 | every setting that recovers the tachycardia admits T waves in the quiet window |
| Local (posture-aware) amplitude reference for the phantom filter | +4 beats | - | no effect; dropped |

The frontier is the same in every variant: **any single threshold that
recovers fast beats admits slow-rhythm T waves.** The T wave's smoothed
gradient is ~2x the spike's, so no gradient or slope feature separates
them; what separates them is rate and timing.

## Design

Keep NeuroKit as the primary detector (it is validated, and it handles
slow rhythm, T waves at normal rates, and PVCs). Add two post-passes in
`detection/detect.py`, each a pure function `(cleaned, peaks, sample_rate)
-> peaks` gated on the local rhythm so that it can only act where its
failure mode exists:

```
detect_beats:
  ecg_clean -> ecg_peaks -> _reject_low_amplitude_peaks
            -> fill_fast_gaps -> drop_interpolated_t_waves -> widths -> Beats
```

### `fill_fast_gaps` (rate-gated search-back)

For each consecutive pair of peaks (A, B) with at least 3 prior RRs:

- `local` = median of the previous `LOCAL_RR_BEATS = 8` RRs.
- Act only if `local < FAST_RR_SEC = 0.8` (>= 75 bpm) **and**
  `(B - A) > GAP_FACTOR * local`, `GAP_FACTOR = 1.5`. In slow rhythm a
  1.5x gap is ordinary sinus arrhythmia; in fast rhythm it is a missed beat.
- Candidates are local maxima of the same smoothed-gradient feature
  NeuroKit uses (|gradient| boxcar-smoothed over 0.1 s) inside
  `(A + refractory, B - refractory)`, at least `FILL_FEATURE_FRACTION = 0.35`
  of the median feature of the surrounding detected beats, at least
  `refractory` apart.
- Each candidate's fiducial is the largest |deflection| from the local
  baseline within +/-50 ms; the refractory `FILL_REFRACTORY_SEC = 0.25`
  is enforced **after** placement, against the previous accepted peak and
  B. (The prototype without this produced a 432 bpm "maximum heart rate".)
- T waves cannot be filled in: below 75 bpm the pass is off, and above it
  the T wave falls inside the refractory.

### `drop_interpolated_t_waves`

Walk peaks A, B, C with a running RR history of accepted intervals:

- Slow rhythm: the median of at least three of the last `LOCAL_RR_BEATS`
  accepted RRs is over `SLOW_RR_SEC`, and the gate is mandatory - a
  cold-start fallback to the references alone was tried and dropped real
  beats at tachycardia, because the look-ahead that skips C's T wave
  skips the real next beat there and a double interval matches A -> C.
  `SLOW_RR_SEC` was 0.8 s here and is 0.6 s since the PVC false-positives
  spec (2026-08-26).
- References: the accepted interval before A, and C to the next peak more
  than `T_WAVE_MAX_COUPLING_SEC` after C (so C's own T wave is skipped);
  only intervals over `SLOW_RR_SEC` count. Neighbouring intervals rather
  than the running median because resting sinus arrhythmia moves the RR
  by 30-50% within a few beats - the first version used the median and
  removed one T wave in five.
- Drop B if `(B - A) < T_WAVE_MAX_COUPLING_SEC = 0.45` **and** A -> C is
  within `T_WAVE_RHYTHM_TOLERANCE = 0.25` of a reference: removing B
  leaves the rhythm undisturbed, which a T wave does and a PVC does not
  (a PVC that early either resets the sinus rhythm or is followed by a
  compensatory pause, so A -> C is far from a neighbouring sinus interval).
  A T wave whose A -> C falls in a sinus-arrhythmia swing larger than the
  tolerance survives (one of five in the lying_t fixture); the tolerance
  is not widened past 0.3 because that is where R-on-T PVCs that reset the
  rhythm start to be dropped.
- When B is dropped, `C - A` joins the RR history and the walk resumes at C.

Known cost, accepted: a genuinely interpolated R-on-T PVC at rest (coupling
< 0.45 s in a rhythm slower than 75 bpm with no pause after) is dropped.
That is rare, and a screening tool that reports 38 T waves an hour as PVCs
is worse than one that misses it; the strips still show every beat.

### Thresholds

All new constants live beside the existing ones at the top of `detect.py`
with the reasoning in comments. They are provisional like every threshold
in this tool; the fixtures below are what future retuning is measured
against.

### Data contracts

Unchanged. `detect_beats(samples, sample_rate) -> list[Beat]`; the classifier
and everything downstream see the same `Beat`. Sample-rate-agnostic: every
window is in seconds.

### Canine ground truth in CI

Four 10-20 s three-channel slices of Teeny's 2026-08-25 recording are
committed as `tests/fixtures/teeny_2026-08-25/<name>.npz` (`channels`,
`sample_rate`, `beat_times` hand-counted from zoomed plots) - the tachy,
lying-down, and quiet windows above, plus `lying_t`, the same posture
with the T waves NeuroKit detects as beats. A test runs `detect_beats` on channel
0 of each and asserts sensitivity and precision at 150 ms. This is the
first canine ground truth in the repo; ~100 KB total, no network.

## Results (measured, both passes)

| | before | after |
|---|---|---|
| tachy / lying / quiet fixtures (sensitivity, precision at 150 ms) | 20/23, 12/13, 7/7 | 21/23 (1.00), 12/13 (1.00), 7/7 (1.00) |
| lying_t fixture (13 QRS; T waves detected as beats, P waves detected in place of a 0.03-0.3 mV QRS) | 18 detections | 14 detections, one T wave left; 0.77 / 0.71 capped by the P-wave offset |
| VT runs (4+) | 1 | 0 |
| longest pause | 9.69 s | 6.77 s |
| pauses | 895 | 883 |
| PVCs | 96 | 93 (14:25-15:25: 38 -> 15) |
| couplets / triplets | 2 / 1 | 4 / 2 |
| max HR (5-beat median) | 193 | 235 at 06:43:06 - real: NeuroKit found 2 beats in 8 s of QRS spikes every 0.26 s on all three channels |
| MIT-BIH 119 | 1.000 / 1.000 | 1.000 / 1.000 |
| 2026-08-23 recording | 7 PVCs, 1 couplet, 31 pauses | 7 PVCs, 0 couplets, 29 pauses; max HR 193 -> 240 |

The tachycardia fix does what it was for: the VT run is gone, the longest
pause halves, and beats at 230+ bpm are found where NeuroKit found almost
none. The PVC count barely moves (96 -> 93) because the T-wave rule's
gains in the lying-down hours are offset by the search-back reaching
noisy, fast stretches (couplets 2 -> 4). The remaining ~93 PVCs are not
believed accurate: they cluster in the exercise hour (08:25-09:25, 20;
motion noise at 4 mm/mV) and the lying-down stretch (14:25-16:25, 27),
not where ectopy would. The next piece of work is per-beat noise
rejection (template-correlation SQI, the VT-safe option named in the
2026-08-26 signal-quality spec); it depends on this one, because noise
rules tuned on top of missed beats would be tuned wrong.

Found while building the fixtures: lying down, Ch 3 shows the QRS at
3.5 mV where Ch 1 shows a 0.03-0.3 mV notch, and with a wandering
pacemaker the tall P wave on Ch 1 is what the detector finds on alternate
beats. Multi-lead detection is the structural fix for that posture; it is
out of scope here and the lying_t fixture is its acceptance test in waiting.

## Testing

- TDD, synthetic signals with literal expected peak sets:
  `fill_fast_gaps` fills a 2x gap in a 0.4 s rhythm; leaves a 2x gap in a
  1.2 s rhythm; ignores a candidate under the feature fraction; respects
  the refractory after fiducial placement. `drop_interpolated_t_waves`
  drops the interpolated candidate; keeps one followed by a compensatory
  pause; keeps one that resets the rhythm; is off in fast rhythm.
- Existing MIT-BIH 119 and phantom tests unchanged and green.
- The three Teeny fixtures with sensitivity/precision thresholds.
- Manual (not CI): the before/after table above regenerated on both
  Teeny recordings and pasted into the PR.

## Non-goals

- Multi-lead detection (the lying-down morphology is a lead-axis problem;
  a different analysis channel or a cross-channel feature is a separate
  design).
- Any change to `classify/rules.py`; the "wide" calls were T waves, not a
  width-measurement fault.
- Per-beat noise rejection (next spec).
