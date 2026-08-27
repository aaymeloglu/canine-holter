# PVC false positives: QRS width in noise, the width criterion, and the T-wave floor

**Date:** 2026-08-26
**Status:** approved

## Problem

After the detector work (#22), Teeny's 2026-08-25 report still counted 93
PVCs. Every one of the 93 was reviewed by eye on a 2 s three-channel strip
with its RR ratio, width ratio, and local noise printed beside it (the
contact sheets are reproducible from the scratch script described under
Method). They are:

| class | count | why the classifier called it V |
|---|---|---|
| real PVCs | 5 | 106-156 ms wide against a 61-72 ms baseline (ratio 1.65-2.27), early (RR 0.32-0.69x), morphologically different, mostly large |
| normal beats | 62 | early by resting sinus arrhythmia or a tachycardia acceleration (RR 0.5-0.85x), and "wide" by 3-5 samples: 78-89 ms vs 61 ms, ratio 1.27-1.45 |
| T waves the T-rule missed | 18 | clusters in a 0.75-0.85 s rhythm, at the rule's 0.8 s slow-rhythm floor, so no neighbouring interval qualifies as a reference |
| noise with no QRS underneath | 8 | hash in the exercise hour |

The planned approach - a per-beat template-correlation SQI - was tested
and rejected: real PVCs correlate 0.75-0.81 with the local normal-beat
template over a 60 ms window (best lag +/-17 ms), the same as normal beats
(0.76-0.98), because at QRS scale a wide monophasic deflection and a narrow
spike are both "one bump". Local high-frequency noise level (0.01-0.06 of
QRS amplitude) does not separate them either. Width does: no real PVC is
under 1.65x, no normal beat over 1.45x - once width is measured correctly.

Two things make width wrong today:

1. **Noise holds the energy envelope up.** `_qrs_width` walks out from the
   R peak until the derivative-energy envelope drops below 10% of its
   value at the peak. Hash noise keeps the envelope above that, and the
   crossing lands on the noise, not the QRS edge.
2. **The 1.25 ratio is three samples at 180 Hz.** A normal Doberman QRS
   on this recorder is 11 samples (61 ms); 1.25x is 14 samples (78 ms).
   One sample of jitter on each edge and a normal beat is "wide". The
   ratio was chosen on 360 Hz MIT-BIH data where it is seven samples.

## Method

Ground truth is the author's read of every flagged beat on both real
recordings, not a cardiologist's; the width thresholds below have a
physical justification (sample quantization; the measured 1.45 / 1.65
separation) that does not depend on that read, and the spec says which
numbers do.

Every variant was scored by which of the 93 it keeps, by class, plus the
2026-08-23 recording's 7 flagged beats (5 normal or noise, 2 real: a pair
at 17:35:35 and 17:35:38, both 4.1 mV against 1.7 mV neighbours, the first
100 ms / 1.49x, the second 106 ms / 1.65x).

| variant | 2026-08-25: PVCs (real / T / noise / normal) | 2026-08-23 |
|---|---|---|
| today | 93 (5 / 18 / 8 / 62) | 7 (2 real) |
| T floor 0.6 only | 91 (5 / 16 / 8 / 62) | - |
| + noise-floor width, k = 2 / 3 / 4 / 6 | 66 / 61 / 53 / 53; k = 6 loses a real one | - |
| + width ratio 1.5 | 19 (5 / 8 / 3 / 3) | 1 - loses 17:35:35 |
| + width ratio 1.25 and margin >= 30 ms | **20 (5 / 9 / 3 / 3)** | **2 (both real)** |
| + prematurity 0.75 instead of 0.85 | no further change once width is right | - |
| template-correlation SQI | no separation | - |

## Design

Three changes, each behind a named constant.

### 1. Noise-floor QRS width - `detection/detect.py::_qrs_width`

The crossing threshold becomes
`max(QRS_WIDTH_THRESHOLD_FRACTION * envelope[peak], QRS_NOISE_FLOOR_FACTOR * median(envelope over +/-1 s))`
with `QRS_NOISE_FLOOR_FACTOR = 4`. The median over +/-1 s is the noise
floor: QRS complexes occupy well under half of any second. If the
threshold reaches the peak itself the beat is buried in noise and the
width is `None`, as for any beat whose width cannot be measured (the
classifier then labels it `U`). The signature and the search window are
unchanged. k = 4 keeps all five real PVCs at their measured widths; k = 6
narrows one below the criterion.

### 2. Width criterion with an absolute margin - `classify/rules.py`

A beat is wide when `qrs > QRS_WIDTH_RATIO * baseline` **and**
`qrs - baseline >= QRS_WIDTH_MARGIN_SEC`, with `QRS_WIDTH_RATIO = 1.25`
unchanged and `QRS_WIDTH_MARGIN_SEC = 0.030`. Thirty milliseconds is more
than five samples at 180 Hz and about the smallest difference a reader
would call "wider" on paper (one small square is 40 ms). Against a
61-72 ms baseline it requires 91-102 ms; the real PVCs measure 100-156 ms
and the false ones 78-89. On MIT-BIH 119 PVCs are 80+ ms wider than normal
beats, unaffected. The ratio stays so that a large baseline (a dog with a
genuinely wide QRS) still needs a proportionate difference.

### 3. T-rule floor - `detection/detect.py::SLOW_RR_SEC`

`SLOW_RR_SEC` 0.8 -> 0.6 (under 100 bpm rather than under 75). At 75-100
bpm the T wave still falls 0.25-0.35 s after the QRS, past NeuroKit's
300 ms minimum spacing, and the rule's other guards are unchanged:
tachycardia is excluded by the median gate (a 0.4 s rhythm is never
"slow"), a PVC with a compensatory pause fails the A->C match, and a
PVC that resets the rhythm fails it by 0.3 s / RR. The search-back's
`FAST_RR_SEC` stays at 0.8; the two constants no longer share a value,
which is fine - one asks "could a beat be missing here", the other "could
this be a T wave", and the answers overlap between 75 and 100 bpm.

### Data contracts

Unchanged. `Beat.qrs_duration` still means the energy-envelope width in
seconds; `classify_beats` still labels `N`/`V`/`U` from RR and width only.

## Results (measured)

2026-08-25: 93 -> 20 PVCs (5 real, 9 residual T waves in sinus-arrhythmia
swings the T-rule's tolerance cannot follow, 3 noise, 3 normal): 01:01:54,
06:08:05, 08:45:58, 11:36:41, 16:15:51 are the real ones. Couplets 4 -> 1,
triplets 2 -> 0, 60 beats now `U` (width unmeasurable in noise).
2026-08-23: 7 -> 2, both real (17:35:35, 17:35:38).

Found while implementing: the T-wave rule's cold-start fallback (references
alone when fewer than three intervals are known) dropped real beats at
tachycardia once the floor moved to 0.6 s - the look-ahead that skips C's
T wave skips the real next beat in fast rhythm. The slow-rhythm gate is now
mandatory; the `lying_t` fixture's precision floor is 0.65 because its
first T wave starts cold. The residual T waves are the largest remaining class and are a
single-lead problem (the lying-down posture): multi-lead detection remains
the structural fix, as noted in the detector spec.

## Testing

- `_qrs_width`: a synthetic beat with white noise added measures the
  same width as without (within one sample) under the noise floor, and
  wider without it; a beat buried in noise returns `None`.
- `classify_beats`: a beat 1.3x wider but only 15 ms wider than baseline
  is `N`; 1.3x and 40 ms wider is `V`; the existing ratio tests keep
  passing with widths that satisfy the margin.
- `drop_interpolated_t_waves`: a T wave in a 0.7 s rhythm is dropped; the
  existing fast-rhythm and compensatory-pause tests keep passing.
- `tests/fixtures/teeny_2026-08-25/`: unchanged thresholds; `lying_t`
  may improve (its residual T wave is in a 0.75 s rhythm).
- MIT-BIH 119 validation unchanged.
- Manual: both recordings regenerated, PVC list compared with the contact
  sheets; numbers pasted into the PR.

## Non-goals

- Per-beat noise SQI that suppresses V calls in noisy stretches: it would
  hide real PVCs during exercise, which is when a Doberman throws them.
- A morphology model: no labelled canine data, and the correlation
  feature already failed.
- Changing `PREMATURITY_RATIO`: it adds nothing once width is right, and
  it is the threshold a cardiologist's read should calibrate.
