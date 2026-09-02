# QRS width by lead agreement

**Date:** 2026-09-01
**Status:** approved design.

## Problem

A PVC is a beat that starts in the ventricle, so its shape differs from
the sinus beat on every lead at once. Our classifier measures QRS width
on one lead, and Teeny's leads take turns being unreliable as she changes
posture:

- On the 2026-08-27 DR400 recording analyzed on channel 1, 36 of 47
  "PVCs" sit in 23:55-00:13. There channel 1 shows each QRS as a small
  fractured biphasic wiggle that the energy envelope measures at 90-130 ms
  against a 60 ms baseline, while channels 0 and 2 show the same beats as
  identical, narrow, 2 mV complexes. "Premature" is her sleeping sinus
  arrhythmia (alternating ~0.9 s and ~1.5 s intervals). Early plus wide on
  one lead is a PVC by the current rule; on the other two leads the beat
  is plainly normal.
- Channel 0, the lead the app analyzes, is the weak lead from 14:00 to
  20:00 on the same recording (0.55 mV median peak-to-peak vs 3.2 mV on
  channel 1) and the strong lead at midnight. Choosing one channel for
  the recording only moves the errors: the channel-0 report has 10 PVCs
  and a false 111 s pause, the channel-1 report 47 PVCs and no such pause.

The six beats on the two real recordings that a careful eye calls PVCs
(2026-08-25 at 01:01:54, 06:08:05, 08:45:58, 11:36:41, 16:15:51;
2026-08-27 at 16:07:03) are all bigger, wider, and differently shaped on
all three leads simultaneously.

## Goals

1. A beat is wide only when it is wide on at least two of the three
   leads, so one unreliable lead can neither create a PVC nor hide one.
2. Keep every one of the six real PVCs above flagged.
3. Keep the code small: no new classifier state, no change to `Beat`.

## Non-goals

- The other false PVCs on channel 0 of both recordings: motion-noise
  bursts (08:30, 11:36:33, 14:03, 19:53 on 08-25; 12:05, 14:15, 15:24,
  19:01, 07:13 on 08-27) and T waves detected as beats in the lying
  posture (12:17, 13:01, 14:50, 15:12-15:15, 19:17 on 08-25). Those are
  wide on every lead because the "beat" is not a QRS on any lead; they are
  detection problems, and this rule does not claim to fix them.
- Multi-lead R-peak detection (the lying-down phantom pauses). The beat
  times still come from the analysis lead.
- Single-lead inputs (WFDB records) are unchanged.

## Design

### Detection (`detection/detect.py`)

`detect_beats(samples, sample_rate, channels=None)`. R-peak detection,
amplitude rejection, gap fill, and T-wave rejection run on `samples` as
before. QRS width is then measured on every row of `channels` and each
beat's `qrs_duration` is the **median of the per-lead widths**, ignoring
leads that return no width. With three leads the median exceeds a
threshold exactly when at least two leads do, so the classifier's
existing width test becomes the two-of-three agreement rule with no
change to `classify_beats`, `Beat`, or the report. With `channels=None`
the width comes from `samples` alone, as today.

On each lead the width is measured at that lead's own energy-envelope
peak within `FIDUCIAL_HALF_SEC` (50 ms) of the analysis lead's R-peak,
because the steepest slope of the same QRS falls a few samples apart from
lead to lead and `_qrs_width` needs the envelope's local maximum.

The recording-wide baseline in the classifier is then the median of
per-beat medians. A lead whose normal QRS measures systematically wider
(the fractured channel-1 morphology) is outvoted beat by beat, which is
what a per-lead baseline would have achieved with three times the state.

### Rejected alternatives

- **Width from the lead with the largest QRS for that beat.** Amplitude is
  the property that misleads us: channel 1 at midnight is not small, it
  is fractured. A strongest-lead pick can land on it.
- **Narrowest width across leads.** A real PVC can read narrow on a lead
  where it is nearly isoelectric and the envelope has nothing to measure.
- **Per-lead baselines in the classifier.** Same decision as the median
  against the shared baseline, but `Beat` would carry three widths and the
  classifier three baselines.

### Pipeline and contracts

`run_analysis` passes `rec.channels` to `detect_beats`. `Recording`'s
docstring and CLAUDE.md's data contract change from "only the report
reads channels" to: R-peak detection, quality, and classification consume
`samples`; QRS width is measured on every channel; strips display them.

## Testing

- Unit (`tests/detection/test_detect.py`), synthetic three-lead signals:
  a beat wide on the analysis lead and narrow on the other two gets the
  narrow width; a lead with no measurable width is ignored; all leads
  unmeasurable gives `None`; a lead whose QRS peaks 30 ms after the
  analysis lead's R-peak is measured at its own peak; `channels=None`
  reproduces the single-lead width.
- Fixtures (`tests/fixtures/teeny_2026-08-27/`, cut by
  `scripts/extract_teeny_fixtures.py`, three channels, with `pvc_times`):
  `midnight` (00:03:00 +40 s, no PVCs; analyzed on channel 1, the old
  path flags several and the agreement path flags none) and `pvc`
  (16:07:03 -8 s, +12 s, one PVC). From 2026-08-25, `pvc_run_end`
  (11:36:41 -8 s, +12 s, one PVC after a run of small beats). The
  classification test runs detection with channels and asserts the V
  beats match `pvc_times` within 150 ms, exactly.
- Acceptance on the full recordings, channel 0 and channel 1: all six
  real PVCs remain flagged; the channel-1 midnight hour drops from 36 to
  0; every other change in the PVC lists is inspected on three-lead
  strips and recorded in this spec's acceptance table.

## Acceptance

Filled in after implementation.
