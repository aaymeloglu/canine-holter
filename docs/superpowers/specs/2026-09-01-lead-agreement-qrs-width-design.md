# Beats and QRS width by lead agreement

**Date:** 2026-09-01
**Status:** approved design, revised during implementation (see "What
changed while building").

## Problem

A QRS is on every lead at once, and a PVC is a beat that starts in the
ventricle, so its shape differs from the sinus beat on every lead at
once. The detector measured one lead, and Teeny's leads take turns being
unreliable as she changes posture:

- On the 2026-08-27 DR400 recording analyzed on channel 1, 36 of 47
  "PVCs" sit in 23:55-00:13. Asleep, with sinus arrhythmia alternating
  ~0.9 s and ~1.5 s intervals, channel 1 shows each QRS as a small
  fractured wiggle and the T wave 0.3 s later as the larger deflection;
  the detector takes the T wave for a beat, early by the rolling baseline
  and wide (a T wave is broad on every lead). Channels 0 and 2 show the
  QRS as identical narrow 2 mV complexes and detect no beat at the T.
- Channel 0, the lead the app analyzed, is the weak lead from 14:00 to
  20:00 on the same recording (0.55 mV median peak-to-peak vs 3.2 mV on
  channel 1) and the strong lead at midnight. Choosing one channel only
  moves the errors: the channel-0 report has 10 PVCs and a false 111 s
  pause, the channel-1 report 47 PVCs and no such pause.
- The `lying_t` fixture from 2026-08-25 (0.77 sensitivity, 0.67
  precision on channel 0: P waves and T waves detected, the notch-sized
  QRS missed) was the pending acceptance case for multi-lead detection.

The six beats on the two real recordings that a careful eye calls PVCs
(2026-08-25 at 01:01:54, 06:08:05, 08:45:58, 11:36:41, 16:15:51;
2026-08-27 at 16:07:03) are all bigger, wider, and differently shaped on
all three leads simultaneously.

## Goals

1. A beat exists only where at least two of the three leads detect it,
   and is wide only when at least two leads measure it wide, so one
   unreliable lead can neither invent a beat or a PVC nor hide one.
2. Keep every one of the six real PVCs flagged.
3. Keep the code small: no new classifier state, no change to `Beat`,
   no channel option in the CLI or GUI.

## Non-goals

- False PVCs that every lead agrees on: motion-noise bursts, and a T wave
  every lead detects as a beat. Agreement cannot remove what all leads
  see. Single-lead inputs (WFDB records) are unchanged.
- Quality gating still judges `samples` (channel 0).

## Design

### Detection (`detection/detect.py`)

`detect_beats(leads, sample_rate)` takes one lead (1-D) or all leads
(`(n_leads, n_samples)`). Each lead runs the existing single-lead
pipeline on its own: NeuroKit2 R-peaks, amplitude rejection, fast-rhythm
search-back, T-wave rejection. Then `_agree` groups the leads' peaks into
clusters no wider than `AGREEMENT_TOLERANCE_SEC = 0.15` and keeps a
cluster as a beat when at least
`MIN_AGREEING_LEADS = 2` leads are in it (or every lead, when fewer are
given). The beat's time is the median of the agreeing leads' peaks.

The tolerance is half the distance to the nearest distinct event: a T
wave taken for a beat sits 250-350 ms after its QRS and the shortest RR
seen is 255 ms at 235 bpm. It has to be that wide because the same QRS's
fiducial does not land at the same time on every lead. On 2026-08-27,
channel 1's and channel 2's peaks fall within 22 ms of channel 0's for
90 percent of beats, but where channel 1 renders the QRS as a fractured
wiggle (asleep, around midnight) its detector settles 120-140 ms early,
on the onset or the P wave. At 100 ms those beats had only channel 0,
channel 2 having missed them, and 23:59:45-23:59:53 became an 8.3 s
pause. The 100-150 ms band holds 1.9 percent of channel 1's detections
on that recording, all of this kind.

QRS width is measured on every lead **at that lead's own peak**, or at
the consensus position for a lead that missed the beat, and the beat's
`qrs_duration` is the median of the per-lead widths, ignoring leads that
return none. With three leads the median exceeds a threshold exactly
when at least two leads do, so the classifier's existing width test
becomes the two-of-three rule with no change to `classify_beats`, `Beat`,
or the report. Measuring at each lead's own peak matters: the 16:07:03
PVC peaks 50 ms later on channels 1 and 2 than on channel 0, and
measured at the consensus position it read 72-94 ms instead of
106-128 ms and was lost.

### Pipeline and contracts

`run_analysis` passes `rec.channels` when present, else `rec.samples`.
`Recording.samples` is the lead quality gating judges and, for a
single-lead input, the lead beats are detected on; `channels` is every
lead, detected on together and drawn by the report. CLAUDE.md's data
contract says the same.

### What changed while building

The approved design measured width on every lead at the analysis lead's
R-peak and left detection single-lead. Two measurements overturned it:

- The midnight cluster is T-wave detections, not width: their widths are
  90-156 ms on all three leads. Width agreement alone left the channel-1
  midnight flags in place and, because the shared baseline became the
  narrow channel-0/2 width, raised the channel-1 count from 47 to 130.
- The 16:07:03 PVC was lost when widths were measured at one shared
  position (above).

Detection-level agreement is the same principle applied one stage
earlier, and it also resolves the lying-down misses that channel
selection was meant to address.

### Rejected alternatives

- **Width from the lead with the largest QRS for that beat.** Amplitude is
  the property that misleads us; a strongest-lead pick can land on a
  fractured lead.
- **Narrowest width across leads.** A real PVC can read narrow on a lead
  where it is nearly isoelectric.
- **Per-lead baselines in the classifier.** Same decision as the median
  against the shared baseline, with three times the state.
- **A channel option in the CLI/GUI.** The best lead changes with
  posture within one recording.

## Testing

- Unit (`tests/detection/test_detect.py`), synthetic three-lead Gaussian
  pulses at 500 Hz: the width is the median of the leads (wide on one,
  narrow on two, and the reverse); a beat on one lead only is not a beat;
  a beat on two of three is, with the width from both; a lead whose QRS
  peaks 40 ms later is measured at its own peak and the beat time is the
  median; peaks 300 ms apart are different beats; two leads must both
  agree.
- Fixtures cut by `scripts/extract_teeny_fixtures.py` (three channels,
  `beat_times` for detection windows, `pvc_times` for classification
  windows). `tests/detection/test_teeny_fixtures.py` keeps the single-lead
  thresholds and adds an all-leads variant: tachy >= 0.95/1.00, lying,
  lying_t, quiet 1.00/1.00. `tests/classify/test_teeny_fixtures.py` runs
  detection on all leads and classification and requires the V beats to
  match `pvc_times` within 150 ms, exactly: `teeny_2026-08-27/midnight`
  (00:03:00 +40 s, none), `teeny_2026-08-27/pvc` (16:06:55 +20 s, one at
  8.17 s), `teeny_2026-08-25/pvc_run_end` (11:36:33 +20 s, one at 8.27 s).
- The pipeline's synthetic native flash carries its spike train on two
  channels, since a beat needs two leads.

## Acceptance

Filled in from the full recordings after implementation.
