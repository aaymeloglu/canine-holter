# Beats with no local maximum: the negative-QRS fiducial

**Date:** 2026-09-02
**Status:** approved design (stage 2 of
`2026-09-02-cardiologist-report-parity-design.md`).

## Problem

The cardiologist's HE/LX report gives the 2026-08-27 recording a longest
RR of 4.333 s at 03:15; ours said 4.73 s at 06:35:47, with 4.52 s at
06:35:01 close behind. Both of our gaps contain a beat that is plainly
visible on all three leads. The 2026-08-25 recording has the same fault:
its 5.21 s and 4.89 s "pauses" at 03:07 are missed beats, and the true
longest RR is the 4.67 s sinus pause at 17:06:30 that the `quiet` fixture
already documents.

The mechanism is in NeuroKit's default detector. It finds QRS complexes as
bursts of smoothed absolute gradient above 1.5x a 0.75 s running average,
then, inside each burst, keeps the most prominent *local maximum* of the
signal. On leads 2 and 3 of the DR400 patch a sleeping dog's QRS is a tiny
r and a deep S; the burst covers the downstroke and the S, and for some
beats (the ventricular escape beats of 03:00-06:00 especially) there is no
local maximum inside it at all. NeuroKit then returns nothing for that
burst, lead 1 is the only lead with a peak, and two-lead agreement drops
the beat. Negating leads 2 and 3 finds every one of these beats, which is
how the cause was confirmed.

## Approaches tried and rejected

- **Rectify the signal** (`nk.ecg_peaks(abs(cleaned))`): finds the 06:35
  beats but changes the burst statistics; NeuroKit finds 15 % fewer bursts
  per lead, a 4.95 s false gap appears in the 12:06 sinus tachycardia, and
  three new false PVCs appear. Rejected.
- **Detect both polarities and merge within 150 ms**: finds the beats, but
  the negated run also lands on T waves 170-250 ms after the R, two leads
  agree on them, and the recording gains RRs down to 0.083 s and five
  false PVCs. Rejected.
- **Largest absolute deflection in every burst** (replacing NeuroKit's
  rule outright): finds the beats and leaves the PVC count close, but
  moves every fiducial to wherever the deflection is largest, and the
  width walk that starts there measures the 16:07:03 (08-27) and 06:08:05
  (08-25) PVCs at 78 ms instead of 94-100 ms, under the 30 ms margin, so
  two real PVCs are lost. Anchoring the width on the envelope peak instead
  makes that worse (every wide beat narrows). Rejected: the width
  measurement depends on the fiducial sitting on the apex, and that
  dependency is not this change's to fix.

## Design

`detection/detect.py` gets `_qrs_fiducials(cleaned, sample_rate)`, a copy
of NeuroKit's burst logic with the same parameters (0.1 s smoothing,
0.75 s average, 1.5x threshold, 0.4x minimum burst length, 0.3 s minimum
delay) and the same most-prominent-local-maximum rule, plus one fallback:
a burst that has no local maximum takes the sample of largest absolute
deflection from the median of the 0.2 s before the burst. Every beat
NeuroKit finds today keeps exactly its fiducial, so widths, agreement, and
the PVC calls are unchanged; only bursts that were dropped gain a beat.
`_lead_peaks` calls it in place of `nk.ecg_peaks`; `nk.ecg_clean` and the
three correction passes are untouched.

The copied logic is ~25 lines under NeuroKit2's MIT licence; the function's
docstring says where it comes from so a future NeuroKit change is easy to
compare against.

## Acceptance

Measured on the prototype before implementation:

| | 2026-08-27 before | after | 2026-08-25 before | after |
|---|---|---|---|---|
| Beats | 84,562 | 84,602 | 86,246 | 86,261 |
| PVC flags | 7 (same 7 times) | 7 | 7 (same 7 times) | 7 |
| Longest RR | 4.73 s at 06:35:47 | 4.34 s at 03:15:46 | 5.21 s at 03:07:36 | 4.67 s at 17:06:30 |
| Pauses >= 2.5 s | 1772 | 1768 | 878 | 875 |
| Shortest RR | 0.222 s | 0.222 s | 0.206 s | 0.206 s |

Two new hand-counted fixtures, `escape_a` (06:34:54, 20 s) and `escape_b`
(06:35:37, 20 s) from the 08-27 recording, each hold one of the missed
beats among ordinary sleeping-rhythm beats; all-lead detection must score
1.00 sensitivity and precision on both. A unit test shows that a train of
V-shaped negative complexes, which NeuroKit misses entirely, is found at
its troughs, and that a positive train yields exactly NeuroKit's peaks.

## Not changed

QRS width measurement still walks from the fiducial; its sensitivity to
where the fiducial lands is noted above for a later change. Ventricular
escape beats are detected after this change but still labeled `N` (stage
3).
