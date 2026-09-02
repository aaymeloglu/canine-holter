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
delay) and the same most-prominent-local-maximum rule. It returns two
arrays: the *resolved* fiducials, which are exactly what `nk.ecg_peaks`
returns, and a *fallback* fiducial for every burst that has no local
maximum, placed by the existing `_fiducial` rule at the largest absolute
deflection within 50 ms of the burst's steepest sample (the T wave, when
the burst runs on into it, is not the steepest part).

A fallback is not a beat on its own. `_lead_peaks` runs the three
single-lead correction passes on the resolved fiducials only, so
single-lead detection is unchanged to the sample, then adds the
fallbacks that survive two filters: the amplitude rule judged against
the resolved peaks' median, and a proximity rule that drops a fallback
within 0.45 s (`T_WAVE_MAX_COUPLING_SEC`) after a resolved peak, which is
that beat's T wave, or within 0.3 s before one, which is part of that
beat. `_agree` then requires a cluster to contain at least one resolved
fiducial as well as two leads. So a lead that shows a QRS as a bare
trough can vote for a beat another lead saw, and a lead's negative T
waves can never make one.

### What was tried on the way

- Letting the fallback stand alone: the `lying` fixture's single lead
  gained a detection on a T trough whose QRS that lead never resolves,
  and the `tachy` fixture lost three beats because fallbacks inside the
  correction passes stopped the fast-gap search-back from filling real
  gaps. Both fixed by the corroborate-only rule and by keeping fallbacks
  out of the passes.
- Sharing NeuroKit's 0.3 s minimum delay between resolved and fallback
  fiducials: a fallback T trough then blocked the next resolved QRS at
  fast rates. The delay for resolved fiducials is now against resolved
  fiducials only, so they match NeuroKit exactly.

The copied logic is ~25 lines under NeuroKit2's MIT licence; the function's
docstring says where it comes from so a future NeuroKit change is easy to
compare against.

## Acceptance

Measured with the implemented detector:

| | 2026-08-27 before | after | 2026-08-25 before | after |
|---|---|---|---|---|
| Beats | 84,562 | 84,572 | 86,246 | 86,253 |
| PVC flags | 7 | the same 7 | 7 | the same 7 |
| Longest RR | 4.73 s at 06:35:47 | 4.34 s at 03:15:46 | 5.21 s at 03:07:36 | 4.67 s at 17:06:30 |
| Pauses >= 2.5 s | 1772 | 1768 | 878 | 876 |
| Shortest RR | 0.228 s | 0.228 s | 0.206 s | 0.206 s |

Two new hand-counted fixtures, `escape_a` (06:34:53, 21 s) and `escape_b`
(06:35:36, 21 s) from the 08-27 recording, each hold one of the missed
beats among ordinary sleeping-rhythm beats; all-lead detection scores
1.00 sensitivity and precision on both, and every earlier fixture keeps
its score. Unit tests show that a positive train resolves exactly
NeuroKit's peaks with no fallbacks, that a train of wide negative
complexes NeuroKit drops yields fallbacks on the troughs, that such a lead
alone yields no beat NeuroKit would not, and that beside a positive lead
it corroborates every beat.

## Not changed

QRS width measurement still walks from the fiducial; its sensitivity to
where the fiducial lands is noted above for a later change. Ventricular
escape beats are detected after this change but still labeled `N` (stage
3).
