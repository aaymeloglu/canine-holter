# Beat shape as a ventricular criterion

**Date:** 2026-09-02
**Status:** evidence gathered; not approved for implementation. This
records why a beat-shape rule is the right next feature and why it cannot
be built yet.

## Problem

Stage 3 (`2026-09-02-ventricular-escape-beats-design.md`) labels a beat a
ventricular escape beat when it is wide and late. On the 2026-08-27
recording that finds 8; the cardiologist's HE/LX report counts 32, and
their strips name beats at 04:30:09, 04:30:26 and 05:53:37 (their clock;
ours runs 49 s ahead) that we call normal. Those three beats measure
67-72 ms, the same as the sleeping-rhythm beats around them, so no width
threshold will find them. What distinguishes them on our leads is shape:
no P wave in front, and a deep negative deflection after the spike that
the normal beats do not have. HE/LX calls ventricular beats by template
("they differ significantly from the normal; they are not necessarily
premature"). Width is our proxy for shape, and it is a weak one.

## The measurement

For each beat, the three leads' cleaned signal in a window around the
fiducial, concatenated, baseline-removed, correlated (Pearson) against
the median of the recent normal beats' windows. Causal, like the
classifier's baselines; a `V` or `E` never enters the template.

On 2026-08-27, with a -60..+120 ms window and the last 8 normal beats as
template, the scores separate cleanly where it matters:

| Beats | Correlation |
|---|---|
| Our 8 escape beats | 0.10-0.44 |
| The 4 real PVCs | 0.44-0.62 |
| The 2 false PVCs (a T wave, a noise burst) | below 0 |
| The escape beats at 04:31:00 and 04:31:17 we miss | 0.71 |
| The one at 05:54:28 | 0.87 |
| Normal beats, 5th percentile | 0.92 |
| Normal beats, 1st percentile | 0.72 |

The score also shows something the width rule hides: between 06:15 and
06:24 there is a run of beats every ~2.7 s scoring 0.1-0.3, which is a
ventricular escape rhythm at ~22 bpm and is what the report's "slowest
heart rate 21 bpm at 06:23:38" is. HE/LX counts only 5 ventricular beats
in that hour, so either it calls that stretch something else (its
comments mention junctional escape beats and second-degree AV block) or
it misses it; without a beat-level read we cannot tell.

## The rule, and why it fails

Three versions of "a beat is `V` when premature and (wide or different),
`E` when late and (wide or different)" were simulated causally on both
real recordings. Ground truth for `V`: about 9 on 08-27 (the
cardiologist's early beats and pairs) and 5 on 08-25 (our reviewed
list). The stage-3 rule, with no shape term, gives 7 and 7.

| Variant | 08-27 V / E | 08-25 V / E |
|---|---|---|
| Window -60..+120 ms, template of 8, threshold 0.5 | 395 / 80 | 103 / 83 |
| QRS-only window -40..+60 ms, template of 32, alignment +/-11 ms, noise gate, threshold 0.5 | 169 / 41 | 30 / 34 |
| Window -50..+70 ms, template of 32, alignment +/-44 ms, and `V` also premature against the previous RR, threshold 0.5 | 59 / 30 | 27 / 27 |

The last is 6-7x too many PVCs on both recordings, and raising the
threshold makes it worse (80 / 47 at 0.6, 102 / 73 at 0.7 on 08-27).
The false calls are of four kinds, each seen on a strip:

1. Plain sinus rhythm at 11:31-11:52 on 08-27, beats that look identical
   to the eye, scoring under 0.6. Beat-to-beat amplitude and S-wave depth
   vary with breathing, and with sinus arrhythmia many of these beats are
   also "premature" by either definition. 24 of the 59 false PVCs are here.
2. Sinus tachycardia onset (08-25 06:30): every beat is premature against
   a template built while asleep, and the QRS shape changes with rate.
3. Motion-noise bursts (08-25 12:25): the burst is flagged, and the beats
   after it score low against the template.
4. The 06:15-06:24 stretch on 08-27, where the rule calls 20-37 escape
   beats against the cardiologist's 5 for the hour, and we do not know
   who is right.

Even with the wide alignment search, 5 % of normal beats score under 0.93
and 1 % under 0.72-0.78 on both recordings: the tail of normal shape
variation overlaps the ventricular beats. A rolling correlation with a
fixed threshold cannot be made both sensitive to the 04:31 beats (0.71)
and quiet on sinus rhythm (1st percentile 0.72).

## What would unblock it

Beat-level ground truth. HE/LX holds a label for every beat of the 08-27
recording; the practice can export its beat table, or failing that, a
cardiologist's beat-by-beat read of these stretches would settle each
failure above:

- 08-27 06:15-06:25: ventricular escape rhythm, junctional, or sinus with
  AV block?
- 08-27 03:00-04:00: where their 13 late ventricular beats are.
- 08-27 11:31-11:52: any ventricular beats at all (their 11:51:05 VPB is
  the one known).
- 08-27 04:30:50-04:31:20 and 05:54:20-05:54:35: the named escape beats.
- 08-25 06:29-06:31 and 12:24-12:26.

With that, the shape measure can be judged per beat instead of per
recording, the confounds above can be modelled (respiration, rate, noise)
or the rule restricted to where it is reliable (asleep, quiet baseline),
and a threshold chosen with a measured false-call rate. Until then the
report keeps the width rule and says so.

## Rejected for now

- Any shape rule at a fixed correlation threshold, for the numbers above.
- Restricting the shape rule to late beats only (`E` via shape, `V`
  unchanged): the 06:15-06:24 question is exactly the one it would decide,
  and it is the one we cannot check.
