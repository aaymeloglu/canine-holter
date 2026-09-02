# Ventricular escape beats

**Date:** 2026-09-02
**Status:** approved design (stage 3 of
`2026-09-02-cardiologist-report-parity-design.md`).

## Problem

The cardiologist's HE/LX report for 2026-08-27 counts 42 ventricular
beats, "Early/Late 5/32". The 32 late ones are ventricular escape beats:
wide, ventricular-shaped beats that arrive *after* a gap longer than the
sinus cycle, because a ventricular focus fired when the sinus node and AV
junction did not. Their strips show them at 02:14:55, 03:13:36, 04:30:09,
04:30:26, 05:53:37 and 21:28:46 (their clock, 49 s behind ours), and the
hourly table puts 31 of the 32 between 02:00 and 07:00.

Our classifier labels a beat `V` only when it is premature *and* wide, so
it reports none of these, and until stage 2 the detector missed some of
them outright. A report that says "7 PVCs" and nothing else about beats
that a cardiologist would list as ventricular is incomplete.

## Rule

A beat is a ventricular escape beat, label `E`, when it is wide by the
PVC rule (QRS over `QRS_WIDTH_RATIO` times the local baseline and at least
`QRS_WIDTH_MARGIN_SEC` wider) and its RR is at least `ESCAPE_RR_RATIO =
1.5` times the local RR baseline. The same causal baseline of the last
eight `N` beats serves both rules, and an `E` beat feeds it no more than a
`V` does: its long RR and wide QRS are not the rhythm to compare the next
beat with.

Why 1.5. Measured on the stage-2 beats with the width rule as it stands:

| RR over baseline | 08-27 candidates | 08-25 candidates |
|---|---|---|
| >= 1.0x (any wide, non-premature beat) | 13 | 18 |
| >= 1.25x | 10 | 12 |
| >= 1.5x | 8 | 10 |
| >= 2.0x | 3 | 5 |

At 1.5x the 08-27 set is 21:29:37, 02:15:45, 03:14:27 and five beats in
the 06:00 hour, every one after an RR of 2.1-3.5 s while asleep. The
02:15:45 and 03:14:27 beats are the ones on the cardiologist's strips, and
five in the 06:00 hour is exactly their count for that hour. The beats
that 1.5x excludes are daytime candidates after RRs of 0.6-1.0 s in
motion noise (18:22:48 twice, 17:35:19), which the resting sinus
arrhythmia of this dog produces at 1.1-1.4x all day. One 06:31:23 beat
after a 2.52 s RR is lost at 1.5x because the sinus itself had slowed to
1.8 s; an absolute floor would catch it but adds a second provisional
number, and this stage keeps one.

The cardiologist's 32 against our 8 is the width measurement, not the
rule: the 03:00-06:00 hours hold 26 of their 32, and the wide-by-width
test finds one there. The report must not read as if 8 were the count of
escape beats in the recording; the row's caption says the rule.

## Report

- `Beat.label` gains `E`. `types.py` and CLAUDE.md say so.
- `ArrhythmiaSummary.escape_beats: list[float]` (times) and `HourRow.escapes`.
  `pvc_runs` still groups consecutive `V` beats only: an `E` ends a run,
  and a run of `E` beats (an idioventricular rhythm) is not counted or
  named in this stage.
- Ventricular ectopy panel: a row `Escape beats` with the count, reference
  `wide, RR >= 1.5x local`, uncoloured. The ESVC bands count premature
  beats, so escape beats stay out of `PVCs` and `PVCs per 24 h`.
- Hourly table: an `Escapes` column after `Runs (3+)`.
- Timeline: an `Escape` lane between `PVC` and `Pause`.
- Strips: a section `Ventricular escape beats` after the isolated PVCs
  and before the hourly strips, capped like the PVC sections. Caption:
  the RR and QRS behind the label, as the PVC captions do, and a
  significance line that an escape beat is the ventricle stepping in
  after a long gap, so the gap is the finding (see the pauses), and that
  several in a row at a slow rate are an idioventricular rhythm worth a
  cardiologist's look. Uncoloured.
- Strip labels: `E` over the beat; the primer explains it.

## Testing

Classifier: wide and 1.5x late is `E`; wide and 1.49x is `N`; narrow and
late is `N`; wide and premature is still `V`; an `E` does not feed the
baseline. Burden: escape times and the hourly column. Report: the panel
row, the hourly header, the section's order and captions, the strip
letter, the timeline lane. Acceptance on the real recordings: 08-27
reports 8 escape beats with 5 in the 06:00 hour and the 02:15:45 and
03:14:27 beats among them; 08-25 reports 10; PVC flags unchanged on both.

## Not changed

Width measurement, so the undercount stands. Junctional escape beats
(narrow, late) are indistinguishable from sinus arrhythmia without P
waves and stay out; AV block likewise.
