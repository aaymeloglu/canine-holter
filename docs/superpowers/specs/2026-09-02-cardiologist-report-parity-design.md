# Cardiologist-report parity

**Date:** 2026-09-02
**Status:** approved design. Stage 1 is this spec's implementation scope;
stages 2 and 3 are recorded here so their motivation is not lost, and each
gets its own spec before it is built.

## Problem

The cardiologist's report for Teeny's 2026-08-27 recording (NorthEast
HE/LX Analysis, `~/Downloads/Holter_(Report).pdf`) has front-page numbers
ours lacks, and some of its numbers differ from ours. Andy wants a program
that produces a complete analysis, so every stat on their front page should
either be on ours or be explicitly stated as not assessed, and every
disagreement should be understood.

## What the comparison found

Our beats agree with theirs. Heart-rate variability from our beats is
SDNN 528.6 / RMSSD 492 / pNN50 69.8 % against their 527.1 / 489.6 / 71.1 %;
clock-hour beat counts are within 2 % in every hour; mean heart rate is 58
on both. The differences are definitional or small and specific:

| Stat | HE/LX | Ours | Cause |
|---|---|---|---|
| Analyzed, beats | 23h38m, 82,205 | 24h23m, 84,562 | They stopped at 10:17 and excluded 10 min in each of the 22:00 and 23:00 hours; our signal there is clean ECG. Ours is right. |
| Max HR | 245 at 18:20 | 234 at 12:06 | Their HR is a 4-beat weighted average that freezes during a sudden run (manual NEMM027 Rev P, Appendix A); their 245 is motion noise, our 234 is a real sinus tachycardia on all three leads. |
| Min HR | 29 at 03:08 | 21 at 06:23 | Same average updates by 1/32 on an RR jump over 25 %, so it never reaches the real 2.2-3.5 s sinus-arrhythmia intervals at 06:23. |
| Pauses | 0 (> 4.5 s) | 1772 (>= 2.5 s) | Threshold. |
| Longest RR | 4.333 s at 03:14:51 | 4.73 s at 06:35:47 | Ours is a missed beat: NeuroKit finds it on lead 1 but not on leads 2 and 3, where the QRS is a small r and a deep S, and two-lead agreement drops it. Negating those leads finds it. 8 of our 1773 gaps over 2.5 s have this problem. |
| Ventricular ectopy | 42 (5 early, 32 late, 2 pairs) | 7 | Their "late" beats are ventricular escape beats (wide, after a long RR, mostly 03:00-06:00); our rule needs premature and wide. Of their early beats we match 4; the two we miss (11:51:54, 12:06:14) are large different-shaped beats that were not premature. Our 11:42:28-29 (T wave) and 12:37:30 (noise burst) are false. Their pairs at 12:24 and 19:00 sit in motion noise and could not be confirmed. |
| Supraventricular ectopy | 35 | none | Theirs comes from beat templates plus a prematurity setting. Any RR-only prematurity rule fires thousands of times on this dog (pNN50 70 %). |
| RR variability, brady/tachy beat share, clock-hour table, one strip per hour | present | absent | Additions. |

Their strip times run 49 s behind ours: HE/LX takes the start as 10:18:00,
we use the header's 10:18:49.

Decision: keep our heart-rate definition (5-beat median for extremes, mean
over every RR). It proved more faithful than theirs on this recording and is
already documented on the report.

## Stage 1: report stats (this implementation)

No detection or classification change. Everything below is aggregation in
`arrhythmia/burden.py` and content in `report/generate.py`, plus the page
layout in `report/pdf.py`.

### Aggregation (`arrhythmia/burden.py`)

- `HeartRateVariability(sdnn_ms, rmssd_ms, pnn50_pct, nn_intervals)`,
  built by `heart_rate_variability(beats)`. An NN interval is the RR of a
  beat labeled `N` whose previous beat is also `N`; a successive difference
  is between the NN intervals of two consecutive beats. SDNN is the standard
  deviation of the NN intervals, RMSSD the root mean square of the
  successive differences, pNN50 the percentage of successive differences
  over 50 ms. `None` with fewer than two successive differences.
  `ArrhythmiaSummary.heart_rate_variability` carries it.
- Rate shares: `slow_beats`, `fast_beats`, and `rated_beats` on
  `ArrhythmiaSummary`, counting the `HR_EXTREME_WINDOW_BEATS`-beat median
  windows (the same windows as the extremes) whose rate is below the
  bradycardia threshold or above the tachycardia threshold for the weight
  class. The thresholds used are carried as `brady_threshold_bpm` and
  `tachy_threshold_bpm` so the report can print them.
- `long_pauses`: RR intervals over `LONG_PAUSE_THRESHOLD_SEC = 5.0`, the
  report's concern line, counted beside the existing 2.5 s pauses.
- Clock-hour rows: `hourly_rows` and `summarize` take `start_time`. With a
  known start the first row runs from the recording start to the next
  clock hour and every later row is a clock hour; without one the rows
  bin from the recording start as before. `HourRow` is unchanged, so the
  labels, timeline, and table need no change. `pipeline.py` passes the
  recording's start.

### Content (`report/generate.py`)

Six summary panels, in order: Recording, Heart rate, Ventricular ectopy,
Supraventricular ectopy, Pauses, RR variability.

- Heart rate adds `Under <brady> bpm` and `Over <tachy> bpm` rows, each
  `count (pct%)` of rated windows with the reference `5-beat median`, and
  the Brady/Tachy events rows state their rule in the reference
  (`3+ beats < 45 bpm`).
- Supraventricular ectopy has one row, `SVPBs: not assessed`, with the
  reference `needs P-wave analysis`. Absent must read as absent, not zero.
- Pauses adds `Pauses > 5 s` between the count and the longest.
- RR variability lists SDNN, RMSSD, and pNN50, uncoloured (no canine
  bands), with the NN-interval count as reference; a single
  `not computed` row when there is no `HeartRateVariability`.
- A last strip section, `One strip per hour`: for each hourly row with
  beats, a strip centred on the first beat at or after the row's start,
  captioned with the hour and the hour's min/mean/max rate. Its cap is
  `MAX_HOURLY_STRIPS = 48` (two days), stated in the heading like the
  other sections. The primer gains a sentence saying why these strips are
  there.

### Layout (`report/pdf.py`)

The summary page becomes a 3x2 grid of panels; the footer moves below the
third row. The tallest panel (Heart rate, seven rows) fits the existing
row pitch.

### Testing

TDD per rule: literal-value HRV (hand-computed SDNN, RMSSD, pNN50), the NN
exclusions around `V` beats and missing RRs, windowed rate shares, the long
pause count, clock alignment with a start of 10:18:49 and with no start,
the six groups and their rows, the hourly strip section and its cap, and
the six-panel page. Acceptance on the real recording: the 08-27 report's
clock-hour table lines up with the cardiologist's, SDNN/RMSSD/pNN50 within
1 %, and the below-60 share near 49 % when the medium class is used.

## Stage 2: detection polarity (follow-up spec)

Detect each lead in both polarities and merge peaks within the QRS search
window, so a lead whose QRS is a small r and deep S no longer loses beats.
A naive union of the two runs put a second peak on T waves 110 ms after
the R (12:53:35-42 became a false run of four), so the merge needs the
fiducial rule, not a fixed 100 ms. Acceptance: longest RR 4.34 s at
03:15:46 on the 08-27 recording, the 06:35:01 and 06:35:47 beats found,
no new PVC flags, no RR under 0.2 s.

## Stage 3: ventricular escape beats (follow-up spec)

A new beat label for wide beats that arrive late after a long RR, counted
and stripped separately from PVCs. Depends on stage 2 because the fiducial
move changes widths. A first count with the current widths found 13
candidates against their 32, so the width measure will undercount; the
spec should say so.

## Out of scope

Supraventricular ectopy, AV block, and junctional escape beats all need
P-wave analysis; the report states the first as not assessed and the
CLAUDE.md limits list the rest.
