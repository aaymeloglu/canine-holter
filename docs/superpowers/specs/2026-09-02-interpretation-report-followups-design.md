# Follow-ups from the cardiologist's interpretation

**Date:** 2026-09-02
**Status:** approved design.

## Source

Dr. Bordelon's interpretation of the 2026-08-27 Holter (Texas Veterinary
Cardiology, 2026-09-02) reads the HE/LX numbers differently from the
summary page, and four of her sentences bear on what we compute:

1. "The longest reported pause was 4.33 seconds, but this likely
   underestimates the length of the periods of sinus arrest as most of
   the periods of sinus arrest are interrupted by a ventricular or
   junctional escape beat."
2. "49% of all heart beats were conducted at a heart rate of 60bpm or
   less" and, in the assessment, "the percentage of time spent under
   60bpm is higher than normal."
3. "2 slow ventricular couplets were noted."
4. "Teeny is not symptomatic for the arrhythmia" alongside "episodes of
   collapse", and the HE/LX page's "No events were recorded while wearing
   monitor."

## 1. Sinus arrest bridged by escape beats

An escape beat interrupts a sinus arrest without ending it: the interval
the sinus node missed runs from the last sinus beat before the escape
beat to the first sinus beat after it. We label escape beats (`E`), so
that interval is computable. On 2026-08-27 it reaches 5.41 s at 06:26:55
against a longest RR of 4.34 s; on 2026-08-25, 5.19 s against 4.67 s.
Both cross the report's 5 s concern line, which the longest RR never
does.

`burden.py` gets `SinusArrest(start_time, end_time, escape_beats)` and
`sinus_arrests(beats)`: every interval between consecutive `N` beats
with only `E` beats between them and an RR on every beat inside (a
missing RR is a quality-gate boundary). `ArrhythmiaSummary.sinus_arrests`
holds the bridged ones; `longest_sinus_interval` is the longest sinus
interval, bridged or not, so it is never shorter than the longest N-to-N
RR. `longest_pause_sec` stays the longest RR: the two numbers answer
different questions and the report shows both.

Report: the Pauses panel gains `Sinus interval` (the longest; the label
is short because the value column is narrow), coloured by the pause band,
with the reference saying how many escape beats bridged it, and `Sinus
arrests`, the count bridged by escape beats. When the longest
sinus interval is bridged, the extremes section gets a strip bracketing
it from sinus beat to sinus beat with the escape beats marked; when it
is a plain RR it is already the longest-pause strip.

## 2. The 60 bpm line beside the class threshold

Our bradycardia threshold for a large dog is 45 bpm, a provisional table
written before any cardiologist framing existed. She reads against
60 bpm, which is also HE/LX's default. The share under 60 by our 5-beat
median is 48 % on 08-27 (her 49 %) and 47 % on 08-25.

`BRADYCARDIA_LINE_BPM = 60` and `slow_beats_at_line` on the summary. The
Heart rate panel prints `Under 60 bpm` first and `Under <class> bpm`
after it, omitting the second when the class threshold is 60. The
provisional table is otherwise untouched; this stage adds the line she
uses rather than deciding the table.

## 3. Escape couplets and runs

Consecutive escape beats are her "slow ventricular couplets"; three or
more are an idioventricular rhythm. `escape_runs(beats)` groups them like
`pvc_runs`; the summary counts `escape_couplets` (exactly two) and
`escape_runs` (`MIN_RUN_BEATS` or more), the Ventricular ectopy panel
prints both, and the escape-beat strip section shows a run as one strip
with every beat marked and its rate in the caption. With the width rule
finding 8 of her 32 escape beats, both counts are zero on both
recordings today; the counting exists so a report never silently ignores
them once beat shape is usable.

## 4. Event-button marks: not in the file

Both native recordings were scanned block by block. Besides the metadata
block, one patient-information block and one zero block at the start,
every valid block is an ECG block; there is no event block type. The
36 bytes after the session key in each ECG block are not a flag either:
the DR400 writes a fixed 34-byte record in every other block and zeros
between, and the DR200 writes a rolling ASCII log. Neither recording had
a button press (HE/LX reports none for 08-27), so nothing can be learned
from them about how a press is stored. `docs/dr200-format.md` records
this. To add event marks, wear the recorder for a short test and press
the button at noted clock times, then diff those blocks against these.

## Layout

The Heart rate panel grows to eight rows and Ventricular ectopy to ten,
so fixed panel-row tops no longer fit. `pdf.py` stacks the three panel
rows from the tallest panel in each row plus a gap, instead of
`_PANEL_TOP`. The value column moves right for the wider labels, which
left no room for the time in the run rows; the run strips carry it. A
test renders the page and checks that no two texts on a line overlap.

## Testing

Burden: sinus intervals with one and two escape beats, the missing-RR
boundary, the longest interval choosing between a bridged arrest and a
plain RR, escape run grouping and counts, the 60 bpm share. Report: the
new rows, the omitted duplicate row for a small dog, the bridged-arrest
strip present and absent, escape-run strips, the stacked panel layout.
Acceptance on 08-27: longest sinus interval 5.41 s at 06:26:55 bridged by
one escape beat, 8 sinus arrests, under-60 share 48 %.
