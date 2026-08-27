# Strip presentation: three leads, clinical scale, and captions a non-expert can read

**Date:** 2026-08-26
**Status:** approved design.

## Problem

The rhythm strips are a bare single-lead waveform with a faint vertical line
on the flagged beat, an absolute y-axis carrying the decoder's ~13 mV
offset, no grid, no beat labels, and a caption that states only the time.
A reader who is not a cardiologist cannot tell what the strip shows, why
the software flagged it, or whether it matters. Clinical strips (QRS
Diagnostics sample report; HE/LX saved strips) print at 25 mm/s and
10 mm/mV on the standard ECG grid, stack two or three leads, and label
every beat.

## Goals

1. Strips that look like the ones a cardiologist reads: true scale, ECG
   grid, all recorded leads stacked, beat labels, RR intervals, the flagged
   beats shaded.
2. Every strip says, in plain English, what it shows (with the measured
   numbers behind the software's call) and whether it is significant,
   using the same published bands and green/amber/red vocabulary as page 1.
3. A one-page "How to read these strips" primer before the first strip.
4. The analysis path is untouched: detection, gating, and classification
   keep working on the single analysis lead.

## Non-goals

- Choosing the analysis lead (still channel 0), lead-quality switching.
- Loading sibling `flashcN.dat` files for vendor-extracted input (stays
  single-channel; the strips then show one lead).
- Any change to which strips are chosen or how many (caps unchanged).

## 1. Recording carries all leads

```python
@dataclass(frozen=True, eq=False)
class Recording:
    samples: np.ndarray                      # the analysis lead, 1-D, mV (unchanged)
    sample_rate: float
    start_time: datetime | None
    source: str
    channels: np.ndarray | None = None       # every recorded lead, (n_channels, n_samples), mV, recorder order
    channel_names: tuple[str, ...] = ()      # one per channel row
```

`__post_init__` rejects `channels` that is not 2-D, whose length differs
from `samples`, or whose row count differs from `channel_names` (fail
closed: a mismatched lead set is not silently drawn).

- `load_native_flash` decodes all three DR200 channels in one pass
  (`_DELTA_COUNTS[encoded]` on the (304, 3) nibble array, per-column
  cumulative sums, pacemaker markers interpolated per channel). `channel`
  still selects the analysis lead. Names `Ch 1`, `Ch 2`, `Ch 3` (the DR200
  and HE/LX call them channels, not leads).
- `load_local_record` (WFDB): `channels = p_signal.T`, names from
  `sig_name`.
- `load_decoded_channel`: `channels=None`. The report then draws the
  analysis lead alone, named `ECG`.

Only the report consumes `channels`. `run_analysis` passes
`rec.channels` / `rec.channel_names` to `write_report`.

## 2. Drawing (`report/strip.py`)

- **Scale.** 25 mm/s and 10 mm/mV. A 6 s window is 150 mm wide on the
  page; each channel panel is 30 mm tall and shows a 3 mV range centred on
  the midpoint of the channel's window. When the signal exceeds 3 mV the
  range grows to the next whole mV and the vertical scale shrinks; when a
  long pause widens the window the horizontal scale shrinks. The scale
  actually used is printed on every strip (`25 mm/s · 10 mm/mV`, or e.g.
  `21 mm/s · 7 mm/mV`), never implied.
- **Grid.** Standard ECG paper: 1 mm minor (0.04 s / 0.1 mV) light pink,
  5 mm major (0.2 s / 0.5 mV) darker pink.
- **Baseline.** Each channel is drawn minus its window median, so the
  y-axis is millivolts around zero.
- **Leads.** One panel per channel, stacked without gaps, channel name at
  the left; the analysis lead is marked `(analysis)`.
- **Beats.** Above the top panel every detected beat in the window gets a
  label: `N` grey, `V` red bold, `?` grey for undetermined. Flagged beats
  (the strip's `mark_times`) get a translucent red band across all panels.
  RR intervals, in seconds, are printed between beats for the flagged beats
  and their immediate neighbours. A pause strip gets a double-headed arrow
  between the two beats labelled `2.97 s gap`.
- **Layout.** Two strips per page (three panels each), captions above each.

## 3. Captions (`report/generate.py`)

`StripSection` gains `captions: list[StripCaption]`, one per run, where

```python
@dataclass(frozen=True)
class StripCaption:
    title: str            # "Event 1 · 17:50:20" / "Isolated PVC 3 · 16:01:12" / "Longest pause · 17:24:29"
    what: str             # what the strip shows, with the measured numbers
    significance: str     # is it significant, in plain English
    status: str | None    # "ok" | "caution" | "alert" | None, colours the significance line
```

Measured numbers for PVC strips come from the beats themselves: each
flagged beat's RR and QRS against the median RR and QRS of the up-to-8
preceding `N` beats (the classifier's own baseline window), e.g.
"The marked beat arrived 0.42 s after the beat before it (typical here
0.80 s) and its QRS lasts 0.11 s (typical 0.06 s): early and wide is what
makes it a PVC."

| Strip | Significance line | Status |
|---|---|---|
| Isolated PVC | "One PVC on its own is common in healthy dogs; what matters is the total per 24 h (page 1)." | None |
| Couplet / triplet | "Any couplet/triplet is worth a cardiologist's review, whatever the PVC count." | alert |
| Run of 4+ at >= 180 bpm | "N PVCs in a row at R bpm is ventricular tachycardia." | alert |
| Run of 4+ under 180 bpm | "N PVCs in a row at R bpm: an accelerated idioventricular rhythm, generally less concerning than ventricular tachycardia." | caution |
| Fastest heart rate | "Fast rates during play or excitement are expected; a rate this fast at rest is not." | None |
| Slowest heart rate | "Resting dogs commonly slow to this (sinus arrhythmia); the gaps between beats are printed in seconds." | None |
| Longest pause | from the pause band: "< 2.5 s: within the usual range" / "2.5-5 s: common in resting dogs" / "> 5 s, or any pause with fainting or collapse: worth review" | pause_status |

The primer page, `HOW_TO_READ_STRIPS` in `report/common.py`, is a list of
lines: paper speed and squares, the labels and red band, why three leads
("the same heartbeat seen from three angles: a beat that appears in all
three is real; a spike in only one is usually movement"), what a PVC
looks like ("a beat with a different shape from its neighbours that
arrives early, usually followed by a longer gap"), that the significance
lines use the bands on page 1, and that the labels are the software's
provisional calls which the strips exist to let a reader check.

## 4. PDF (`report/pdf.py`)

`write_pdf` takes `channels` and `channel_names` (the analysis lead as a
one-row array when there are no channels). With no waveform at all the
text-page path is unchanged. Page order: summary, timeline (+ table
pages), primer page (only when strips will be drawn), strip pages at two
per page.

## 5. Tests

- `Recording` invariants; three-channel DR200 decode against literal
  values from the documented delta table (channels 1 and 2 of the synthetic
  factory block); WFDB channels and names.
- `strip.py`: one axes per channel, grid lines present, beat label texts,
  one band per flagged beat per channel, RR texts, pause arrow, the scale
  string for a 6 s and a widened window.
- `generate.py`: caption texts with literal numbers for an isolated PVC and
  a couplet; statuses at 179/180 bpm and 2.49/2.5/5/5.01 s; one caption per
  run in every section.
- `pdf.py`: page counts at two strips per page and the primer page.
- End to end: `report_text` includes captions; Teeny's recording
  regenerated by hand and inspected.
