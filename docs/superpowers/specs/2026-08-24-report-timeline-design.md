# Report Wall-Clock Times & Event Timeline — Design

Date: 2026-08-24

## Motivation

The first real DR200 recording of Teeny (2026-08-23, 2 h 28 min) ran cleanly
through the pipeline, but the report was hard to relate to what actually
happened. Flagged events were labelled by elapsed seconds (`~t=8232.8s`), and
the summary gave only totals — 64 pauses, 46 bradycardia events — with no way
to see *when* they occurred. In that recording most of the pauses were packed
into the last eight minutes, after the vest slipped off; the report could not
show that.

## Goals

- Label every event by wall-clock time, using the start time the recorder
  stamps into `flash.dat`, with a CLI override for when the device clock is
  known to be wrong.
- Add a timeline figure to the report that shows heart rate over the whole
  recording and where PVCs, pauses, bradycardia, and tachycardia cluster.

## Non-goals

- Signal-quality / off-body gating. This work makes bad-contact stretches
  *visible*; excluding them is a separate feature.
- Threshold lines on the heart-rate panel, a per-event table, or any change to
  the GUI beyond what it inherits from `run_analysis`.

## Wall-clock labels

`write_report` gains a `start_time: datetime | None` argument. `Recording`
already carries `start_time` (parsed from the DR200 header; `None` for WFDB
records); `run_analysis` passes it through.

A single formatter in `report/generate.py` renders an elapsed-seconds value as:

- `16:50:21 (t=8232.8s)` when `start_time` is known
- `t=8232.8s` when it is not

It is used for flagged-event bullets and for rhythm-strip titles, so the
markdown and the images agree.

The Summary section gains two lines, placed before the beat counts:

- `Recording start: 2026-08-23 15:33:08` (or `unknown`)
- `Duration: 2h 28m`, from the last beat time. Beats are the only timing
  source that exists on every path (no samples on the no-waveform path), and
  the last beat is within seconds of the true end.

### `--start-time` override

`cli.py` adds `--start-time`, accepting `HH:MM`, `HH:MM:SS`, or
`YYYY-MM-DD HH:MM[:SS]`. Time-only values take their date from the header
start time, or today's date when the header has none. Parsing lives in
`pipeline.py` (`parse_start_time(text, header_start)`) so it is testable
without argparse. `run_analysis` accepts `start_time: datetime | None = None`
and applies it with `dataclasses.replace(rec, start_time=...)` — `Recording`
is a frozen dataclass. Unparseable input raises `ValueError`; the CLI turns
that into an argparse error.

## Timeline figure

New module `report/timeline.py` with

```
plot_timeline(beats, summary, start_time, out_path) -> None
```

It writes `timeline.png` in the report directory. `write_report` links it from
a new `## Timeline` section placed between Summary and Flagged events. The
figure needs only `beats` and `summary`, so it is produced even when no
waveform samples are supplied.

Layout: one figure, two vertically stacked panels sharing the x-axis.

- **Heart rate** (top, taller): median RR per 1-minute bin, converted to bpm,
  drawn as a line. Median is robust to the occasional PVC; 1-minute bins give
  ~150 points on a 2.5 h recording. Bins with fewer than 2 beats are left as
  gaps rather than plotted as zero.
- **Event lanes** (bottom): four horizontal lanes, top to bottom **PVC**,
  **Pause**, **Brady**, **Tachy**.
  - PVC: one vertical tick per beat labelled `V`.
  - Pause: one tick per entry in `summary.pauses`.
  - Brady / Tachy: a horizontal span per `(start, end)` tuple, so a sustained
    stretch reads as a bar. Spans shorter than the figure's pixel resolution
    get a minimum visible width.

X-axis: `HH:MM` wall-clock ticks when `start_time` is known (matplotlib date
axis), otherwise minutes elapsed. The lane drawing code is the same in both
cases; only the x-values and the axis formatter differ.

Colors follow the `dataviz` skill's palette guidance; each lane has one
color, used for both its ticks and its label.

Empty inputs (no PVCs, no pauses, no events, or no beats at all) render an
empty lane or panel rather than raising.

## Error handling

- `--start-time` that does not match any accepted form → argparse error with
  the accepted forms listed.
- Missing `start_time` anywhere degrades to elapsed-time labels; nothing
  requires a clock.

## Testing

- `tests/report/test_generate.py`: label formatting with and without
  `start_time`; the Summary contains the start/duration lines; report contains
  `timeline.png` and the file exists; strip title / bullet use wall-clock text
  when a start is given.
- `tests/report/test_timeline.py`: renders with a full set of event types;
  renders with zero events; renders with `start_time=None`; renders with an
  empty beat list.
- `tests/test_pipeline.py`: `parse_start_time` for all three forms, the
  date-borrowing rule, and rejection of garbage; `run_analysis(...,
  start_time=...)` overrides the header value in the report.
- `tests/test_cli.py`: `--start-time` reaches the report; a bad value exits
  with an argparse error.
