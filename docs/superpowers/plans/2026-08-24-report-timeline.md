# Report Wall-Clock Times & Event Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label report events by wall-clock time (with a `--start-time` override) and add a heart-rate + event-lane timeline figure to the report.

**Architecture:** `write_report` gains a `start_time` argument and a shared elapsed→label formatter; a new `report/timeline.py` renders `timeline.png` from beats + summary only; `pipeline.run_analysis` gains `start_time` and a `parse_start_time` helper that the CLI's new `--start-time` flag feeds. Spec: `docs/superpowers/specs/2026-08-24-report-timeline-design.md`.

**Tech Stack:** Python 3, matplotlib (Agg backend, already used for strips), numpy, pytest.

---

## File map

- Modify `src/canine_holter/report/generate.py` — `format_time`, start/duration summary lines, wall-clock labels in bullets and strip titles, `## Timeline` section.
- Create `src/canine_holter/report/timeline.py` — `plot_timeline(beats, summary, start_time, out_path)`.
- Modify `src/canine_holter/pipeline.py` — `parse_start_time`, `start_time=` parameter, pass `rec.start_time` into `write_report`.
- Modify `src/canine_holter/cli.py` — `--start-time` flag.
- Tests: `tests/report/test_generate.py`, `tests/report/test_timeline.py` (new), `tests/test_pipeline.py`, `tests/test_cli.py`.

Run all tests with `.venv/bin/pytest -q` from the repo root.

---

### Task 1: `format_time` helper

**Files:** Modify `src/canine_holter/report/generate.py`; Test `tests/report/test_generate.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/report/test_generate.py`)

```python
from datetime import datetime
from canine_holter.report.generate import format_time


def test_format_time_with_start_gives_wall_clock_and_elapsed():
    start = datetime(2026, 8, 23, 15, 33, 8)
    assert format_time(8232.8, start) == "17:50:20 (t=8232.8s)"


def test_format_time_without_start_gives_elapsed_only():
    assert format_time(8232.8, None) == "t=8232.8s"
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/report/test_generate.py -q -k format_time` — expect ImportError on `format_time`.
- [ ] **Step 3: Implement** in `generate.py` (below `STRIP_WINDOW_SEC`):

```python
from datetime import datetime, timedelta


def format_time(elapsed_sec: float, start_time: datetime | None) -> str:
    """Render an elapsed-seconds offset as a wall-clock label when the
    recording start is known, falling back to elapsed seconds otherwise."""
    elapsed = f"t={elapsed_sec:.1f}s"
    if start_time is None:
        return elapsed
    clock = (start_time + timedelta(seconds=elapsed_sec)).strftime("%H:%M:%S")
    return f"{clock} ({elapsed})"
```

- [ ] **Step 4: Run** the same command — expect 2 passed.
- [ ] **Step 5: Commit** `git commit -am "Add format_time helper for wall-clock report labels"`.

### Task 2: `write_report` takes `start_time`; summary lines and labels

**Files:** Modify `src/canine_holter/report/generate.py`; Test `tests/report/test_generate.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_report_summary_includes_start_and_duration():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(150.0, 0.8, "N")]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    with tempfile.TemporaryDirectory() as out_dir:
        path = write_report(beats, summary, out_dir, samples=None, sample_rate=None, start_time=start)
        content = open(path).read()
        assert "- Recording start: 2026-08-23 15:33:08" in content
        assert "- Duration: 0h 2m" in content


def test_report_summary_without_start_says_unknown():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as out_dir:
        path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        assert "- Recording start: unknown" in open(path).read()


def test_flagged_event_uses_wall_clock_label_when_start_known():
    beats = [_beat(0.0, None, "N")] + [_beat(i * 0.8, 0.8, "V") for i in range(1, 3)]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    with tempfile.TemporaryDirectory() as out_dir:
        path = write_report(beats, summary, out_dir, samples=None, sample_rate=None, start_time=start)
        assert "Event 1: 2 consecutive PVCs at ~15:33:09 (t=1.2s)" in open(path).read()
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/report/test_generate.py -q` — expect TypeError (unexpected kwarg `start_time`) / assertion failures.
- [ ] **Step 3: Implement.** Signature becomes `write_report(beats, summary, out_dir, samples, sample_rate, start_time: datetime | None = None)`. `_plot_strip` gains a `title: str` parameter and uses `ax.set_title(title)`. In `write_report`:

```python
    duration_sec = beats[-1].time if beats else 0.0
    hours, rem = divmod(int(duration_sec), 3600)
    start_text = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "unknown"
    lines = [
        ...,
        "## Summary",
        f"- Recording start: {start_text}",
        f"- Duration: {hours}h {rem // 60}m",
        f"- Total beats: ...",
```

and in the flagged loop:

```python
            label = format_time(center_time, start_time)
            lines.append(f"- Event {i + 1}: {len(run)} consecutive PVCs at ~{label}")
            ...
                _plot_strip(samples, sample_rate, center_time, plot_path, title=f"Rhythm strip around {label}")
```

Update the existing `_plot_strip` test call to pass `title="strip"`.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -q` — expect all passed.
- [ ] **Step 5: Commit** `git commit -am "Label report events with wall-clock times"`.

### Task 3: `plot_timeline`

**Files:** Create `src/canine_holter/report/timeline.py`; Test `tests/report/test_timeline.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import tempfile
from datetime import datetime
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.timeline import plot_timeline


def _beat(time, rr, label):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def _summary(**kw):
    base = dict(total_beats=0, pvc_count=0, pvc_burden_pct=0.0, couplets=0, triplets=0,
                vtach_runs=0, bradycardia_events=[], tachycardia_events=[], pauses=[])
    base.update(kw)
    return ArrhythmiaSummary(**base)


def _render(beats, summary, start_time):
    with tempfile.TemporaryDirectory() as out_dir:
        out = os.path.join(out_dir, "timeline.png")
        plot_timeline(beats, summary, start_time, out)
        assert os.path.getsize(out) > 0


def test_renders_with_every_event_type_and_wall_clock():
    beats = [_beat(i * 0.8, 0.8 if i else None, "V" if i % 10 == 0 else "N") for i in range(300)]
    summary = _summary(bradycardia_events=[(10.0, 30.0)], tachycardia_events=[(100.0, 100.5)], pauses=[50.0, 60.0])
    _render(beats, summary, datetime(2026, 8, 23, 15, 33, 8))


def test_renders_without_start_time():
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(300)]
    _render(beats, _summary(), None)


def test_renders_with_no_events_and_no_beats():
    _render([], _summary(), None)
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/report/test_timeline.py -q` — expect ModuleNotFoundError.
- [ ] **Step 3: Implement** `src/canine_holter/report/timeline.py`:

```python
"""Whole-recording timeline: heart-rate trend plus lanes marking where
PVCs, pauses, and sustained brady/tachycardia occur."""
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.types import Beat

HR_BIN_SEC = 60.0
MIN_SPAN_SEC = 5.0  # so a sub-second tachy event is still visible

# Categorical slots 1-4 of the dataviz reference palette; validated adjacent-CVD safe.
LANES = [  # (label, color)
    ("PVC", "#2a78d6"),
    ("Pause", "#eb6834"),
    ("Brady", "#1baf7a"),
    ("Tachy", "#eda100"),
]
HR_COLOR = "#52514e"


def _heart_rate_trend(beats):
    """(bin_center_sec, bpm) per 1-min bin with >= 2 RR intervals; NaN gaps elsewhere."""
    times = np.array([b.time for b in beats if b.rr_interval])
    rr = np.array([b.rr_interval for b in beats if b.rr_interval])
    if len(times) == 0:
        return np.array([]), np.array([])
    n_bins = int(times.max() // HR_BIN_SEC) + 1
    idx = (times // HR_BIN_SEC).astype(int)
    bpm = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = rr[idx == i]
        if len(sel) >= 2:
            bpm[i] = 60.0 / np.median(sel)
    centers = (np.arange(n_bins) + 0.5) * HR_BIN_SEC
    return centers, bpm


def plot_timeline(beats, summary, start_time, out_path):
    if start_time is None:
        to_x = lambda sec: sec / 60.0
        to_w = lambda sec: sec / 60.0
    else:
        to_x = lambda sec: mdates.date2num(start_time + timedelta(seconds=float(sec)))
        to_w = lambda sec: sec / 86400.0

    fig, (ax_hr, ax_ev) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]}
    )
    centers, bpm = _heart_rate_trend(beats)
    ax_hr.plot([to_x(c) for c in centers], bpm, color=HR_COLOR, linewidth=1.5)
    ax_hr.set_ylabel("Heart rate (bpm)")
    ax_hr.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax_hr.set_title("Recording timeline")

    lane_data = [
        [b.time for b in beats if b.label == "V"],
        list(summary.pauses),
        list(summary.bradycardia_events),
        list(summary.tachycardia_events),
    ]
    for lane, ((label, color), items) in enumerate(zip(LANES, lane_data)):
        y = len(LANES) - 1 - lane
        for item in items:
            if isinstance(item, tuple):
                start, end = item
                width = to_w(max(end - start, MIN_SPAN_SEC))
                ax_ev.broken_barh([(to_x(start), width)], (y - 0.35, 0.7), color=color)
            else:
                ax_ev.vlines(to_x(item), y - 0.35, y + 0.35, color=color, linewidth=1.2)
    ax_ev.set_yticks(range(len(LANES)))
    ax_ev.set_yticklabels([label for label, _ in reversed(LANES)])
    ax_ev.set_ylim(-0.6, len(LANES) - 0.4)
    for ax in (ax_hr, ax_ev):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    if start_time is None:
        ax_ev.set_xlabel("minutes from start")
    else:
        ax_ev.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_ev.set_xlabel("time of day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report/test_timeline.py -q` — expect 3 passed.
- [ ] **Step 5: Commit** `git add -A src/canine_holter/report/timeline.py tests/report/test_timeline.py && git commit -m "Add heart-rate and event-lane timeline figure"`.

### Task 4: Link the timeline from the report

**Files:** Modify `src/canine_holter/report/generate.py`; Test `tests/report/test_generate.py`

- [ ] **Step 1: Write the failing test**

```python
def test_report_links_timeline_png():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as out_dir:
        path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        content = open(path).read()
        assert "## Timeline" in content
        assert "![timeline](timeline.png)" in content
        assert os.path.exists(os.path.join(out_dir, "timeline.png"))
```

Update the two existing tests that assert `plot_files == []` to assert no `event_*` files instead:

```python
        strip_files = [f for f in os.listdir(out_dir) if f.startswith("event_")]
        assert strip_files == []
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/report -q` — expect the new test to fail.
- [ ] **Step 3: Implement.** In `write_report`, after the summary block and before flagged events:

```python
    timeline_path = os.path.join(out_dir, "timeline.png")
    plot_timeline(beats, summary, start_time, timeline_path)
    lines += ["## Timeline", "![timeline](timeline.png)", ""]
```

with `from canine_holter.report.timeline import plot_timeline` at the top.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -q` — expect all passed.
- [ ] **Step 5: Commit** `git commit -am "Embed timeline figure in report"`.

### Task 5: `parse_start_time` and `run_analysis(start_time=)`

**Files:** Modify `src/canine_holter/pipeline.py`; Test `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date, datetime
import pytest
from canine_holter.pipeline import parse_start_time

HEADER = datetime(2026, 8, 23, 15, 33, 8)


def test_parse_start_time_hh_mm_borrows_header_date():
    assert parse_start_time("15:36", HEADER) == datetime(2026, 8, 23, 15, 36)


def test_parse_start_time_hh_mm_ss():
    assert parse_start_time("15:36:10", HEADER) == datetime(2026, 8, 23, 15, 36, 10)


def test_parse_start_time_full_datetime_ignores_header():
    assert parse_start_time("2026-08-22 09:00", HEADER) == datetime(2026, 8, 22, 9, 0)


def test_parse_start_time_time_only_without_header_uses_today():
    result = parse_start_time("15:36", None)
    assert result.date() == date.today()
    assert (result.hour, result.minute) == (15, 36)


def test_parse_start_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_start_time("half past three", HEADER)


def test_run_analysis_start_time_string_is_parsed_against_header():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir, start_time="2026-08-23 15:36")
        assert "- Recording start: 2026-08-23 15:36:00" in open(report_path).read()


def test_run_analysis_start_time_override_appears_in_report():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir, start_time=datetime(2026, 8, 23, 15, 36))
        assert "- Recording start: 2026-08-23 15:36:00" in open(report_path).read()
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_pipeline.py -q` — expect ImportError.
- [ ] **Step 3: Implement** in `pipeline.py`:

```python
from dataclasses import replace
from datetime import date, datetime

_START_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M")


def parse_start_time(text: str, header_start: datetime | None) -> datetime:
    """Parse a --start-time value. Time-only forms take their date from the
    recording header, or today when the header has no clock."""
    for fmt in _START_TIME_FORMATS:
        try:
            parsed = datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
        if fmt.startswith("%Y"):
            return parsed
        base = header_start.date() if header_start else date.today()
        return datetime.combine(base, parsed.time())
    raise ValueError(
        f"Unrecognized start time {text!r}; use HH:MM, HH:MM:SS, or YYYY-MM-DD HH:MM[:SS]"
    )


def run_analysis(input_path, out_dir, dog_weight_class="medium", start_time: datetime | str | None = None) -> str:
    rec = load_recording(input_path)
    if isinstance(start_time, str):
        start_time = parse_start_time(start_time, rec.start_time)
    if start_time is not None:
        rec = replace(rec, start_time=start_time)
    ...
    return write_report(labeled, summary, out_dir, samples=rec.samples,
                        sample_rate=rec.sample_rate, start_time=rec.start_time)
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_pipeline.py -q` — expect all passed.
- [ ] **Step 5: Commit** `git commit -am "Pass recording start time through the pipeline with an override"`.

### Task 6: `--start-time` CLI flag

**Files:** Modify `src/canine_holter/cli.py`; Test `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_main_start_time_override_reaches_report(monkeypatch):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(sys, "argv",
            ["canine-holter", INPUT_PATH, "--out", out_dir, "--start-time", "2026-08-23 15:36"])
        main()
        content = open(os.path.join(out_dir, "report.md")).read()
        assert "- Recording start: 2026-08-23 15:36:00" in content


def test_main_rejects_unparseable_start_time(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as out_dir:
        monkeypatch.setattr(sys, "argv",
            ["canine-holter", INPUT_PATH, "--out", out_dir, "--start-time", "teatime"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "start-time" in capsys.readouterr().err
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_cli.py -q` — expect failures.
- [ ] **Step 3: Implement.** A time-only value needs the header's date, which only exists after loading, so `run_analysis` accepts the raw string and parses it itself (Task 5's signature is `start_time: datetime | str | None`; a `str` goes through `parse_start_time(text, rec.start_time)`). `cli.py`:

```python
    parser.add_argument(
        "--start-time",
        help="Override the recording start time: HH:MM, HH:MM:SS, or 'YYYY-MM-DD HH:MM[:SS]'. "
             "Time-only values use the recording's own date.",
    )
    args = parser.parse_args()

    try:
        report_path = run_analysis(args.input, args.out,
                                   dog_weight_class=args.dog_weight_class,
                                   start_time=args.start_time)
    except ValueError as exc:
        parser.error(f"--start-time: {exc}")
```

- [ ] **Step 4: Run** `.venv/bin/pytest -q` — expect all passed.
- [ ] **Step 5: Commit** `git commit -am "Add --start-time CLI override"`.

### Task 7: Verify on the real recording and update docs

- [ ] Run `.venv/bin/canine-holter <scratchpad>/drive/flash.dat --out <scratchpad>/report2 --dog-weight-class large`, open `timeline.png` and `report.md`, confirm wall-clock labels and that the timeline shows the pause cluster at ~17:56–18:01.
- [ ] Add `--start-time` to the README usage block.
- [ ] Commit `git commit -am "Document --start-time"`.
