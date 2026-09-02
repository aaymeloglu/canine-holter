# Report parity stats (stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put every front-page stat of the cardiologist's HE/LX report on ours (or state it as not assessed) without touching detection or classification.

**Architecture:** New aggregates (HRV, rate shares, long pauses, clock-hour rows) go in `arrhythmia/burden.py` and ride on `ArrhythmiaSummary`; `report/generate.py` turns them into two new summary panels, new rows, and a last strip section; `report/pdf.py` lays the six panels out 3x2. Spec: `docs/superpowers/specs/2026-09-02-cardiologist-report-parity-design.md`.

**Tech Stack:** Python 3.11, NumPy, pytest. Run tests with `.venv/bin/pytest`.

---

### Task 1: Heart-rate variability

**Files:**
- Modify: `src/canine_holter/arrhythmia/burden.py`
- Test: `tests/arrhythmia/test_burden.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/arrhythmia/test_burden.py`)

```python
import pytest
from canine_holter.arrhythmia.burden import heart_rate_variability


def _chain(rrs, labels=None):
    """Beats at cumulative times from a list of RRs (first beat has none)."""
    labels = labels or ["N"] * len(rrs)
    t, beats = 0.0, []
    for rr, label in zip(rrs, labels):
        t += rr or 0.0
        beats.append(_beat(t, rr, label))
    return beats


def test_hrv_sdnn_rmssd_pnn50_from_literal_nn_intervals():
    hrv = heart_rate_variability(_chain([None, 0.8, 0.9, 0.8, 1.0, 1.04]))
    # NN = 800, 900, 800, 1000, 1040 ms: mean 908, population SD sqrt(9856)
    assert hrv.nn_intervals == 5
    assert hrv.sdnn_ms == pytest.approx(99.277, abs=0.01)
    # successive differences 100, -100, 200, 40: RMS sqrt(15400); three of four over 50 ms
    assert hrv.rmssd_ms == pytest.approx(124.097, abs=0.01)
    assert hrv.pnn50_pct == pytest.approx(75.0)


def test_hrv_skips_a_pvc_its_follower_and_the_first_beat():
    beats = _chain([None, 0.8, 0.5, 1.1, 0.8, 0.8, 0.8], ["N", "N", "V", "N", "N", "N", "N"])
    hrv = heart_rate_variability(beats)
    # The 0.5 (V) and the 1.1 (after a V) are not NN; the chain restarts at the 0.8s.
    assert hrv.nn_intervals == 4
    assert hrv.sdnn_ms == 0.0 and hrv.rmssd_ms == 0.0 and hrv.pnn50_pct == 0.0


def test_hrv_is_none_with_fewer_than_two_successive_differences():
    assert heart_rate_variability(_chain([None, 0.8, 0.9])) is None
    assert heart_rate_variability([]) is None


def test_summary_carries_hrv():
    assert summarize(_chain([None, 0.8, 0.9, 0.8])).heart_rate_variability.nn_intervals == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/arrhythmia/test_burden.py -k hrv -v`
Expected: ImportError on `heart_rate_variability`.

- [ ] **Step 3: Implement** in `src/canine_holter/arrhythmia/burden.py`

After `HeartRateStats`:

```python
MIN_SUCCESSIVE_DIFFERENCES = 2  # under this, variability is one number's noise


@dataclass(frozen=True)
class HeartRateVariability:
    """Time-domain variability of the NN intervals: RRs between consecutive
    normal beats. A successive difference is between the NN intervals of
    two consecutive beats, so a PVC, an unmeasured beat, or a quality-gate
    boundary breaks the chain rather than contributing a false jump."""
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float
    nn_intervals: int
```

After `heart_rate_stats`:

```python
def _nn_intervals(beats: list[Beat]) -> list[float | None]:
    """Per beat, its NN interval in seconds: the RR of a normal beat whose
    predecessor is normal; None otherwise."""
    return [
        b.rr_interval if i > 0 and b.label == "N" and beats[i - 1].label == "N" and b.rr_interval else None
        for i, b in enumerate(beats)
    ]


def heart_rate_variability(beats: list[Beat]) -> HeartRateVariability | None:
    """SDNN, RMSSD, and pNN50 over the NN intervals, or None with fewer
    than MIN_SUCCESSIVE_DIFFERENCES successive differences."""
    nn = _nn_intervals(beats)
    values = np.array([v for v in nn if v is not None]) * 1000.0
    diffs = np.array([nn[i] - nn[i - 1] for i in range(1, len(nn)) if nn[i] is not None and nn[i - 1] is not None]) * 1000.0
    if len(diffs) < MIN_SUCCESSIVE_DIFFERENCES:
        return None
    return HeartRateVariability(
        sdnn_ms=float(values.std()),
        rmssd_ms=float(np.sqrt(np.mean(diffs**2))),
        pnn50_pct=float(np.mean(np.abs(diffs) > 50.0) * 100.0),
        nn_intervals=len(values),
    )
```

Add the field to `ArrhythmiaSummary` after `heart_rate`:

```python
    heart_rate_variability: HeartRateVariability | None = None  # None with too few NN intervals
```

and in `summarize`'s return: `heart_rate_variability=heart_rate_variability(beats),`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/arrhythmia -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/arrhythmia/burden.py tests/arrhythmia/test_burden.py
git commit -m "Burden: heart-rate variability over NN intervals"
```

### Task 2: Rate shares and long pauses

**Files:**
- Modify: `src/canine_holter/arrhythmia/burden.py`
- Test: `tests/arrhythmia/test_burden.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_rate_shares_count_five_beat_median_windows_against_the_class_thresholds():
    beats = _chain([None] + [0.3] * 9 + [1.5] * 9)  # 200 bpm then 40 bpm
    s = summarize(beats, dog_weight_class="medium")  # 50 / 160 bpm
    # 18 RRs make 14 windows; a window's median flips from 0.3 to 1.5 once three of five are slow.
    assert (s.fast_beats, s.slow_beats, s.rated_beats) == (7, 7, 14)
    assert (s.brady_threshold_bpm, s.tachy_threshold_bpm) == (50, 160)


def test_rate_shares_are_zero_with_too_few_beats():
    s = summarize(_chain([None, 0.3, 0.3]))
    assert (s.fast_beats, s.slow_beats, s.rated_beats) == (0, 0, 0)


def test_long_pauses_count_rrs_over_five_seconds():
    s = summarize(_chain([None, 0.8, 3.0, 5.0, 5.5, 0.8]))
    assert len(s.pauses) == 3
    assert s.long_pauses == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/arrhythmia/test_burden.py -k "rate_shares or long_pauses" -v`
Expected: AttributeError on `fast_beats` / `long_pauses`.

- [ ] **Step 3: Implement**

Constant after `PAUSE_THRESHOLD_SEC`:

```python
LONG_PAUSE_THRESHOLD_SEC = 5.0  # the report's concern line (report/reference.py PAUSE_CONCERN_SEC); counted beside the 2.5 s pauses so a vendor report with a higher setting reads side by side
```

Fields on `ArrhythmiaSummary` after `longest_pause_sec`:

```python
    long_pauses: int = 0  # RR intervals over LONG_PAUSE_THRESHOLD_SEC
    slow_beats: int = 0  # HR_EXTREME_WINDOW_BEATS-beat median windows under the bradycardia threshold ...
    fast_beats: int = 0  # ... and over the tachycardia threshold ...
    rated_beats: int = 0  # ... out of this many windows
    brady_threshold_bpm: float = 0.0
    tachy_threshold_bpm: float = 0.0
```

In `summarize`, after `tachy_threshold = ...`:

```python
    _, window_rr = _windowed_rr(beats)
    window_bpm = 60.0 / window_rr if len(window_rr) else window_rr
```

and in the return:

```python
        long_pauses=sum(1 for rr in rr_intervals if rr > LONG_PAUSE_THRESHOLD_SEC),
        slow_beats=int(np.sum(window_bpm < brady_threshold)),
        fast_beats=int(np.sum(window_bpm > tachy_threshold)),
        rated_beats=len(window_bpm),
        brady_threshold_bpm=brady_threshold,
        tachy_threshold_bpm=tachy_threshold,
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/arrhythmia -v` — PASS.

- [ ] **Step 5: Commit** `git commit -am "Burden: rate shares and pauses over 5 s"`

### Task 3: Clock-hour rows

**Files:**
- Modify: `src/canine_holter/arrhythmia/burden.py` (`hourly_rows`, `summarize`)
- Modify: `src/canine_holter/pipeline.py`
- Test: `tests/arrhythmia/test_burden.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime


def test_hourly_rows_align_to_clock_hours_when_the_start_is_known():
    beats = _hours_of_beats(2.0, 0.5)  # beats at 0, 0.5, ..., 7200.0
    rows = summarize(beats, start_time=datetime(2026, 8, 27, 10, 18, 49)).hourly
    # 10:18:49 is 2471 s before 11:00:00.
    assert [(r.start_sec, r.end_sec) for r in rows] == [(0.0, 2471.0), (2471.0, 6071.0), (6071.0, 7200.0), (7200.0, 7200.0)]
    assert [r.beats for r in rows] == [4942, 7200, 2258, 1]


def test_hourly_rows_from_a_start_on_the_hour_match_the_unaligned_rows():
    beats = _hours_of_beats(1.5, 0.5)
    aligned = summarize(beats, start_time=datetime(2026, 8, 27, 10, 0, 0)).hourly
    assert [(r.start_sec, r.end_sec) for r in aligned] == [(r.start_sec, r.end_sec) for r in summarize(beats).hourly]
```

In `tests/test_pipeline.py`, find the test that runs `run_analysis` on a native fixture with a start time and add an assertion on the hourly labels; if none fits, add:

```python
def test_hourly_rows_align_to_the_recorder_clock(tmp_path, report_text, native_flash_file):
    run_analysis(str(native_flash_file), str(tmp_path), start_time="10:18:49")
    assert "10:18-11:00" in report_text()
```

(Check the fixture names in `tests/test_pipeline.py` first and reuse the one that already produces a report with a known start.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/arrhythmia/test_burden.py -k clock -v`
Expected: TypeError, unexpected keyword `start_time`.

- [ ] **Step 3: Implement**

Replace `hourly_rows` with:

```python
def _row_edges(duration_sec: float, last_beat: float | None, start_time: datetime | None) -> list[tuple[float, float]]:
    """(start, end) of each row. With a known start the first row ends at
    the next clock hour; rows are otherwise HOUR_SEC from the recording
    start. A beat exactly at a duration on the hour still gets its
    (zero-length) row rather than vanishing from the table."""
    end = max(duration_sec, last_beat or 0.0)
    offset = (start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6) if start_time else 0.0
    edges, start, step = [], 0.0, HOUR_SEC - offset
    while start < end or (start == end and last_beat == end):
        edges.append((start, min(start + step, max(end, start))))
        start, step = start + step, HOUR_SEC
    return edges


def hourly_rows(
    beats: list[Beat], duration_sec: float, quality: SignalQuality | None = None,
    start_time: datetime | None = None,
) -> list[HourRow]:
    """Per-hour counts and rates from the recording start to duration_sec;
    see HourRow and _row_edges."""
    if not beats and duration_sec <= 0:
        return []
    centers, window_rr = _windowed_rr(beats)
    runs = pvc_runs(beats)
    rows = []
    for start, end in _row_edges(duration_sec, beats[-1].time if beats else None, start_time):
        stop = end if end > start else start + HOUR_SEC  # the zero-length row still holds its beat
        in_hour = [b for b in beats if start <= b.time < stop] if end > start else [b for b in beats if b.time == start]
        rr = np.array([b.rr_interval for b in in_hour if b.rr_interval])
        sel = (centers >= start) & (centers < stop)
        enough = len(rr) >= HR_EXTREME_WINDOW_BEATS and sel.any()
        hour_runs = [r for r in runs if start <= r[0].time < stop]
        rows.append(HourRow(
            start_sec=float(start),
            end_sec=float(end),
            beats=len(in_hour),
            min_bpm=60.0 / float(window_rr[sel].max()) if enough else None,
            mean_bpm=60.0 / float(rr.mean()) if enough else None,
            max_bpm=60.0 / float(window_rr[sel].min()) if enough else None,
            pvcs=sum(1 for b in in_hour if b.label == "V"),
            couplets=sum(1 for r in hour_runs if len(r) == 2),
            runs=sum(1 for r in hour_runs if len(r) >= MIN_RUN_BEATS),
            pauses=sum(1 for b in in_hour if b.rr_interval and b.rr_interval >= PAUSE_THRESHOLD_SEC),
            analyzed_sec=quality.analyzed_within(start, end) if quality else end - start,
        ))
    return rows
```

Add `from datetime import datetime` at the top of burden.py. Give `summarize` a `start_time: datetime | None = None` parameter (document it: "aligns the hourly rows to clock hours") and pass it: `hourly=hourly_rows(beats, duration_sec, quality, start_time)`. Update the `HourRow` docstring: "One row of the hourly table: a clock hour when the recording start is known (the first and last rows are partial), else an hour from the recording start."

In `pipeline.py`: `summary = summarize(labeled, dog_weight_class=dog_weight_class, quality=quality, start_time=rec.start_time)`.

Simplify: if the existing tests all pass with the plain `[start, end)` membership (the zero-length row taken as `b.time == start`), drop the `stop` variable; keep whichever reads cleaner and passes `test_hourly_rows_keep_a_beat_exactly_on_the_last_boundary`.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/arrhythmia tests/test_pipeline.py -v` — PASS.

- [ ] **Step 5: Commit** `git commit -am "Burden: hourly rows align to clock hours when the start is known"`

### Task 4: Summary panels

**Files:**
- Modify: `src/canine_holter/report/generate.py`
- Test: `tests/report/test_generate.py`

- [ ] **Step 1: Write the failing tests** (replace `test_content_has_four_summary_groups_in_order_and_a_footer`; add the rest)

```python
def test_content_has_six_summary_groups_in_order_and_a_footer():
    beats = _couplet_and_single()
    content = build_content(beats, summarize(beats), None)
    assert [g.title for g in content.summary_groups] == [
        "Recording", "Heart rate", "Ventricular ectopy", "Supraventricular ectopy", "Pauses", "RR variability",
    ]
    assert any("not a diagnosis" in line for line in content.footer_lines)


def test_heart_rate_group_shares_and_event_rules_name_the_thresholds():
    beats = _steady(10, 0.5)  # 120 bpm: 6 windows, none slow or fast for a medium dog
    rows = _rows(build_content(beats, summarize(beats), None), "Heart rate")
    assert rows["Below 50 bpm"] == SummaryRow("Below 50 bpm", "0 (0%)", "5-beat median")
    assert rows["Above 160 bpm"] == SummaryRow("Above 160 bpm", "0 (0%)", "5-beat median")
    assert rows["Brady events"] == SummaryRow("Brady events", "0", "3+ beats < 50 bpm")
    assert rows["Tachy events"] == SummaryRow("Tachy events", "0", "3+ beats > 160 bpm")


def test_heart_rate_shares_show_count_and_percent():
    beats = _steady(10, 0.3)  # 200 bpm: every window is fast
    rows = _rows(build_content(beats, summarize(beats), None), "Heart rate")
    assert rows["Above 160 bpm"].value == "6 (100%)"


def test_supraventricular_group_says_not_assessed():
    beats = _couplet_and_single()
    rows = _rows(build_content(beats, summarize(beats), None), "Supraventricular ectopy")
    assert rows["SVPBs"] == SummaryRow("SVPBs", "not assessed", "needs P-wave analysis")


def test_pause_group_counts_pauses_over_five_seconds():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(3.77, 2.97, "N"), _beat(9.77, 6.0, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "Pauses")
    assert rows["Pauses"].value == "2"
    assert rows["Pauses > 5 s"] == SummaryRow("Pauses > 5 s", "1")
    assert rows["Longest"].status == "alert"


def test_variability_group_rounds_and_counts_nn_intervals():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.7, 0.9, "N"), _beat(2.5, 0.8, "N"), _beat(3.5, 1.0, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "RR variability")
    assert rows["SDNN"] == SummaryRow("SDNN", "83 ms", "4 NN intervals")
    assert rows["RMSSD"] == SummaryRow("RMSSD", "141 ms")
    assert rows["pNN50"] == SummaryRow("pNN50", "100%")


def test_variability_group_says_when_not_computed():
    beats = _steady(3, 0.5)
    rows = _rows(build_content(beats, summarize(beats), None), "RR variability")
    assert rows["RR variability"].value == "not computed (fewer than 2 successive NN differences)"
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/report/test_generate.py -v` — the new tests FAIL (KeyError / group count).

- [ ] **Step 3: Implement** in `generate.py`

Imports from burden: add `LONG_PAUSE_THRESHOLD_SEC, MIN_SUCCESSIVE_DIFFERENCES, SUSTAINED_EVENT_MIN_BEATS`.

```python
def _share(count: int, total: int) -> str:
    return f"{count} ({100.0 * count / total:.0f}%)" if total else "0 (0%)"


def _heart_rate_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    hr = summary.heart_rate
    window = f"{HR_EXTREME_WINDOW_BEATS}-beat median"
    brady, tachy = summary.brady_threshold_bpm, summary.tachy_threshold_bpm
    if hr is None:
        rate_rows = [
            SummaryRow("Heart rate", f"not computed (fewer than {HR_EXTREME_WINDOW_BEATS} beats with an RR)")
        ]
    else:
        rate_rows = [
            SummaryRow("Mean", f"{hr.mean_bpm:.0f} bpm"),
            SummaryRow("Slowest", f"{hr.min_bpm:.0f} bpm at {short_time(hr.min_time, start_time)}", window),
            SummaryRow("Fastest", f"{hr.max_bpm:.0f} bpm at {short_time(hr.max_time, start_time)}", window),
            SummaryRow(f"Below {brady:g} bpm", _share(summary.slow_beats, summary.rated_beats), window),
            SummaryRow(f"Above {tachy:g} bpm", _share(summary.fast_beats, summary.rated_beats), window),
        ]
    return SummaryGroup("Heart rate", rate_rows + [
        SummaryRow("Brady events", str(len(summary.bradycardia_events)), f"{SUSTAINED_EVENT_MIN_BEATS}+ beats < {brady:g} bpm"),
        SummaryRow("Tachy events", str(len(summary.tachycardia_events)), f"{SUSTAINED_EVENT_MIN_BEATS}+ beats > {tachy:g} bpm"),
    ])


def _supraventricular_group() -> SummaryGroup:
    """Stated, not counted: a premature narrow beat cannot be told from
    sinus arrhythmia by timing alone (pNN50 runs ~70 % in a resting dog),
    and P waves are not analyzed. Absent must read as absent, not zero."""
    return SummaryGroup("Supraventricular ectopy", [SummaryRow("SVPBs", "not assessed", "needs P-wave analysis")])


def _pause_group(summary: ArrhythmiaSummary) -> SummaryGroup:
    longest = summary.longest_pause_sec
    return SummaryGroup("Pauses", [
        SummaryRow("Pauses", str(len(summary.pauses)), f">= {PAUSE_THRESHOLD_SEC:g} s"),
        SummaryRow(f"Pauses > {LONG_PAUSE_THRESHOLD_SEC:g} s", str(summary.long_pauses)),
        SummaryRow(
            "Longest", f"{longest:.2f} s" if longest is not None else "n/a", PAUSE_BAND, pause_status(longest)
        ),
    ])


def _variability_group(summary: ArrhythmiaSummary) -> SummaryGroup:
    """Uncoloured: there are no canine reference bands for these."""
    hrv = summary.heart_rate_variability
    if hrv is None:
        rows = [SummaryRow("RR variability", f"not computed (fewer than {MIN_SUCCESSIVE_DIFFERENCES} successive NN differences)")]
    else:
        rows = [
            SummaryRow("SDNN", f"{hrv.sdnn_ms:.0f} ms", f"{hrv.nn_intervals} NN intervals"),
            SummaryRow("RMSSD", f"{hrv.rmssd_ms:.0f} ms"),
            SummaryRow("pNN50", f"{hrv.pnn50_pct:.0f}%"),
        ]
    return SummaryGroup("RR variability", rows)


def summary_groups(summary: ArrhythmiaSummary, start_time: datetime | None) -> list[SummaryGroup]:
    """The six summary panels, in reading order (a 3x2 grid on the page)."""
    return [
        _recording_group(summary, start_time),
        _heart_rate_group(summary, start_time),
        _ectopy_group(summary, start_time),
        _supraventricular_group(),
        _pause_group(summary),
        _variability_group(summary),
    ]
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -v` — PASS (fix any test that counted four groups).

- [ ] **Step 5: Commit** `git commit -am "Report: rate shares, SVPB not assessed, pauses over 5 s, RR variability panels"`

### Task 5: One strip per hour

**Files:**
- Modify: `src/canine_holter/report/generate.py`
- Test: `tests/report/test_generate.py`

- [ ] **Step 1: Write the failing tests**

```python
from canine_holter.report.generate import HOURLY_TITLE, MAX_HOURLY_STRIPS


def test_hourly_strips_come_last_one_per_hour_at_the_hours_first_beat():
    beats = _steady(int(2.5 * 3600 / 0.5), 0.5)  # 2.5 h at 120 bpm
    content = build_content(beats, summarize(beats), datetime(2026, 8, 27, 10, 18, 49))
    section = content.sections[-1]
    assert section.heading == HOURLY_TITLE
    assert [item.caption.title for item in section.items] == [
        "Hour 10:18-11:00 · 10:18:49", "Hour 11:00-12:00 · 11:00:00", "Hour 12:00-12:48 · 12:00:00",
    ]
    assert [item.run[0].time for item in section.items] == [0.0, 2471.0, 6071.0]
    assert section.items[0].caption.what == "The first beats of the hour. This hour: 120-120 bpm, mean 120."
    assert section.items[0].caption.significance == ""


def test_hourly_strips_skip_an_hour_without_beats_and_say_when_rates_are_missing():
    beats = [_beat(0.0, None, "N"), _beat(1.0, 1.0, "N"), _beat(7300.0, None, "N")]
    section = build_content(beats, summarize(beats), None).sections[-1]
    assert [item.caption.title for item in section.items] == ["Hour 0:00-1:00 · t=0s", "Hour 2:00-2:01 · t=7300s"]
    assert section.items[0].caption.what == "The first beats of the hour. Too few beats this hour for a rate."


def test_hourly_strips_are_capped_and_the_heading_says_so():
    beats = _steady(50 * 6, 600.0)  # 50 h, a beat every 10 min
    section = build_content(beats, summarize(beats), None).sections[-1]
    assert len(section.items) == MAX_HOURLY_STRIPS
    assert section.heading == f"{HOURLY_TITLE} ({MAX_HOURLY_STRIPS} of 50 shown, evenly spaced through the recording)"


def test_primer_explains_the_hourly_strips():
    assert any("every hour" in line for line in HOW_TO_READ_STRIPS)
```

- [ ] **Step 2: Run** — FAIL (ImportError on `HOURLY_TITLE`).

- [ ] **Step 3: Implement**

Constants after `ISOLATED_TITLE`:

```python
HOURLY_TITLE = "One strip per hour"
MAX_HOURLY_STRIPS = 48  # two days of hours; stated in the heading when it applies
```

Primer: append to `HOW_TO_READ_STRIPS`:

```python
    "",
    "The last section shows one strip at the start of every hour, so the underlying rhythm can be",
    "checked through the day and night, not only where the software flagged something.",
```

Refactor the heading rule out of `_section` and add the section:

```python
def _capped_heading(title: str, shown: int, total: int) -> str:
    return title if shown == total else f"{title} ({shown} of {total} shown, evenly spaced through the recording)"


def _hour_rate_text(row: HourRow) -> str:
    if row.mean_bpm is None:
        return "Too few beats this hour for a rate."
    return f"This hour: {row.min_bpm:.0f}-{row.max_bpm:.0f} bpm, mean {row.mean_bpm:.0f}."


def _hourly_section(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> StripSection | None:
    """A strip at the first beat of every hour that has one: the underlying
    rhythm, checkable through the day, not only where something was
    flagged (HE/LX prints the same 'one per hour' strips)."""
    rows = [row for row in summary.hourly if row.beats]
    if not rows:
        return None
    shown = select_evenly(rows, MAX_HOURLY_STRIPS)
    items = []
    for row in shown:
        beat = next(b for b in beats if b.time >= row.start_sec)
        items.append(StripItem([beat], StripCaption(
            f"Hour {_hour_label(row, start_time)} · {short_time(beat.time, start_time)}",
            f"The first beats of the hour. {_hour_rate_text(row)}",
            "",
        )))
    return StripSection(_capped_heading(HOURLY_TITLE, len(shown), len(rows)), items)
```

In `_section`, replace the inline heading with `_capped_heading(title, len(shown), len(runs))`. In `build_content`, append `_hourly_section(beats, summary, start_time)` to `sections` and mention it in the docstring ("then one strip per hour").

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -v` — PASS. Existing tests that assert `content.sections[1:]` or the last section (for example `test_extremes_section_comes_first...`, `test_zero_pvcs_yields_no_sections`, the pdf page-count tests) now see the hourly section: update their expectations to include it rather than weakening them.

- [ ] **Step 5: Commit** `git commit -am "Report: one strip per hour"`

### Task 6: Six-panel summary page

**Files:**
- Modify: `src/canine_holter/report/pdf.py`
- Test: `tests/report/test_pdf.py`

- [ ] **Step 1: Write the failing test** (extend `test_summary_page_renders_groups_with_status_colors_and_footer`, or add)

```python
def test_summary_page_lays_six_panels_out_in_three_rows():
    from canine_holter.report.pdf import _PANEL_TOP
    assert len(_PANEL_TOP) == 3
    beats = _couplet_and_single()  # whatever helper the file uses
    fig = pdf._summary_page(build_content(beats, summarize(beats), None))
    titles = [t.get_text() for t in fig.texts if t.get_text().isupper()]
    assert titles == ["RECORDING", "HEART RATE", "VENTRICULAR ECTOPY", "SUPRAVENTRICULAR ECTOPY", "PAUSES", "RR VARIABILITY"]
```

- [ ] **Step 2: Run** — FAIL (`_PANEL_TOP` has two entries).

- [ ] **Step 3: Implement**

```python
_PANEL_TOP = (0.86, 0.68, 0.50)  # top of each panel row; the tallest panel (seven rows) fits the 0.18 pitch
```

Update `_summary_page`'s docstring: "the six panels in a 3x2 grid".

- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all PASS.

- [ ] **Step 5: Commit** `git commit -am "Report: six summary panels in a 3x2 grid"`

### Task 7: Docs and acceptance on the real recording

**Files:**
- Modify: `CLAUDE.md` (architecture bullet for `arrhythmia/burden.py` and `report/`; Known limits)
- Modify: `README.md` if it lists the report's contents

- [ ] **Step 1: Update CLAUDE.md**

`arrhythmia/burden.py` bullet: add "heart-rate variability (SDNN, RMSSD, pNN50 over NN intervals), rate shares, and hourly rows aligned to clock hours when the start is known".
`report/` bullet: add "six summary panels" and "one strip per hour".
Known limits: add "Supraventricular ectopy, AV block, and junctional escape beats are not assessed; they need P-wave analysis. The report says so for SVPBs."

- [ ] **Step 2: Run the full suite with coverage**

Run: `.venv/bin/pytest --cov=canine_holter --cov-report=term-missing -q`
Expected: all pass; no new uncovered lines in burden.py or generate.py.

- [ ] **Step 3: Generate the 08-27 report**

Run: `.venv/bin/canine-holter ~/Downloads/flash2.dat --out ~/Downloads/teeny-holter-2026-08-27-v3 --dog-weight-class large`
Then `pdftotext -layout ~/Downloads/teeny-holter-2026-08-27-v3/report.pdf - | head -60` and check: six panels, SDNN ~529 ms / RMSSD ~492 / pNN50 ~70 %, hourly rows labelled `10:18-11:00`, `11:00-12:00`, ..., and the hourly-strip section present. Open page 1 as an image (`pdftoppm -r 60 -f 1 -l 1 -png`) and confirm nothing overlaps.

- [ ] **Step 4: Commit and open the PR**

```bash
git add -A && git commit -m "Docs: report parity stats"
git push -u origin report-parity-stats
gh pr create --title "Report: HRV, rate shares, clock-hour table, one strip per hour" --body "..."
```
