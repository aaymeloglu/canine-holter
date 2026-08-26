# Signal-Quality Gating and Summary Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude non-analyzable stretches of a recording (off-body, lead-off, flat, hookup/removal), report time analyzed per recording and per hour, and redraw the summary page as four colour-coded panels with the reference band beside each value.

**Architecture:** A new `quality/` stage turns samples into a frozen `SignalQuality` (duration + excluded spans); `exclude_beats` applies it to the detector's beats; `summarize` carries duration/analyzed/excluded through `ArrhythmiaSummary`; the report reads only those. The summary page becomes `SummaryGroup`/`SummaryRow` data rendered as a 2x2 grid; the reference-ranges block collapses into inline bands, status colours, and a two-line footer.

**Tech Stack:** Python 3.11, numpy, matplotlib (PdfPages), pytest. Spec: `docs/superpowers/specs/2026-08-26-signal-quality-and-summary-page-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| Create `src/canine_holter/quality/__init__.py` | package marker |
| Create `src/canine_holter/quality/gate.py` | `SignalQuality`, `assess_quality`, `exclude_beats`, the rule constants |
| Create `tests/quality/__init__.py`, `tests/quality/test_gate.py` | rule tests against literal spans |
| Modify `src/canine_holter/arrhythmia/burden.py` | `duration_sec`/`analyzed_sec`/`excluded` on the summary, `analyzed_sec` per hour |
| Modify `src/canine_holter/report/reference.py` | bands as short strings, status rules, footer text; `reference_lines`/`pvc_per_24h_line` removed |
| Modify `src/canine_holter/report/common.py` | `short_time` helper |
| Modify `src/canine_holter/report/generate.py` | `SummaryRow`/`SummaryGroup`, group builders, `Analyzed (min)` column |
| Modify `src/canine_holter/report/pdf.py` | 2x2 panel summary page, excluded-band caption |
| Modify `src/canine_holter/report/timeline.py` | grey hatched bands for excluded spans, axis to recording end |
| Modify `src/canine_holter/pipeline.py` | wire quality between ingest and detection |
| Modify `tests/conftest.py`, `tests/test_pipeline.py`, `tests/test_cli.py`, `tests/report/*`, `tests/arrhythmia/test_burden.py` | new text shape |
| Modify `CLAUDE.md`, `README.md`, `docs/superpowers/specs/2026-08-24-phantom-beats-pvc-strips-reference-ranges-design.md` | quality stage in the architecture, colour-coding is expected, old non-goal superseded |

---

### Task 1: `SignalQuality` and `assess_quality`

**Files:**
- Create: `src/canine_holter/quality/__init__.py`
- Create: `src/canine_holter/quality/gate.py`
- Create: `tests/quality/__init__.py`
- Create: `tests/quality/test_gate.py`

- [ ] **Step 1: Write the failing tests**

`tests/quality/__init__.py` is empty. `tests/quality/test_gate.py`:

```python
"""Quality gating against literal expected spans. The synthetic signal is a
1.5 Hz sine at 100 Hz (peak-to-peak 2.0 in every 5 s window); artifact is
injected by scaling, flattening, or shrinking stretches of it."""
import numpy as np
import pytest
from canine_holter.quality.gate import SignalQuality, assess_quality
from canine_holter.types import Beat

FS = 100.0


def _sine(seconds):
    t = np.arange(0, seconds, 1 / FS)
    return np.sin(2 * np.pi * 1.5 * t)


def _at(x, start_sec, end_sec):
    return slice(int(start_sec * FS), int(end_sec * FS))


def test_clean_recording_excludes_only_the_first_and_last_minute():
    q = assess_quality(_sine(600), FS)
    assert q.duration_sec == 600.0
    assert q.excluded == ((0.0, 60.0), (540.0, 600.0))
    assert q.analyzed_sec == 480.0


def test_high_amplitude_burst_is_excluded_with_two_second_padding():
    x = _sine(600)
    x[_at(x, 200, 210)] *= 10  # 20 peak-to-peak vs a median of 2
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (198.0, 212.0), (540.0, 600.0))


def test_flat_stretch_is_excluded():
    x = _sine(600)
    x[_at(x, 300, 320)] = 0.7
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (298.0, 322.0), (540.0, 600.0))


def test_low_amplitude_stretch_is_excluded():
    x = _sine(600)
    x[_at(x, 400, 410)] *= 0.01  # 0.02 peak-to-peak, under 0.1x the median
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (398.0, 412.0), (540.0, 600.0))


def test_bursts_within_bridge_sec_form_one_span():
    x = _sine(600)
    x[_at(x, 200, 205)] *= 10
    x[_at(x, 230, 235)] *= 10  # 25 s gap: bridged
    assert assess_quality(x, FS).excluded == ((0.0, 60.0), (198.0, 237.0), (540.0, 600.0))


def test_bursts_further_apart_than_bridge_sec_stay_separate():
    x = _sine(600)
    x[_at(x, 200, 205)] *= 10
    x[_at(x, 245, 250)] *= 10  # 40 s gap: not bridged
    assert assess_quality(x, FS).excluded == (
        (0.0, 60.0), (198.0, 207.0), (243.0, 252.0), (540.0, 600.0)
    )


def test_burst_near_the_edge_merges_into_the_edge_span():
    x = _sine(600)
    x[_at(x, 80, 85)] *= 10  # 20 s after the first minute: bridged into it
    assert assess_quality(x, FS).excluded == ((0.0, 87.0), (540.0, 600.0))


def test_all_zero_recording_is_fully_excluded():
    q = assess_quality(np.zeros(60000), FS)
    assert q.excluded == ((0.0, 600.0),)
    assert q.analyzed_sec == 0.0


def test_recording_shorter_than_two_edge_minutes_is_fully_excluded():
    assert assess_quality(_sine(30), FS).excluded == ((0.0, 30.0),)


def test_empty_recording():
    q = assess_quality(np.array([]), FS)
    assert (q.duration_sec, q.excluded, q.analyzed_sec) == (0.0, (), 0.0)


def test_analyzed_within_subtracts_overlap_with_excluded_spans():
    q = SignalQuality(600.0, ((0.0, 60.0), (198.0, 212.0), (540.0, 600.0)))
    assert q.analyzed_within(0.0, 60.0) == 0.0
    assert q.analyzed_within(100.0, 150.0) == 50.0
    assert q.analyzed_within(180.0, 240.0) == pytest.approx(46.0)
    assert q.analyzed_within(500.0, 600.0) == 40.0


def test_contains_is_inclusive_at_span_edges():
    q = SignalQuality(100.0, ((10.0, 20.0),))
    assert q.contains(10.0) and q.contains(20.0) and q.contains(15.0)
    assert not q.contains(9.99) and not q.contains(20.01)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/quality/test_gate.py -q`
Expected: ImportError / ModuleNotFoundError for `canine_holter.quality.gate`.

- [ ] **Step 3: Implement**

`src/canine_holter/quality/__init__.py`:

```python
"""Signal-quality gating: which stretches of a recording are analyzable ECG."""
```

`src/canine_holter/quality/gate.py`:

```python
"""Decide which stretches of a recording are analyzable ECG and which are
artifact (off-body, lead-off, saturation, flat line, hookup and removal),
so nothing downstream counts a beat, pause, or run inside them.

The rules are amplitude and flat-line only, judged per window against the
recording's own median (DR200 samples carry a decoder DC offset and gain
varies by recorder and lead). Kurtosis- and spectrum-based noise measures
were tested and rejected: they exclude ventricular flutter and VT, which
are near-sinusoidal like noise, and those are exactly what this tool must
keep. Evidence and rejected rules:
docs/superpowers/specs/2026-08-26-signal-quality-and-summary-page-design.md.
"""
from dataclasses import dataclass, replace
import numpy as np
from canine_holter.types import Beat

WINDOW_SEC = 5.0
MAX_AMPLITUDE_RATIO = 4.0  # window peak-to-peak over this multiple of the median: off-body swings, saturation, gross motion
MIN_AMPLITUDE_RATIO = 0.1  # under this multiple: lead-off, flat line at a rail
MAX_FLAT_FRACTION = 0.5  # more than this share of zero sample-to-sample steps: flat line
EDGE_SEC = 60.0  # hookup and removal; the HE/LX vendor software calls the first and last minute artifact unconditionally
BRIDGE_SEC = 30.0  # excluded windows this close are one span: quiet stretches inside an off-body tail are not ECG either
PAD_SEC = 2.0  # beats right at a span's edge are half-buried in noise


@dataclass(frozen=True)
class SignalQuality:
    """duration_sec: length of the recording. excluded: (start, end) seconds
    of artifact, sorted, non-overlapping, clipped to the recording."""
    duration_sec: float
    excluded: tuple[tuple[float, float], ...]

    @property
    def analyzed_sec(self) -> float:
        return self.duration_sec - sum(end - start for start, end in self.excluded)

    def analyzed_within(self, start: float, end: float) -> float:
        """Seconds of [start, end) not excluded."""
        total = max(0.0, end - start)
        for s, e in self.excluded:
            total -= max(0.0, min(e, end) - max(s, start))
        return total

    def contains(self, t: float) -> bool:
        return any(s <= t <= e for s, e in self.excluded)


def assess_quality(samples: np.ndarray, sample_rate: float) -> SignalQuality:
    """Judge the recording in WINDOW_SEC windows; see the module docstring
    for the rules. A recording with no signal in any window (zero median
    peak-to-peak) is excluded whole rather than analyzed as flat."""
    duration = len(samples) / sample_rate
    if duration == 0:
        return SignalQuality(0.0, ())
    window = int(WINDOW_SEC * sample_rate)
    n = len(samples) // window
    if n == 0:  # shorter than one window: the edge rule covers all of it
        return SignalQuality(duration, ((0.0, duration),))
    windows = samples[: n * window].reshape(n, window)
    ptp = windows.max(axis=1) - windows.min(axis=1)
    flat = np.mean(np.diff(windows, axis=1) == 0, axis=1)
    median = float(np.median(ptp))
    if median <= 0:
        return SignalQuality(duration, ((0.0, duration),))
    bad = (ptp > MAX_AMPLITUDE_RATIO * median) | (ptp < MIN_AMPLITUDE_RATIO * median) | (flat > MAX_FLAT_FRACTION)
    spans = [[i * WINDOW_SEC, (i + 1) * WINDOW_SEC] for i in np.flatnonzero(bad)]
    spans.append([0.0, EDGE_SEC])
    spans.append([duration - EDGE_SEC, duration])
    return SignalQuality(duration, _bridge_and_pad(spans, duration))


def _bridge_and_pad(spans: list[list[float]], duration: float) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= BRIDGE_SEC:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    padded = [(max(0.0, s - PAD_SEC), min(duration, e + PAD_SEC)) for s, e in merged]
    # Padding can only overlap neighbours closer than 2 * PAD_SEC, and those
    # were bridged already, so the result is still sorted and disjoint.
    return tuple(padded)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/quality/test_gate.py -q`
Expected: all pass. If `test_recording_shorter_than_two_edge_minutes_is_fully_excluded` fails with a negative start, the `duration - EDGE_SEC` span has start < 0: the sort + pad clip handles it via `max(0.0, ...)`; check `_bridge_and_pad` sorts before bridging.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/quality tests/quality
git commit -m "Quality gate: SignalQuality and assess_quality with amplitude, flat, and edge rules"
```

---

### Task 2: `exclude_beats`

**Files:**
- Modify: `src/canine_holter/quality/gate.py`
- Modify: `tests/quality/test_gate.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/quality/test_gate.py`)

```python
from canine_holter.quality.gate import exclude_beats


def _beats(times, rr=0.5):
    return [Beat(time=t, rr_interval=None if i == 0 else rr, qrs_duration=0.06, label=None) for i, t in enumerate(times)]


def test_exclude_beats_drops_beats_inside_spans_and_resets_the_next_rr():
    beats = _beats([9.0, 9.5, 10.0, 12.0, 20.0, 20.5, 21.0])
    kept = exclude_beats(beats, SignalQuality(30.0, ((10.0, 20.0),)))
    assert [b.time for b in kept] == [9.0, 9.5, 20.5, 21.0]
    assert [b.rr_interval for b in kept] == [None, 0.5, None, 0.5]


def test_exclude_beats_keeps_everything_with_no_spans():
    beats = _beats([1.0, 1.5, 2.0])
    assert exclude_beats(beats, SignalQuality(30.0, ())) == beats


def test_exclude_beats_handles_consecutive_spans():
    beats = _beats([5.0, 5.5, 12.0, 25.0, 25.5, 40.0, 40.5])
    kept = exclude_beats(beats, SignalQuality(50.0, ((6.0, 20.0), (30.0, 35.0))))
    assert [(b.time, b.rr_interval) for b in kept] == [(5.0, None), (5.5, 0.5), (25.0, None), (25.5, 0.5), (40.0, None), (40.5, 0.5)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/quality/test_gate.py -q -k exclude_beats`
Expected: ImportError for `exclude_beats`.

- [ ] **Step 3: Implement** (append to `gate.py`)

```python
def exclude_beats(beats: list[Beat], quality: SignalQuality) -> list[Beat]:
    """Drop beats inside excluded spans. The first beat after each span
    gets rr_interval=None - the contract's "no previous beat" - so a span
    can never read as a pause, a run, or a sustained brady/tachy event."""
    kept: list[Beat] = []
    after_gap = False
    for beat in beats:
        if quality.contains(beat.time):
            after_gap = True
            continue
        if after_gap:
            beat = replace(beat, rr_interval=None)
            after_gap = False
        kept.append(beat)
    return kept
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/quality -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/quality/gate.py tests/quality/test_gate.py
git commit -m "Quality gate: exclude_beats drops beats in artifact spans and resets the RR after each"
```

---

### Task 3: Duration, analyzed time, and excluded spans in the summary

**Files:**
- Modify: `src/canine_holter/arrhythmia/burden.py`
- Modify: `tests/arrhythmia/test_burden.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/arrhythmia/test_burden.py`; `_steady` and `_hours_of_beats` already exist there)

```python
from canine_holter.quality.gate import SignalQuality


def test_summary_without_quality_uses_the_last_beat_as_duration_and_excludes_nothing():
    s = summarize(_hours_of_beats(2.5, 0.5))
    assert (s.duration_sec, s.analyzed_sec, s.excluded) == (9000.0, 9000.0, ())


def test_summary_with_quality_carries_duration_analyzed_and_excluded():
    q = SignalQuality(10000.0, ((0.0, 60.0), (9500.0, 10000.0)))
    s = summarize(_hours_of_beats(2.5, 0.5), quality=q)
    assert (s.duration_sec, s.analyzed_sec, s.excluded) == (10000.0, 9440.0, q.excluded)


def test_hourly_rows_run_to_the_recording_end_not_the_last_beat():
    q = SignalQuality(10000.0, ((9000.0, 10000.0),))
    rows = summarize(_hours_of_beats(2.5, 0.5), quality=q).hourly  # beats end at 9000
    assert [(r.start_sec, r.end_sec) for r in rows] == [(0.0, 3600.0), (3600.0, 7200.0), (7200.0, 10000.0)]
    assert [r.analyzed_sec for r in rows] == [3600.0, 3600.0, 1800.0]


def test_hourly_rows_analyzed_equals_the_hour_without_quality():
    rows = summarize(_hours_of_beats(2.5, 0.5)).hourly
    assert [r.analyzed_sec for r in rows] == [3600.0, 3600.0, 1800.0]


def test_hourly_rows_keep_a_beat_exactly_on_the_last_boundary():
    """Duration on the hour with a beat right at it: the beat's hour is listed."""
    beats = _hours_of_beats(2.0, 0.5)  # last beat at 7200.0
    rows = summarize(beats, quality=SignalQuality(7200.0, ())).hourly
    assert [(r.start_sec, r.end_sec, r.beats) for r in rows] == [(0.0, 3600.0, 7200), (3600.0, 7200.0, 7200), (7200.0, 7200.0, 1)]


def test_summary_empty_beats_with_quality_still_reports_duration():
    s = summarize([], quality=SignalQuality(120.0, ((0.0, 120.0),)))
    assert (s.duration_sec, s.analyzed_sec, s.total_beats) == (120.0, 0.0, 0)
    assert [(r.start_sec, r.end_sec, r.beats, r.analyzed_sec) for r in s.hourly] == [(0.0, 120.0, 0, 0.0)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/arrhythmia/test_burden.py -q -k "quality or recording_end or analyzed or boundary"`
Expected: TypeError (`summarize() got an unexpected keyword argument 'quality'`) / AttributeError.

- [ ] **Step 3: Implement**

In `burden.py`, add the import and fields:

```python
import math
from canine_holter.quality.gate import SignalQuality
```

`HourRow` gains a last field:

```python
    pauses: int
    analyzed_sec: float  # seconds of the hour not excluded by quality gating
```

`ArrhythmiaSummary` gains three fields after `hourly`:

```python
    hourly: list[HourRow] = field(default_factory=list)
    duration_sec: float = 0.0  # recording length; the last beat's time when no quality was given
    analyzed_sec: float = 0.0  # duration minus excluded spans
    excluded: tuple[tuple[float, float], ...] = ()  # artifact spans, from SignalQuality
```

Replace `hourly_rows`:

```python
def hourly_rows(beats: list[Beat], duration_sec: float, quality: SignalQuality | None = None) -> list[HourRow]:
    """Per-hour counts and rates from the recording start to duration_sec;
    see HourRow. A beat exactly at a duration that falls on the hour still
    gets its (empty-length) row rather than vanishing from the table."""
    if not beats and duration_sec <= 0:
        return []
    n_hours = math.ceil(duration_sec / HOUR_SEC) if duration_sec > 0 else 0
    if beats:
        n_hours = max(n_hours, int(beats[-1].time // HOUR_SEC) + 1)
    centers, window_rr = _windowed_rr(beats)
    runs = pvc_runs(beats)
    rows = []
    for hour in range(n_hours):
        start = hour * HOUR_SEC
        end = min(start + HOUR_SEC, max(duration_sec, start))
        in_hour = [b for b in beats if start <= b.time < start + HOUR_SEC]
        rr = np.array([b.rr_interval for b in in_hour if b.rr_interval])
        sel = (centers >= start) & (centers < start + HOUR_SEC)
        enough = len(rr) >= HR_EXTREME_WINDOW_BEATS and sel.any()
        hour_runs = [r for r in runs if start <= r[0].time < start + HOUR_SEC]
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

In `summarize`, change the signature and the tail:

```python
def summarize(
    beats: list[Beat], dog_weight_class: str = "medium", quality: SignalQuality | None = None
) -> ArrhythmiaSummary:
    """Aggregate a labeled Beat sequence into an ArrhythmiaSummary.

    dog_weight_class: "small", "medium", or "large" - selects brady/tachy
    thresholds. These are provisional defaults; real calibration happens
    against Teeny's own recordings over time (see design spec).

    quality: the recording's SignalQuality. Without it the duration is the
    last beat's time and nothing is excluded (the report-only path).
    """
    ...
    if quality is not None:
        duration_sec, analyzed_sec, excluded = quality.duration_sec, quality.analyzed_sec, quality.excluded
    else:
        duration_sec = beats[-1].time if beats else 0.0
        analyzed_sec, excluded = duration_sec, ()

    return ArrhythmiaSummary(
        ...existing fields...,
        hourly=hourly_rows(beats, duration_sec, quality),
        duration_sec=duration_sec,
        analyzed_sec=analyzed_sec,
        excluded=excluded,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/arrhythmia -q`
Expected: all pass, including the pre-existing hourly tests (same rows as before when no quality is given: the last beat at 9000 gives `ceil(9000/3600)=3` rows with the third ending at 9000).

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/arrhythmia/burden.py tests/arrhythmia/test_burden.py
git commit -m "Summary carries duration, analyzed time, and excluded spans; hours run to the recording end"
```

---

### Task 4: Reference bands as short strings, status rules, footer

**Files:**
- Modify: `src/canine_holter/report/reference.py`
- Modify: `tests/report/test_reference.py`

- [ ] **Step 1: Replace `tests/report/test_reference.py`**

```python
"""Reference bands and the status each value gets. The bands come from the
ESVC Doberman DCM screening guidelines (Wess et al. 2017); the tests pin
the literal boundaries so a wording edit cannot silently move a band."""
from canine_holter.report.reference import (
    FOOTER_LINES,
    PAUSE_BAND,
    PVC_24H_BAND,
    RUN_RATE_BAND,
    analyzed_status,
    count_status,
    pause_status,
    pvc_24h_status,
    pvc_per_24h,
    run_rate_status,
)

H = 3600.0


def test_pvc_per_24h_not_computed_under_20_analyzed_hours():
    assert pvc_per_24h(65, 2.5 * H) is None
    assert pvc_per_24h(65, 19.9 * H) is None


def test_pvc_per_24h_equals_count_for_24_analyzed_hours():
    assert pvc_per_24h(120, 24 * H) == 120


def test_pvc_per_24h_scales_48_analyzed_hours_down():
    assert pvc_per_24h(120, 48 * H) == 60


def test_pvc_24h_status_bands():
    assert pvc_24h_status(49) == "ok"
    assert pvc_24h_status(50) == "caution"
    assert pvc_24h_status(300) == "caution"
    assert pvc_24h_status(301) == "alert"


def test_count_status_is_ok_only_at_zero():
    assert count_status(0) == "ok"
    assert count_status(1) == "alert"


def test_run_rate_status():
    assert run_rate_status(None) == "ok"
    assert run_rate_status(179.9) == "caution"
    assert run_rate_status(180.0) == "alert"


def test_pause_status_bands():
    assert pause_status(None) == "ok"
    assert pause_status(2.49) == "ok"
    assert pause_status(2.5) == "caution"
    assert pause_status(5.0) == "caution"
    assert pause_status(5.01) == "alert"


def test_analyzed_status_needs_20_hours():
    assert analyzed_status(20 * H) == "ok"
    assert analyzed_status(19.99 * H) == "caution"


def test_band_strings_and_footer_carry_the_guideline_values_and_source():
    assert PVC_24H_BAND == "<50 | 50-300 | >300"
    assert PAUSE_BAND == "<2.5 | 2.5-5 | >5 s"
    assert RUN_RATE_BAND == "<180 bpm"
    text = "\n".join(FOOTER_LINES)
    assert "Wess" in text and "2017" in text
    assert "not a diagnosis" in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/report/test_reference.py -q`
Expected: ImportError for the new names.

- [ ] **Step 3: Replace `src/canine_holter/report/reference.py`**

```python
"""Published reference bands printed beside the report's numbers, and the
status (ok / caution / alert) each value gets from them.

PVC bands per 24 h: ESVC screening guidelines for dilated cardiomyopathy
in Doberman Pinschers (Wess et al., J Vet Cardiol 2017): under 50 normal,
50-300 equivocal (repeat within the year), over 300 abnormal; any couplet,
triplet, or run is abnormal. Run rate: ~180 bpm is the usual canine line
between ventricular tachycardia and the less concerning accelerated
idioventricular rhythm. Pauses: canine Holter studies of healthy dogs find
pauses over 2.5 s common with sinus arrhythmia; ~5 s is the usual line for
concern.
"""

PVC_24H_NORMAL_MAX = 50
PVC_24H_EQUIVOCAL_MAX = 300
VT_MIN_BPM = 180
PAUSE_COMMON_MAX_SEC = 2.5
PAUSE_CONCERN_SEC = 5.0
MIN_HOURS_FOR_24H_SCALING = 20  # PVC frequency varies across a day; don't scale a short recording
_SEC_PER_DAY = 24 * 3600.0

PVC_24H_BAND = f"<{PVC_24H_NORMAL_MAX} | {PVC_24H_NORMAL_MAX}-{PVC_24H_EQUIVOCAL_MAX} | >{PVC_24H_EQUIVOCAL_MAX}"
PAUSE_BAND = f"<{PAUSE_COMMON_MAX_SEC} | {PAUSE_COMMON_MAX_SEC}-{PAUSE_CONCERN_SEC:g} | >{PAUSE_CONCERN_SEC:g} s"
RUN_RATE_BAND = f"<{VT_MIN_BPM} bpm"
COUNT_BAND = "0"
ANALYZED_BAND = f">= {MIN_HOURS_FOR_24H_SCALING} h"

FOOTER_LINES = [
    "Colours compare each value with the band printed beside it: green inside the normal band,"
    " amber in the equivocal band, red beyond it. They are not a diagnosis.",
    "Bands: ESVC Doberman DCM screening guidelines (Wess et al., J Vet Cardiol 2017);"
    " pause and run-rate context from canine Holter studies.",
]


def format_duration(duration_sec: float) -> str:
    hours, rem = divmod(int(duration_sec), 3600)
    return f"{hours}h {rem // 60}m"


def pvc_per_24h(pvc_count: int, analyzed_sec: float) -> float | None:
    """PVC count scaled to 24 h of analyzed time, or None when fewer than
    MIN_HOURS_FOR_24H_SCALING hours were analyzed."""
    if analyzed_sec < MIN_HOURS_FOR_24H_SCALING * 3600:
        return None
    return pvc_count * _SEC_PER_DAY / analyzed_sec


def pvc_24h_status(scaled: float) -> str:
    if scaled < PVC_24H_NORMAL_MAX:
        return "ok"
    return "caution" if scaled <= PVC_24H_EQUIVOCAL_MAX else "alert"


def count_status(n: int) -> str:
    return "ok" if n == 0 else "alert"


def run_rate_status(bpm: float | None) -> str:
    if bpm is None:
        return "ok"
    return "alert" if bpm >= VT_MIN_BPM else "caution"


def pause_status(longest_sec: float | None) -> str:
    if longest_sec is None or longest_sec < PAUSE_COMMON_MAX_SEC:
        return "ok"
    return "caution" if longest_sec <= PAUSE_CONCERN_SEC else "alert"


def analyzed_status(analyzed_sec: float) -> str:
    return "ok" if analyzed_sec >= MIN_HOURS_FOR_24H_SCALING * 3600 else "caution"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/report/test_reference.py -q`
Expected: pass. (`tests/report/test_generate.py` and the pipeline tests now fail on the removed `reference_lines`; Task 5 fixes them.)

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/report/reference.py tests/report/test_reference.py
git commit -m "Reference bands as inline strings with ok/caution/alert status rules"
```

---

### Task 5: Summary groups in `ReportContent`

**Files:**
- Modify: `src/canine_holter/report/common.py`
- Modify: `src/canine_holter/report/generate.py`
- Modify: `tests/report/test_common.py` (one test), `tests/report/test_generate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/report/test_common.py`:

```python
from datetime import datetime
from canine_holter.report.common import short_time


def test_short_time_is_clock_with_start_and_elapsed_without():
    assert short_time(8183.2, datetime(2026, 8, 23, 15, 33, 8)) == "17:49:31"
    assert short_time(8183.2, None) == "t=8183s"
```

In `tests/report/test_generate.py`, replace `test_content_summary_has_the_stats` with these (keep the existing `_beat` and `_couplet_and_single` helpers; add the imports at the top):

```python
from canine_holter.quality.gate import SignalQuality
from canine_holter.report.generate import SummaryRow


def _rows(content, title):
    group = next(g for g in content.summary_groups if g.title == title)
    return {r.label: r for r in group.rows}


def test_content_has_four_summary_groups_in_order():
    beats = _couplet_and_single()
    content = build_content(beats, summarize(beats), None)
    assert [g.title for g in content.summary_groups] == ["Recording", "Heart rate", "Ventricular ectopy", "Pauses"]
    assert content.footer_lines  # legend + source


def test_recording_group_reports_duration_analyzed_and_excluded():
    beats = _couplet_and_single()
    q = SignalQuality(7200.0, ((0.0, 60.0), (7140.0, 7200.0)))
    rows = _rows(build_content(beats, summarize(beats, quality=q), datetime(2026, 8, 23, 15, 33, 8)), "Recording")
    assert rows["Start"].value == "2026-08-23 15:33:08"
    assert rows["Duration"].value == "2h 0m"
    assert rows["Analyzed"] == SummaryRow("Analyzed", "1h 58m (98%)", ">= 20 h", "caution")
    assert rows["Excluded"] == SummaryRow("Excluded", "0h 2m", "artifact / off-body")
    assert rows["Total beats"].value == "7"


def test_ectopy_group_values_references_and_statuses():
    beats = _couplet_and_single()
    rows = _rows(build_content(beats, summarize(beats), None), "Ventricular ectopy")
    assert rows["PVCs"] == SummaryRow("PVCs", "3 (42.86%)")
    assert rows["PVCs per 24 h"] == SummaryRow("PVCs per 24 h", "n/a", "needs >= 20 h analyzed")
    assert rows["Couplets"] == SummaryRow("Couplets", "1", "0", "alert")
    assert rows["Triplets"] == SummaryRow("Triplets", "0", "0", "ok")
    assert rows["VT runs (4+)"] == SummaryRow("VT runs (4+)", "0", "0", "ok")
    assert rows["Longest run"] == SummaryRow("Longest run", "none")
    assert rows["Fastest run"] == SummaryRow("Fastest run", "none", "<180 bpm", "ok")


def test_ectopy_group_scales_pvcs_by_analyzed_time_and_colours_the_band():
    beats = [_beat(i * 0.5, 0.5 if i else None, "V" if i % 100 == 0 else "N") for i in range(0, 24 * 3600 * 2)]
    q = SignalQuality(24 * 3600.0, ((0.0, 4 * 3600.0),))  # 20 h analyzed
    rows = _rows(build_content(beats, summarize(beats, quality=q), None), "Ventricular ectopy")
    assert rows["PVCs per 24 h"].reference == "<50 | 50-300 | >300"
    assert rows["PVCs per 24 h"].value == "2074 (scaled from 20h 0m analyzed)"
    assert rows["PVCs per 24 h"].status == "alert"


def test_run_rows_show_beats_rate_and_time():
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(10)]
    for i in (4, 5, 6, 7):
        beats[i] = _beat(beats[i].time, 0.3, "V")
    rows = _rows(build_content(beats, summarize(beats), datetime(2026, 8, 23, 15, 33, 8)), "Ventricular ectopy")
    assert rows["Longest run"].value == "4 beats, 200 bpm, 15:33:11"
    assert rows["Fastest run"] == SummaryRow("Fastest run", "4 beats, 200 bpm, 15:33:11", "<180 bpm", "alert")


def test_heart_rate_and_pause_groups():
    beats = [_beat(i * 0.5, 0.5 if i else None, "N") for i in range(20)]
    beats[10] = _beat(beats[10].time, 3.0, "N")
    content = build_content(beats, summarize(beats), None)
    hr = _rows(content, "Heart rate")
    assert hr["Mean"].value == "111 bpm"
    assert hr["Slowest"].value.endswith("bpm at t=5s") or hr["Slowest"].value.endswith("bpm at t=4s")
    assert hr["Brady events"].status is None and hr["Tachy events"].status is None
    pauses = _rows(content, "Pauses")
    assert pauses["Pauses >= 2.5 s"] == SummaryRow("Pauses >= 2.5 s", "1")
    assert pauses["Longest"] == SummaryRow("Longest", "3.00 s", "<2.5 | 2.5-5 | >5 s", "caution")


def test_hourly_header_and_rows_carry_analyzed_minutes():
    beats = _couplet_and_single()
    q = SignalQuality(4000.0, ((0.0, 60.0), (3940.0, 4000.0)))
    content = build_content(beats, summarize(beats, quality=q), None)
    assert content.hourly_header[:2] == ["Hour", "Analyzed (min)"]
    assert [row[1] for row in content.hourly_rows] == ["59.0", "5.7"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/report/test_common.py tests/report/test_generate.py -q`
Expected: ImportError (`short_time`, `SummaryRow`) and failures on `summary_groups`.

- [ ] **Step 3: Implement**

Append to `common.py`:

```python
def short_time(elapsed_sec: float, start_time: datetime | None) -> str:
    """Clock time when the recording start is known, else elapsed seconds.
    For summary cells, where format_time's combined form is too wide."""
    if start_time is None:
        return f"t={elapsed_sec:.0f}s"
    return (start_time + timedelta(seconds=elapsed_sec)).strftime("%H:%M:%S")
```

Rewrite the top half of `generate.py` (imports, dataclasses, `HOURLY_HEADER`, `_hourly_rows`, and everything down to `_section`); `_section`, `_beat_at`, `_pause_beats`, `_extremes_section`, and `write_report` stay as they are except where noted:

```python
"""Turns labeled beats + summary into the report content (plain data) and
writes it as report.pdf - the only output file."""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import (
    HR_EXTREME_WINDOW_BEATS,
    MIN_RUN_BEATS,
    PAUSE_THRESHOLD_SEC,
    ArrhythmiaSummary,
    HourRow,
    RunStats,
    pvc_runs,
)
from canine_holter.report.common import (
    EVENTS_TITLE,
    EXTREMES_TITLE,
    ISOLATED_TITLE,
    MAX_STRIPS_PER_SECTION,
    event_line,
    flagged_runs,
    format_time,
    isolated_pvcs,
    pvc_line,
    section_heading,
    select_evenly,
    short_time,
)
from canine_holter.report.pdf import write_pdf
from canine_holter.report.reference import (
    ANALYZED_BAND,
    COUNT_BAND,
    FOOTER_LINES,
    MIN_HOURS_FOR_24H_SCALING,
    PAUSE_BAND,
    PVC_24H_BAND,
    RUN_RATE_BAND,
    analyzed_status,
    count_status,
    format_duration,
    pause_status,
    pvc_24h_status,
    pvc_per_24h,
    run_rate_status,
)


@dataclass(frozen=True)
class SummaryRow:
    """One line of a summary panel: the value, the published band it is
    compared with (short text beside it), and where it falls."""
    label: str
    value: str
    reference: str = ""
    status: str | None = None  # "ok" | "caution" | "alert" | None (uncoloured)


@dataclass(frozen=True)
class SummaryGroup:
    title: str
    rows: list[SummaryRow]


@dataclass(frozen=True)
class StripSection:
    """One section of rhythm strips: a heading, the PVC runs shown (already
    capped), and one label per run."""
    heading: str
    runs: list[list[Beat]]
    labels: list[str]


@dataclass(frozen=True)
class ReportContent:
    """Everything textual in the report, independent of how it is rendered."""
    summary_groups: list[SummaryGroup]
    footer_lines: list[str]
    sections: list[StripSection]
    hourly_header: list[str]
    hourly_rows: list[list[str]]  # one row of cells per HourRow, in HOURLY_HEADER order


HOURLY_HEADER = [
    "Hour", "Analyzed (min)", "Beats", "Min HR", "Mean HR", "Max HR", "PVCs", "Couplets",
    f"Runs ({MIN_RUN_BEATS}+)", "Pauses",
]


def _hour_label(row: HourRow, start_time: datetime | None) -> str:
    """'15:33-16:33' with a known start, else elapsed 'h:mm-h:mm'."""
    def clock(sec: float) -> str:
        if start_time is not None:
            return (start_time + timedelta(seconds=sec)).strftime("%H:%M")
        return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}"
    return f"{clock(row.start_sec)}-{clock(row.end_sec)}"


def _hourly_rows(summary: ArrhythmiaSummary, start_time: datetime | None) -> list[list[str]]:
    def bpm(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"
    return [
        [
            _hour_label(row, start_time),
            f"{row.analyzed_sec / 60:.1f}",
            str(row.beats),
            bpm(row.min_bpm),
            bpm(row.mean_bpm),
            bpm(row.max_bpm),
            str(row.pvcs),
            str(row.couplets),
            str(row.runs),
            str(row.pauses),
        ]
        for row in summary.hourly
    ]


def _recording_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    duration, analyzed = summary.duration_sec, summary.analyzed_sec
    pct = 100.0 * analyzed / duration if duration else 0.0
    return SummaryGroup("Recording", [
        SummaryRow("Start", start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "unknown"),
        SummaryRow("Duration", format_duration(duration)),
        SummaryRow("Analyzed", f"{format_duration(analyzed)} ({pct:.0f}%)", ANALYZED_BAND, analyzed_status(analyzed)),
        SummaryRow("Excluded", format_duration(duration - analyzed), "artifact / off-body"),
        SummaryRow("Total beats", str(summary.total_beats)),
    ])


def _heart_rate_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    hr = summary.heart_rate
    if hr is None:
        rate_rows = [SummaryRow("Heart rate", f"not computed (fewer than {HR_EXTREME_WINDOW_BEATS} beats with an RR)")]
    else:
        rate_rows = [
            SummaryRow("Mean", f"{hr.mean_bpm:.0f} bpm"),
            SummaryRow("Slowest", f"{hr.min_bpm:.0f} bpm at {short_time(hr.min_time, start_time)}", f"{HR_EXTREME_WINDOW_BEATS}-beat median"),
            SummaryRow("Fastest", f"{hr.max_bpm:.0f} bpm at {short_time(hr.max_time, start_time)}", f"{HR_EXTREME_WINDOW_BEATS}-beat median"),
        ]
    return SummaryGroup("Heart rate", rate_rows + [
        SummaryRow("Brady events", str(len(summary.bradycardia_events))),
        SummaryRow("Tachy events", str(len(summary.tachycardia_events))),
    ])


def _run_text(run: RunStats, start_time: datetime | None) -> str:
    return f"{run.beats} beats, {run.bpm:.0f} bpm, {short_time(run.start_time, start_time)}"


def _pvc_24h_row(summary: ArrhythmiaSummary) -> SummaryRow:
    scaled = pvc_per_24h(summary.pvc_count, summary.analyzed_sec)
    if scaled is None:
        return SummaryRow("PVCs per 24 h", "n/a", f"needs >= {MIN_HOURS_FOR_24H_SCALING} h analyzed")
    value = f"{round(scaled)} (scaled from {format_duration(summary.analyzed_sec)} analyzed)"
    return SummaryRow("PVCs per 24 h", value, PVC_24H_BAND, pvc_24h_status(scaled))


def _ectopy_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    longest, fastest = summary.longest_run, summary.fastest_run
    return SummaryGroup("Ventricular ectopy", [
        SummaryRow("PVCs", f"{summary.pvc_count} ({summary.pvc_burden_pct:.2f}%)"),
        _pvc_24h_row(summary),
        SummaryRow("Couplets", str(summary.couplets), COUNT_BAND, count_status(summary.couplets)),
        SummaryRow("Triplets", str(summary.triplets), COUNT_BAND, count_status(summary.triplets)),
        SummaryRow("VT runs (4+)", str(summary.vtach_runs), COUNT_BAND, count_status(summary.vtach_runs)),
        SummaryRow("Longest run", _run_text(longest, start_time) if longest else "none"),
        SummaryRow(
            "Fastest run",
            _run_text(fastest, start_time) if fastest else "none",
            RUN_RATE_BAND,
            run_rate_status(fastest.bpm if fastest else None),
        ),
    ])


def _pause_group(summary: ArrhythmiaSummary) -> SummaryGroup:
    longest = summary.longest_pause_sec
    return SummaryGroup("Pauses", [
        SummaryRow(f"Pauses >= {PAUSE_THRESHOLD_SEC:g} s", str(len(summary.pauses))),
        SummaryRow("Longest", f"{longest:.2f} s" if longest is not None else "n/a", PAUSE_BAND, pause_status(longest)),
    ])


def summary_groups(summary: ArrhythmiaSummary, start_time: datetime | None) -> list[SummaryGroup]:
    return [
        _recording_group(summary, start_time),
        _heart_rate_group(summary, start_time),
        _ectopy_group(summary, start_time),
        _pause_group(summary),
    ]
```

`_extremes_section` uses `_run_text` for the fastest-run label; its label text changes to match (the test in `test_generate.py` that asserts the extremes labels, if any, needs the new form `"Fastest run: 4 beats, 200 bpm, 15:33:11"`). Update `build_content`:

```python
def build_content(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> ReportContent:
    """Assemble the report content: summary panels, the strip sections
    (heart-rate extremes, then flagged multi-beat runs, then isolated PVCs;
    the latter two capped at MAX_STRIPS_PER_SECTION with the cap stated in
    the heading), and the hourly table. Event times are wall-clock labels
    when start_time is known."""
    sections = [
        _extremes_section(beats, summary, start_time),
        _section(EVENTS_TITLE, flagged_runs(beats), event_line, start_time),
        _section(ISOLATED_TITLE, isolated_pvcs(beats), pvc_line, start_time),
    ]
    return ReportContent(
        summary_groups=summary_groups(summary, start_time),
        footer_lines=list(FOOTER_LINES),
        sections=[s for s in sections if s is not None],
        hourly_header=HOURLY_HEADER,
        hourly_rows=_hourly_rows(summary, start_time),
    )
```

Delete `_heart_rate_lines`, `_run_line`, `_summary_lines`, and the `duration_sec = beats[-1].time` line (the summary now knows its duration). Keep `_pause_beats` and `_extremes_section`; in `_extremes_section` the pause label still uses `format_time`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/report/test_common.py tests/report/test_generate.py -q`
Expected: the new tests pass; `test_pdf.py` fails until Task 6 (`_summary_page` signature). Fix any extremes-label test to the new `_run_text` wording.

- [ ] **Step 5: Commit**

```bash
git add src/canine_holter/report/common.py src/canine_holter/report/generate.py tests/report/test_common.py tests/report/test_generate.py
git commit -m "Report content: four summary groups with inline bands and statuses; analyzed minutes per hour"
```

---

### Task 6: PDF summary page as a 2x2 grid; timeline bands

**Files:**
- Modify: `src/canine_holter/report/pdf.py`
- Modify: `src/canine_holter/report/timeline.py`
- Modify: `tests/report/test_timeline.py`, `tests/report/test_pdf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/report/test_timeline.py`:

```python
def test_excluded_spans_are_drawn_as_bands_across_the_heart_rate_panel():
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(300)]
    summary = _summary(duration_sec=400.0, excluded=((0.0, 60.0), (340.0, 400.0)))
    fig = plt.figure(figsize=(12, 5))
    ax_hr, ax_ev = draw_timeline(fig, GridSpec(1, 1, figure=fig)[0], beats, summary, None)
    assert len(ax_hr.patches) == 2
    assert len(ax_ev.patches) == 2
    assert ax_ev.get_xlim()[1] >= 400.0 / 60  # axis reaches the recording end, not the last beat
    plt.close(fig)
```

Append to `tests/report/test_pdf.py`:

```python
def test_summary_page_renders_groups_with_statuses_and_footer():
    beats = _beats_with_couplets(1)
    summary = summarize(beats)
    content = build_content(beats, summary, None)
    fig = _summary_page(content.summary_groups, content.footer_lines)
    texts = [t.get_text() for t in fig.texts]
    assert "RECORDING" in texts and "VENTRICULAR ECTOPY" in texts
    assert "Couplets" in texts and "1" in texts
    coloured = {t.get_text(): t.get_color() for t in fig.texts if t.get_color() in STATUS_COLORS.values()}
    assert coloured["1"] == STATUS_COLORS["alert"]
    assert any("not a diagnosis" in t for t in texts)
    plt.close(fig)
```

with `from canine_holter.report.pdf import STATUS_COLORS, _summary_page, write_pdf` and `import matplotlib.pyplot as plt` at the top.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/report/test_timeline.py tests/report/test_pdf.py -q`
Expected: `_summary()` TypeError on `duration_sec`... no: `ArrhythmiaSummary` already has the fields from Task 3, so the timeline test fails on `patches` count 0; the pdf test fails on ImportError `STATUS_COLORS`.

- [ ] **Step 3: Implement the timeline bands**

In `timeline.py`, add a constant and use it in `draw_timeline` after the axes are created, and extend `_recording_span_sec`:

```python
EXCLUDED_COLOR = "#9e9d99"


def _recording_span_sec(beats: list[Beat], summary: ArrhythmiaSummary) -> float:
    """End of the recording: its duration when known, else the last beat or
    event. Floored at one HR bin so an empty recording still has a
    non-degenerate axis."""
    ends = [summary.duration_sec] + [b.time for b in beats] + list(summary.pauses)
    ends += [end for _, end in summary.bradycardia_events + summary.tachycardia_events]
    return max([HR_BIN_SEC] + ends)
```

and, right after `ax_hr.tick_params(labelbottom=False)`:

```python
    for start, end in summary.excluded:
        for ax in (ax_hr, ax_ev):
            ax.axvspan(to_x(start), to_x(end), facecolor=EXCLUDED_COLOR, alpha=0.25, hatch="///", edgecolor=EXCLUDED_COLOR, linewidth=0)
```

- [ ] **Step 4: Implement the summary page**

In `pdf.py` replace `_summary_page` and add the constants:

```python
STATUS_COLORS = {"ok": "#2e7d32", "caution": "#b26a00", "alert": "#c62828"}
LABEL_COLOR = "#52514e"
REFERENCE_COLOR = "#6f6e6b"
_PANEL_X = (_LEFT, 0.53)  # left edge of each panel column
_PANEL_W = 0.42
_VALUE_DX = 0.12  # value column offset inside a panel
_PANEL_TOP = (0.80, 0.60)  # top of each panel row


def _draw_group(fig: Figure, x: float, y: float, group: SummaryGroup) -> float:
    """One panel: title, then label / value / reference per row. Returns the
    y below the last row."""
    fig.text(x, y, group.title.upper(), va="top", fontsize=9, fontweight="bold")
    y -= 0.025
    for row in group.rows:
        fig.text(x, y, row.label, va="top", fontsize=9, color=LABEL_COLOR)
        fig.text(
            x + _VALUE_DX, y, row.value, va="top", fontsize=9,
            color=STATUS_COLORS.get(row.status, "black"),
            fontweight="bold" if row.status else "normal",
        )
        if row.reference:
            fig.text(x + _PANEL_W, y, row.reference, va="top", ha="right", fontsize=7.5, color=REFERENCE_COLOR)
        y -= _LINE_STEP
    return y


def _summary_page(groups: list[SummaryGroup], footer_lines: list[str]) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    y = 0.95
    fig.text(_LEFT, y, REPORT_TITLE, va="top", fontsize=16, fontweight="bold")
    y -= 0.035
    fig.text(_LEFT, y, DISCLAIMER, va="top", fontsize=10, fontstyle="italic")
    bottom = 1.0
    for i, group in enumerate(groups):
        x = _PANEL_X[i % 2]
        top = _PANEL_TOP[i // 2]
        bottom = min(bottom, _draw_group(fig, x, top, group))
    _text_block(fig, bottom - 0.02, footer_lines, fontsize=8, color=REFERENCE_COLOR)
    return fig
```

Add `SummaryGroup` to the `TYPE_CHECKING` import from `generate`. In `write_pdf` call `_summary_page(content.summary_groups, content.footer_lines)`. In `_timeline_page`, after `draw_timeline(...)`, add the caption when anything was excluded:

```python
    if summary.excluded:
        fig.text(_LEFT, 0.62, "Hatched grey bands: excluded from analysis (artifact / off-body).", va="top", fontsize=8, color="#6f6e6b")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/report -q`
Expected: all pass. Page counts in `test_pdf.py` are unchanged (summary page is still one page).

- [ ] **Step 6: Commit**

```bash
git add src/canine_holter/report/pdf.py src/canine_holter/report/timeline.py tests/report/test_pdf.py tests/report/test_timeline.py
git commit -m "PDF: 2x2 summary panels with status colours; timeline shades excluded spans"
```

---

### Task 7: Wire quality into the pipeline; end-to-end tests

**Files:**
- Modify: `src/canine_holter/pipeline.py`
- Modify: `tests/conftest.py`, `tests/test_pipeline.py`, `tests/test_cli.py`

- [ ] **Step 1: Update the `report_text` fixture** in `tests/conftest.py`:

```python
    def spy(out_path, *, content, **kw):
        lines = []
        for group in content.summary_groups:
            lines.append(group.title)
            for row in group.rows:
                lines.append(f"{row.label}: {row.value}" + (f" ({row.reference})" if row.reference else ""))
        captured["text"] = "\n".join(
            lines
            + content.footer_lines
            + [line for s in content.sections for line in [s.heading, *s.labels]]
            + [" | ".join(row) for row in [content.hourly_header, *content.hourly_rows]]
        )
        return real(out_path, content=content, **kw)
```

- [ ] **Step 2: Update the end-to-end assertions**

In `tests/test_pipeline.py`: `r"PVC count:\s*(\d+)"` becomes `r"PVCs:\s*(\d+)"` (both places, and the message strings), and `"- Recording start: 2026-08-23 15:36:00"` becomes `"Start: 2026-08-23 15:36:00"` (two places). In `tests/test_cli.py` the same `Start:` change. Add to `tests/test_pipeline.py`:

```python
def test_run_analysis_reports_analyzed_time_and_edge_exclusion(tmp_path, report_text):
    """The synthetic 25 s flash recording is shorter than the edge minutes,
    so it is excluded whole: the report says so instead of counting beats."""
    flash_path = tmp_path / "flash.dat"
    _write_synthetic_flash(flash_path)
    run_analysis(str(flash_path), str(tmp_path / "out"))
    content = report_text()
    assert re.search(r"Analyzed:\s*0h 0m \(0%\)", content), content
    assert re.search(r"Total beats:\s*0\b", content), content
```

and change `test_run_analysis_end_to_end_on_native_flash` to assert `Total beats: 0` too? No: keep that test meaningful by making its recording long enough to have analyzable time. Change `_write_synthetic_flash` to accept `seconds` (default 25) and, in that test, write 180 s (`n_blocks` scales with samples: read the helper, it builds a spike train from a sample count; multiply). Then the existing `>= 20` beats assertion holds (60 s analyzed at 120 bpm gives ~120 beats).

- [ ] **Step 3: Run to verify they fail**

Run: `pytest tests/test_pipeline.py tests/test_cli.py -q`
Expected: the new test fails (`Analyzed` missing) and `MIT-BIH 119` plausibility still passes only after wiring (it will fail on `PVCs:` until Task 5's labels exist - they do).

- [ ] **Step 4: Wire the pipeline**

```python
from canine_holter.quality.gate import assess_quality, exclude_beats
...
    rec = load_recording(input_path)
    if isinstance(start_time, str):
        start_time = parse_start_time(start_time, rec.start_time)
    if start_time is not None:
        rec = replace(rec, start_time=start_time)
    quality = assess_quality(rec.samples, rec.sample_rate)
    beats = exclude_beats(detect_beats(rec.samples, rec.sample_rate), quality)
    labeled = classify_beats(beats)
    summary = summarize(labeled, dog_weight_class=dog_weight_class, quality=quality)
```

Docstring: add "Signal that fails quality gating (artifact, off-body, the first and last minute) is excluded before detection results are used; the report states the analyzed time."

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: all pass. Note the MIT-BIH 119 fixture is 60 s: the edge rule excludes it whole, so `test_run_analysis_report_contains_plausible_stats_for_known_fixture` will report 0 beats and fail its range check. Resolve by lowering `EDGE_SEC` only if the vendor convention is wrong - it is not - so instead build that test's expectation from a longer signal: tile the fixture's samples? No: the loader reads the fixture. Accept the fixture's limit by making the plausibility test use `assess_quality`-free stages directly (`detect_beats` -> `classify_beats` -> `summarize` -> `build_content`) with the same ranges, and keep `run_analysis` covered by the 180 s synthetic flash test. Update the test accordingly and its docstring ("run_analysis excludes the first and last minute; this 60 s fixture is checked one stage below it").

- [ ] **Step 6: Commit**

```bash
git add src/canine_holter/pipeline.py tests/conftest.py tests/test_pipeline.py tests/test_cli.py
git commit -m "Pipeline: gate signal quality before detection; report analyzed time end to end"
```

---

### Task 8: Docs and the retired "never normal or abnormal" rule

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/superpowers/specs/2026-08-24-phantom-beats-pvc-strips-reference-ranges-design.md`

- [ ] **Step 1: CLAUDE.md**

In "What this is", after the disclaimer sentence, replace "Do not add language anywhere (reports, README, docstrings) that implies diagnostic authority." with: "Comparing numbers with published reference bands - including colour-coding them - is expected; the disclaimer is what carries the not-a-diagnosis framing, and it must stay."

In "Architecture", the flow becomes:

```
ingest → quality → detection → classify → arrhythmia → report
         (loader picks the format)  (exclude_beats)   (+ pipeline wires it, cli/gui drive it)
```

and add a bullet after `ingest/`: "`quality/gate.py` - `assess_quality()`: samples in, `SignalQuality` (duration + excluded artifact spans) out. Amplitude and flat-line rules per 5 s window against the recording's median, plus the first and last minute (hookup/removal, the HE/LX convention). Kurtosis/spectral noise rules were tested and rejected because they exclude ventricular flutter/VT; see the 2026-08-26 spec before adding a noise rule. `exclude_beats()` drops beats inside spans and resets the RR after each, so a span can never read as a pause or run. `summarize()` takes the `SignalQuality` and reports `duration_sec`/`analyzed_sec`/`excluded`; PVCs per 24 h scale by analyzed time."

Update the `arrhythmia/burden.py` and `report/` bullets: hourly rows include `analyzed_sec`; the report's page 1 is four summary panels (`SummaryGroup`/`SummaryRow`, value + inline band + ok/caution/alert status from `reference.py`) with a two-line footer, the timeline shades excluded spans.

Add to "Known limits": "Quality gating catches severe artifact (off-body swings, flat line, lead-off) and the edge minutes; moderate noise with readable QRS and mid-recording hash noise at normal amplitude are not excluded. Beat detection misses beats during tachycardia (~150 bpm on Teeny's 2026-08-23 recording, 130-150 min), which shows up as false pauses - open item in `detection/`."

- [ ] **Step 2: README.md**

Replace the report paragraph's reference-range sentence with: "Page 1 is four panels (recording, heart rate, ventricular ectopy, pauses); each value sits beside the ESVC Doberman screening band it is compared with and is coloured green / amber / red by where it falls. The recording panel states how much of the recording was analyzed: the first and last minute and any off-body, flat, or saturated stretches are excluded, and the timeline shades them."

- [ ] **Step 3: Old spec**

In `2026-08-24-phantom-beats-pvc-strips-reference-ranges-design.md`, under Non-goals, change the wording bullet to: "~~Any wording that implies a diagnosis...~~ Superseded 2026-08-26: values are colour-coded against the published bands; see `2026-08-26-signal-quality-and-summary-page-design.md`."

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md docs/superpowers/specs/2026-08-24-phantom-beats-pvc-strips-reference-ranges-design.md
git commit -m "Docs: quality stage, colour-coded bands, retire the never-normal-or-abnormal rule"
```

---

### Task 9: Manual validation on Teeny's recording

- [ ] **Step 1: Run the CLI**

```bash
canine-holter samples/teeny-2026-08-23/flash.dat --out ~/Downloads/teeny-holter-2026-08-23-v2/
```

- [ ] **Step 2: Check**

Open the PDF (Read tool, pages 1-2). Expect: Analyzed about 2h 21m of 2h 28m; Excluded spans = the first minute, ~142.9-146.8 min bridged into the last minute (one band at the end); PVCs 13 -> fewer (the tail's phantom PVCs gone); pauses 69 -> ~28-30 (the tail's gone; 130-140 min misses remain, see spec follow-up); heart-rate Slowest no longer 11 bpm at 18:00:39 inside the tail.

- [ ] **Step 3: Record the numbers** in the PR body (before/after table) and in the spec's Evidence section if they differ from the plan's expectations.

---

## Self-review

- Spec coverage: rules (T1), edge rule (T1), exclude_beats (T2), summary/hour fields (T3), scaling by analyzed time (T4/T5), groups + statuses + footer (T4/T5/T6), timeline bands and hourly column (T5/T6), pipeline (T7), fixture + e2e (T7), docs and rule retirement (T8), manual validation (T9). The spec's "remainder window" clause is dropped (redundant with the edge rule; spec updated).
- Type consistency: `SignalQuality(duration_sec, excluded)`; `summarize(beats, dog_weight_class, quality)`; `hourly_rows(beats, duration_sec, quality)`; `SummaryRow(label, value, reference, status)`; `ReportContent(summary_groups, footer_lines, sections, hourly_header, hourly_rows)`; `_summary_page(groups, footer_lines)`.
