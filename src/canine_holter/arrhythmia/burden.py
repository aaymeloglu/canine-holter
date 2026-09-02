from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from canine_holter.quality.gate import SignalQuality
from canine_holter.types import Beat

# Provisional defaults - not yet calibrated against real canine recordings.
# See docs/superpowers/specs/2026-08-13-pvc-detection-design.md, "Open items".
PAUSE_THRESHOLD_SEC = 2.5
LONG_PAUSE_THRESHOLD_SEC = 5.0  # the report's concern line (report/reference.py PAUSE_CONCERN_SEC), counted beside the 2.5 s pauses so a vendor report with a higher pause setting reads side by side
BRADYCARDIA_HR_THRESHOLD = {"small": 60, "medium": 50, "large": 45}
TACHYCARDIA_HR_THRESHOLD = {"small": 180, "medium": 160, "large": 150}
SUSTAINED_EVENT_MIN_BEATS = 3  # consecutive beats needed to call it "sustained"
HR_EXTREME_WINDOW_BEATS = 5  # min/max HR are medians over this many RRs, so one phantom beat can't set them
MIN_RUN_BEATS = 3  # triplets and longer; a couplet has only one within-run RR
HOUR_SEC = 3600.0
MIN_SUCCESSIVE_DIFFERENCES = 2  # under this, variability is one number's noise


@dataclass(frozen=True)
class HeartRateStats:
    """Whole-recording heart rate. min/max are HR_EXTREME_WINDOW_BEATS-beat
    medians, timed at the window's centre beat; mean is over every RR."""
    min_bpm: float
    min_time: float
    mean_bpm: float
    max_bpm: float
    max_time: float


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


@dataclass(frozen=True)
class RunStats:
    """One PVC run of MIN_RUN_BEATS or more: its length, its rate from the
    RR intervals inside the run (the first beat's RR is the coupling
    interval to the preceding beat, not part of the run's rate), and the
    time of its first beat."""
    beats: int
    bpm: float
    start_time: float


@dataclass(frozen=True)
class HourRow:
    """One row of the hourly table: a clock hour when the recording start
    is known (the first and last rows are partial), else an hour from the
    recording start. A beat on the boundary belongs to the row it starts;
    runs and couplets count in the row of their first beat. Rates are None for an hour with fewer than HR_EXTREME_WINDOW_BEATS
    RR intervals; min/max use the same windowed medians as HeartRateStats."""
    start_sec: float
    end_sec: float
    beats: int
    min_bpm: float | None
    mean_bpm: float | None
    max_bpm: float | None
    pvcs: int
    couplets: int
    runs: int
    pauses: int
    analyzed_sec: float  # seconds of the hour not excluded by quality gating
    escapes: int = 0  # ventricular escape beats ("E")


@dataclass(frozen=True)
class ArrhythmiaSummary:
    total_beats: int
    pvc_count: int
    pvc_burden_pct: float
    couplets: int
    triplets: int
    vtach_runs: int
    bradycardia_events: list[tuple[float, float]]
    tachycardia_events: list[tuple[float, float]]
    pauses: list[float]
    longest_pause_sec: float | None = None  # longest RR interval in the recording
    escape_beats: list[float] = field(default_factory=list)  # times of ventricular escape beats ("E"): wide and late, never in a PVC count
    long_pauses: int = 0  # RR intervals over LONG_PAUSE_THRESHOLD_SEC
    slow_beats: int = 0  # HR_EXTREME_WINDOW_BEATS-beat median windows under the bradycardia threshold ...
    fast_beats: int = 0  # ... and over the tachycardia threshold ...
    rated_beats: int = 0  # ... out of this many windows
    brady_threshold_bpm: float = 0.0
    tachy_threshold_bpm: float = 0.0
    heart_rate: HeartRateStats | None = None  # None when too few RRs for a window
    heart_rate_variability: HeartRateVariability | None = None  # None with too few NN intervals
    longest_run: RunStats | None = None  # most beats; earliest on a tie
    fastest_run: RunStats | None = None  # highest bpm; earliest on a tie
    hourly: list[HourRow] = field(default_factory=list)
    duration_sec: float = 0.0  # recording length; the last beat's time when no quality was given
    analyzed_sec: float = 0.0  # duration minus excluded spans
    excluded: tuple[tuple[float, float], ...] = ()  # artifact spans, from SignalQuality
    trimmed_sec: float = 0.0  # off-body tail dropped before analysis; the recorder ran duration_sec + trimmed_sec


def pvc_runs(beats: list[Beat]) -> list[list[Beat]]:
    """Group consecutive PVC-labeled beats into runs. An escape beat ("E")
    ends a run: a run of escape beats is an idioventricular rhythm, not
    ventricular tachycardia, and is not counted here."""
    runs: list[list[Beat]] = []
    current: list[Beat] = []
    for beat in beats:
        if beat.label == "V":
            current.append(beat)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _windowed_rr(beats: list[Beat]) -> tuple[np.ndarray, np.ndarray]:
    """Median RR over each HR_EXTREME_WINDOW_BEATS consecutive RRs, with the
    time of the window's centre beat. Empty with too few RRs."""
    timed = [(b.time, b.rr_interval) for b in beats if b.rr_interval]
    if len(timed) < HR_EXTREME_WINDOW_BEATS:
        return np.array([]), np.array([])
    times = np.array([t for t, _ in timed])
    rr = np.array([r for _, r in timed])
    windows = np.lib.stride_tricks.sliding_window_view(rr, HR_EXTREME_WINDOW_BEATS)
    window_rr = np.median(windows, axis=1)
    centers = times[HR_EXTREME_WINDOW_BEATS // 2 : HR_EXTREME_WINDOW_BEATS // 2 + len(window_rr)]
    return centers, window_rr


def heart_rate_stats(beats: list[Beat]) -> HeartRateStats | None:
    """Min/mean/max heart rate over the recording, or None with fewer than
    HR_EXTREME_WINDOW_BEATS RR intervals."""
    centers, window_rr = _windowed_rr(beats)
    if len(window_rr) == 0:
        return None
    rr = np.array([b.rr_interval for b in beats if b.rr_interval])
    slowest, fastest = int(np.argmax(window_rr)), int(np.argmin(window_rr))
    return HeartRateStats(
        min_bpm=60.0 / float(window_rr[slowest]),
        min_time=float(centers[slowest]),
        mean_bpm=60.0 / float(rr.mean()),
        max_bpm=60.0 / float(window_rr[fastest]),
        max_time=float(centers[fastest]),
    )


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
    diffs = np.array(
        [nn[i] - nn[i - 1] for i in range(1, len(nn)) if nn[i] is not None and nn[i - 1] is not None]
    ) * 1000.0
    if len(diffs) < MIN_SUCCESSIVE_DIFFERENCES:
        return None
    return HeartRateVariability(
        sdnn_ms=float(values.std()),
        rmssd_ms=float(np.sqrt(np.mean(diffs**2))),
        pnn50_pct=float(np.mean(np.abs(diffs) > 50.0) * 100.0),
        nn_intervals=len(values),
    )


def run_stats(beats: list[Beat]) -> list[RunStats]:
    """One RunStats per PVC run of MIN_RUN_BEATS or more, in recording order.
    A run whose beats all lack an RR has no rate and is left out."""
    stats = []
    for run in pvc_runs(beats):
        rrs = [b.rr_interval for b in run[1:] if b.rr_interval]
        if len(run) >= MIN_RUN_BEATS and rrs:
            stats.append(RunStats(beats=len(run), bpm=60.0 / float(np.mean(rrs)), start_time=run[0].time))
    return stats


def _row_edges(
    duration_sec: float, last_beat: float | None, start_time: datetime | None
) -> list[tuple[float, float, float]]:
    """(start, end, stop) of each hourly row: end is where the row ends on
    the page, stop the boundary its beats are counted up to, so the beat
    at the end of a partial last row still belongs to it. With a known
    start the first row ends at the next clock hour; otherwise rows are
    HOUR_SEC from the recording start. A beat exactly at a duration that
    falls on a boundary still gets its (zero-length) row rather than
    vanishing from the table."""
    end = max(duration_sec, last_beat or 0.0)
    offset = start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6 if start_time else 0.0
    edges, start, step = [], 0.0, HOUR_SEC - offset
    while start < end or (start == end and last_beat == end):
        edges.append((start, min(start + step, end), start + step))
        start, step = start + step, HOUR_SEC
    return edges


def hourly_rows(
    beats: list[Beat],
    duration_sec: float,
    quality: SignalQuality | None = None,
    start_time: datetime | None = None,
) -> list[HourRow]:
    """Per-row counts and rates from the recording start to duration_sec;
    see HourRow and _row_edges."""
    if not beats and duration_sec <= 0:
        return []
    centers, window_rr = _windowed_rr(beats)
    runs = pvc_runs(beats)
    rows = []
    for start, end, stop in _row_edges(duration_sec, beats[-1].time if beats else None, start_time):
        in_hour = [b for b in beats if start <= b.time < stop]
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
            escapes=sum(1 for b in in_hour if b.label == "E"),
        ))
    return rows


def _sustained_hr_events(
    beats: list[Beat], threshold_bpm: float, direction: str
) -> list[tuple[float, float]]:
    """Find stretches of >= SUSTAINED_EVENT_MIN_BEATS consecutive beats whose
    instantaneous HR is below (direction="brady") or above (direction="tachy")
    threshold_bpm. Returns (start_time, end_time) for each stretch."""
    events: list[tuple[float, float]] = []
    run_start: float | None = None
    run_len = 0
    prev_time = None

    for beat in beats:
        if beat.rr_interval is None or beat.rr_interval <= 0:
            # An invalid RR ends the run rather than being skipped: a beat
            # without an interval is a quality-gate boundary, and an event
            # must not span one.
            if run_len >= SUSTAINED_EVENT_MIN_BEATS and run_start is not None:
                events.append((run_start, prev_time))
            run_start, run_len = None, 0
            prev_time = beat.time
            continue

        hr = 60.0 / beat.rr_interval
        is_match = hr < threshold_bpm if direction == "brady" else hr > threshold_bpm

        if is_match:
            if run_len == 0:
                run_start = beat.time - beat.rr_interval
            run_len += 1
        else:
            if run_len >= SUSTAINED_EVENT_MIN_BEATS and run_start is not None:
                events.append((run_start, prev_time))
            run_start, run_len = None, 0

        prev_time = beat.time

    if run_len >= SUSTAINED_EVENT_MIN_BEATS and run_start is not None:
        events.append((run_start, prev_time))

    return events


def summarize(
    beats: list[Beat],
    dog_weight_class: str = "medium",
    quality: SignalQuality | None = None,
    start_time: datetime | None = None,
) -> ArrhythmiaSummary:
    """Aggregate a labeled Beat sequence into an ArrhythmiaSummary.

    dog_weight_class: "small", "medium", or "large" - selects brady/tachy
    thresholds. These are provisional defaults; real calibration happens
    against Teeny's own recordings over time (see design spec).

    quality: the recording's SignalQuality. Without it the duration is the
    last beat's time and nothing is excluded (the report-only path).

    start_time: the recording's wall-clock start; when given, the hourly
    rows align to clock hours.
    """
    if quality is not None:
        duration_sec, analyzed_sec, excluded = quality.duration_sec, quality.analyzed_sec, quality.excluded
        trimmed_sec = quality.trimmed_sec
    else:
        duration_sec = beats[-1].time if beats else 0.0
        analyzed_sec, excluded, trimmed_sec = duration_sec, (), 0.0
    total_beats = len(beats)
    pvc_beats = [b for b in beats if b.label == "V"]
    pvc_count = len(pvc_beats)
    pvc_burden_pct = (pvc_count / total_beats * 100) if total_beats else 0.0

    couplets = triplets = vtach_runs = 0
    for run in pvc_runs(beats):
        n = len(run)
        if n == 2:
            couplets += 1
        elif n == 3:
            triplets += 1
        elif n >= 4:
            vtach_runs += 1

    pauses = [b.time for b in beats if b.rr_interval and b.rr_interval >= PAUSE_THRESHOLD_SEC]
    rr_intervals = [b.rr_interval for b in beats if b.rr_interval]
    longest_pause_sec = max(rr_intervals) if rr_intervals else None

    brady_threshold = BRADYCARDIA_HR_THRESHOLD[dog_weight_class]
    tachy_threshold = TACHYCARDIA_HR_THRESHOLD[dog_weight_class]
    bradycardia_events = _sustained_hr_events(beats, brady_threshold, "brady")
    tachycardia_events = _sustained_hr_events(beats, tachy_threshold, "tachy")
    _, window_rr = _windowed_rr(beats)
    window_bpm = 60.0 / window_rr if len(window_rr) else window_rr

    runs = run_stats(beats)  # max() keeps the first of equal keys, i.e. the earliest run
    longest_run = max(runs, key=lambda r: r.beats) if runs else None
    fastest_run = max(runs, key=lambda r: r.bpm) if runs else None

    return ArrhythmiaSummary(
        total_beats=total_beats,
        pvc_count=pvc_count,
        pvc_burden_pct=pvc_burden_pct,
        couplets=couplets,
        triplets=triplets,
        vtach_runs=vtach_runs,
        bradycardia_events=bradycardia_events,
        tachycardia_events=tachycardia_events,
        pauses=pauses,
        longest_pause_sec=longest_pause_sec,
        escape_beats=[b.time for b in beats if b.label == "E"],
        long_pauses=sum(1 for rr in rr_intervals if rr > LONG_PAUSE_THRESHOLD_SEC),
        slow_beats=int(np.sum(window_bpm < brady_threshold)),
        fast_beats=int(np.sum(window_bpm > tachy_threshold)),
        rated_beats=len(window_bpm),
        brady_threshold_bpm=brady_threshold,
        tachy_threshold_bpm=tachy_threshold,
        heart_rate=heart_rate_stats(beats),
        heart_rate_variability=heart_rate_variability(beats),
        longest_run=longest_run,
        fastest_run=fastest_run,
        hourly=hourly_rows(beats, duration_sec, quality, start_time),
        duration_sec=duration_sec,
        analyzed_sec=analyzed_sec,
        excluded=excluded,
        trimmed_sec=trimmed_sec,
    )
