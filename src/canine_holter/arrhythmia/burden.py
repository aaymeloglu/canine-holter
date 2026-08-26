import math
from dataclasses import dataclass, field
import numpy as np
from canine_holter.quality.gate import SignalQuality
from canine_holter.types import Beat

# Provisional defaults - not yet calibrated against real canine recordings.
# See docs/superpowers/specs/2026-08-13-pvc-detection-design.md, "Open items".
PAUSE_THRESHOLD_SEC = 2.5
BRADYCARDIA_HR_THRESHOLD = {"small": 60, "medium": 50, "large": 45}
TACHYCARDIA_HR_THRESHOLD = {"small": 180, "medium": 160, "large": 150}
SUSTAINED_EVENT_MIN_BEATS = 3  # consecutive beats needed to call it "sustained"
HR_EXTREME_WINDOW_BEATS = 5  # min/max HR are medians over this many RRs, so one phantom beat can't set them
MIN_RUN_BEATS = 3  # triplets and longer; a couplet has only one within-run RR
HOUR_SEC = 3600.0


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
    """One hour of the recording, counted from its start (the last row is
    the partial hour to the final beat). A beat on the boundary belongs to
    the hour it starts; runs and couplets count in the hour of their first
    beat. Rates are None for an hour with fewer than HR_EXTREME_WINDOW_BEATS
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
    heart_rate: HeartRateStats | None = None  # None when too few RRs for a window
    longest_run: RunStats | None = None  # most beats; earliest on a tie
    fastest_run: RunStats | None = None  # highest bpm; earliest on a tie
    hourly: list[HourRow] = field(default_factory=list)
    duration_sec: float = 0.0  # recording length; the last beat's time when no quality was given
    analyzed_sec: float = 0.0  # duration minus excluded spans
    excluded: tuple[tuple[float, float], ...] = ()  # artifact spans, from SignalQuality


def pvc_runs(beats: list[Beat]) -> list[list[Beat]]:
    """Group consecutive PVC-labeled beats into runs."""
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


def run_stats(beats: list[Beat]) -> list[RunStats]:
    """One RunStats per PVC run of MIN_RUN_BEATS or more, in recording order.
    A run whose beats all lack an RR has no rate and is left out."""
    stats = []
    for run in pvc_runs(beats):
        rrs = [b.rr_interval for b in run[1:] if b.rr_interval]
        if len(run) >= MIN_RUN_BEATS and rrs:
            stats.append(RunStats(beats=len(run), bpm=60.0 / float(np.mean(rrs)), start_time=run[0].time))
    return stats


def hourly_rows(
    beats: list[Beat], duration_sec: float, quality: SignalQuality | None = None
) -> list[HourRow]:
    """Per-hour counts and rates from the recording start to duration_sec;
    see HourRow. A beat exactly at a duration that falls on the hour still
    gets its (zero-length) row rather than vanishing from the table."""
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
            # A single invalid rr_interval resets the run rather than being
            # skipped over, so one noisy/dropped-beat sample sitting inside a
            # real sustained episode can fragment it into two runs each
            # below SUSTAINED_EVENT_MIN_BEATS - going entirely unflagged.
            # A screening tool should err toward false positives over false
            # negatives here; revisit if this proves to matter on real
            # recordings.
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
    beats: list[Beat], dog_weight_class: str = "medium", quality: SignalQuality | None = None
) -> ArrhythmiaSummary:
    """Aggregate a labeled Beat sequence into an ArrhythmiaSummary.

    dog_weight_class: "small", "medium", or "large" - selects brady/tachy
    thresholds. These are provisional defaults; real calibration happens
    against Teeny's own recordings over time (see design spec).

    quality: the recording's SignalQuality. Without it the duration is the
    last beat's time and nothing is excluded (the report-only path).
    """
    if quality is not None:
        duration_sec, analyzed_sec, excluded = quality.duration_sec, quality.analyzed_sec, quality.excluded
    else:
        duration_sec = beats[-1].time if beats else 0.0
        analyzed_sec, excluded = duration_sec, ()
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
        heart_rate=heart_rate_stats(beats),
        longest_run=longest_run,
        fastest_run=fastest_run,
        hourly=hourly_rows(beats, duration_sec, quality),
        duration_sec=duration_sec,
        analyzed_sec=analyzed_sec,
        excluded=excluded,
    )
