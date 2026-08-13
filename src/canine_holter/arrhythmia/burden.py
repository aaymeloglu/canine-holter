from dataclasses import dataclass
from canine_holter.types import Beat

# Provisional defaults - not yet calibrated against real canine recordings.
# See docs/superpowers/specs/2026-08-13-pvc-detection-design.md, "Open items".
PAUSE_THRESHOLD_SEC = 2.5
BRADYCARDIA_HR_THRESHOLD = {"small": 60, "medium": 50, "large": 45}
TACHYCARDIA_HR_THRESHOLD = {"small": 180, "medium": 160, "large": 150}
SUSTAINED_EVENT_MIN_BEATS = 3  # consecutive beats needed to call it "sustained"


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


def summarize(beats: list[Beat], dog_weight_class: str = "medium") -> ArrhythmiaSummary:
    """Aggregate a labeled Beat sequence into an ArrhythmiaSummary.

    dog_weight_class: "small", "medium", or "large" - selects brady/tachy
    thresholds. These are provisional defaults; real calibration happens
    against Teeny's own recordings over time (see design spec).
    """
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

    brady_threshold = BRADYCARDIA_HR_THRESHOLD[dog_weight_class]
    tachy_threshold = TACHYCARDIA_HR_THRESHOLD[dog_weight_class]
    bradycardia_events = _sustained_hr_events(beats, brady_threshold, "brady")
    tachycardia_events = _sustained_hr_events(beats, tachy_threshold, "tachy")

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
    )
