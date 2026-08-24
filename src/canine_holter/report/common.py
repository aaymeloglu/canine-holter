"""Text and flagged-event helpers shared by the markdown and PDF reports."""
from datetime import datetime, timedelta
from canine_holter.arrhythmia.burden import pvc_runs
from canine_holter.types import Beat

REPORT_TITLE = "Holter Analysis Report"
DISCLAIMER = "This is a screening aid, not a diagnosis. Review with a veterinary cardiologist."


def format_time(elapsed_sec: float, start_time: datetime | None) -> str:
    """Render an elapsed-seconds offset as a wall-clock label when the
    recording start is known, falling back to elapsed seconds otherwise."""
    elapsed = f"t={elapsed_sec:.1f}s"
    if start_time is None:
        return elapsed
    clock = (start_time + timedelta(seconds=elapsed_sec)).strftime("%H:%M:%S")
    return f"{clock} ({elapsed})"


def flagged_runs(beats: list[Beat]) -> list[list[Beat]]:
    """PVC runs of 2+ beats (couplets, triplets, VT runs) - the events worth
    a rhythm-strip plot. Isolated single PVCs are counted in the summary
    stats but not individually plotted, to avoid an unbounded number of
    plots on a high-burden recording."""
    return [run for run in pvc_runs(beats) if len(run) >= 2]


def run_center_time(run: list[Beat]) -> float:
    return (run[0].time + run[-1].time) / 2


def event_line(index: int, run: list[Beat], start_time: datetime | None) -> str:
    """'Event N: k consecutive PVCs at ~<time>' - one line per flagged run."""
    label = format_time(run_center_time(run), start_time)
    return f"Event {index + 1}: {len(run)} consecutive PVCs at ~{label}"
