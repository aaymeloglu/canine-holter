"""Text and flagged-event helpers shared by the markdown and PDF reports."""
from datetime import datetime, timedelta
from typing import TypeVar
from canine_holter.arrhythmia.burden import pvc_runs
from canine_holter.types import Beat

REPORT_TITLE = "Holter Analysis Report"
DISCLAIMER = "This is a screening aid, not a diagnosis. Review with a veterinary cardiologist."
MAX_STRIPS_PER_SECTION = 24  # 8 PDF pages at 3 strips/page; never a silent cap

T = TypeVar("T")


def format_time(elapsed_sec: float, start_time: datetime | None) -> str:
    """Render an elapsed-seconds offset as a wall-clock label when the
    recording start is known, falling back to elapsed seconds otherwise."""
    elapsed = f"t={elapsed_sec:.1f}s"
    if start_time is None:
        return elapsed
    clock = (start_time + timedelta(seconds=elapsed_sec)).strftime("%H:%M:%S")
    return f"{clock} ({elapsed})"


def flagged_runs(beats: list[Beat]) -> list[list[Beat]]:
    """PVC runs of 2+ beats (couplets, triplets, VT runs)."""
    return [run for run in pvc_runs(beats) if len(run) >= 2]


def isolated_pvcs(beats: list[Beat]) -> list[list[Beat]]:
    """Single PVCs, as one-beat runs so they share the strip machinery."""
    return [run for run in pvc_runs(beats) if len(run) == 1]


def select_evenly(items: list[T], max_n: int) -> list[T]:
    """All of items if there are at most max_n, else max_n of them spread
    evenly from first to last - a fair sample of a long recording rather
    than its first few minutes."""
    if len(items) <= max_n:
        return list(items)
    step = (len(items) - 1) / (max_n - 1)
    return [items[round(i * step)] for i in range(max_n)]


EXTREMES_TITLE = "Heart-rate extremes, longest pause, fastest run"
EVENTS_TITLE = "Flagged events (couplets, triplets, VT runs)"
ISOLATED_TITLE = "Isolated PVCs"


def section_heading(title: str, shown: int, total: int) -> str:
    """Section title, with the cap spelled out whenever it applied."""
    if shown == total:
        return title
    return f"{title} ({shown} of {total} shown, evenly spaced through the recording)"


def run_center_time(run: list[Beat]) -> float:
    return (run[0].time + run[-1].time) / 2


def event_line(index: int, run: list[Beat], start_time: datetime | None) -> str:
    """'Event N: k consecutive PVCs at ~<time>' - one line per flagged run."""
    label = format_time(run_center_time(run), start_time)
    return f"Event {index + 1}: {len(run)} consecutive PVCs at ~{label}"


def pvc_line(index: int, run: list[Beat], start_time: datetime | None) -> str:
    """'PVC N: isolated PVC at ~<time>' - one line per plotted single PVC."""
    return f"PVC {index + 1}: isolated PVC at ~{format_time(run_center_time(run), start_time)}"


def short_time(elapsed_sec: float, start_time: datetime | None) -> str:
    """Clock time when the recording start is known, else elapsed seconds.
    For summary cells, where format_time's combined form is too wide."""
    if start_time is None:
        return f"t={elapsed_sec:.0f}s"
    return (start_time + timedelta(seconds=elapsed_sec)).strftime("%H:%M:%S")
