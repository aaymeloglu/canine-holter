"""Turns labeled beats + summary into the report content (plain data) and
writes it as report.pdf - the only output file."""
import os
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.common import (
    EVENTS_TITLE,
    ISOLATED_TITLE,
    MAX_STRIPS_PER_SECTION,
    event_line,
    flagged_runs,
    isolated_pvcs,
    pvc_line,
    section_heading,
    select_evenly,
)
from canine_holter.report.pdf import write_pdf
from canine_holter.report.reference import format_duration, pvc_per_24h_line, reference_lines


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
    summary_lines: list[str]
    reference_lines: list[str]
    sections: list[StripSection]


def _summary_lines(summary: ArrhythmiaSummary, start_time: datetime | None, duration_sec: float) -> list[str]:
    """The Summary bullet lines."""
    start_text = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "unknown"
    longest = (
        f"{summary.longest_pause_sec:.2f} s" if summary.longest_pause_sec is not None else "n/a"
    )
    return [
        f"- Recording start: {start_text}",
        f"- Duration: {format_duration(duration_sec)}",
        f"- Total beats: {summary.total_beats}",
        f"- PVC count: {summary.pvc_count}",
        f"- PVC burden: {summary.pvc_burden_pct:.2f}%",
        pvc_per_24h_line(summary.pvc_count, duration_sec),
        f"- Couplets: {summary.couplets}",
        f"- Triplets: {summary.triplets}",
        f"- VT runs (4+ consecutive PVCs): {summary.vtach_runs}",
        f"- Pauses (>= threshold): {len(summary.pauses)}",
        f"- Longest pause: {longest}",
        f"- Sustained bradycardia events: {len(summary.bradycardia_events)}",
        f"- Sustained tachycardia events: {len(summary.tachycardia_events)}",
    ]


def _section(title: str, runs: list[list[Beat]], labeler, start_time: datetime | None) -> StripSection | None:
    if not runs:
        return None
    shown = select_evenly(runs, MAX_STRIPS_PER_SECTION)
    return StripSection(
        heading=section_heading(title, len(shown), len(runs)),
        runs=shown,
        labels=[labeler(i, run, start_time) for i, run in enumerate(shown)],
    )


def build_content(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> ReportContent:
    """Assemble the report text: summary, reference ranges, and the strip
    sections (flagged multi-beat runs first, then isolated PVCs), each
    capped at MAX_STRIPS_PER_SECTION with the cap stated in its heading.
    Event times are wall-clock labels when start_time is known."""
    # The last beat is the only end-of-recording marker available on every
    # path (no samples on the report-only path); it is within seconds of the
    # true end.
    duration_sec = beats[-1].time if beats else 0.0
    sections = [
        _section(EVENTS_TITLE, flagged_runs(beats), event_line, start_time),
        _section(ISOLATED_TITLE, isolated_pvcs(beats), pvc_line, start_time),
    ]
    return ReportContent(
        summary_lines=_summary_lines(summary, start_time, duration_sec),
        reference_lines=reference_lines(duration_sec),
        sections=[s for s in sections if s is not None],
    )


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
    start_time: datetime | None = None,
) -> str:
    """Write report.pdf into out_dir and return its path. Nothing else is
    written. Without waveform samples the strips are listed as text."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")
    write_pdf(
        pdf_path,
        content=build_content(beats, summary, start_time),
        beats=beats,
        summary=summary,
        start_time=start_time,
        samples=samples,
        sample_rate=sample_rate,
    )
    return pdf_path
