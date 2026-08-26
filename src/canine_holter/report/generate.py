"""Turns labeled beats + summary into the report content (plain data) and
writes it as report.pdf - the only output file."""
import os
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import (
    HR_EXTREME_WINDOW_BEATS,
    MIN_RUN_BEATS,
    ArrhythmiaSummary,
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


def _heart_rate_lines(summary: ArrhythmiaSummary, start_time: datetime | None) -> list[str]:
    hr = summary.heart_rate
    if hr is None:
        return [f"- Heart rate: not computed (fewer than {HR_EXTREME_WINDOW_BEATS} beats with an RR interval)"]
    window = f"({HR_EXTREME_WINDOW_BEATS}-beat median)"
    return [
        f"- Mean heart rate: {hr.mean_bpm:.0f} bpm",
        f"- Slowest heart rate {window}: {hr.min_bpm:.0f} bpm at {format_time(hr.min_time, start_time)}",
        f"- Fastest heart rate {window}: {hr.max_bpm:.0f} bpm at {format_time(hr.max_time, start_time)}",
    ]


def _run_text(run: RunStats, start_time: datetime | None) -> str:
    return f"{run.beats} PVCs at {run.bpm:.0f} bpm, starting {format_time(run.start_time, start_time)}"


def _run_line(name: str, run: RunStats | None, start_time: datetime | None) -> str:
    if run is None:
        return f"- {name}: none (no runs of {MIN_RUN_BEATS}+ PVCs)"
    return f"- {name}: {_run_text(run, start_time)}"


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
        *_heart_rate_lines(summary, start_time),
        f"- PVC count: {summary.pvc_count}",
        f"- PVC burden: {summary.pvc_burden_pct:.2f}%",
        pvc_per_24h_line(summary.pvc_count, duration_sec),
        f"- Couplets: {summary.couplets}",
        f"- Triplets: {summary.triplets}",
        f"- VT runs (4+ consecutive PVCs): {summary.vtach_runs}",
        _run_line("Longest run", summary.longest_run, start_time),
        _run_line("Fastest run", summary.fastest_run, start_time),
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


def _beat_at(beats: list[Beat], time: float) -> Beat:
    return next(b for b in beats if b.time == time)


def _pause_beats(beats: list[Beat], summary: ArrhythmiaSummary) -> list[Beat]:
    """The two beats bracketing the longest pause, so the strip is centred on
    the gap and marks both edges."""
    end = next(i for i, b in enumerate(beats) if b.rr_interval == summary.longest_pause_sec)
    return beats[max(0, end - 1) : end + 1]


def _extremes_section(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> StripSection | None:
    """Strips a reader looks at first: fastest and slowest heart rate, the
    longest pause (when one crossed the threshold), and the fastest run (when
    there is one). Absent without heart-rate stats, since a recording that
    short has nothing to show."""
    hr = summary.heart_rate
    if hr is None:
        return None
    runs = [[_beat_at(beats, hr.max_time)], [_beat_at(beats, hr.min_time)]]
    labels = [
        f"Fastest heart rate: {hr.max_bpm:.0f} bpm at {format_time(hr.max_time, start_time)}",
        f"Slowest heart rate: {hr.min_bpm:.0f} bpm at {format_time(hr.min_time, start_time)}",
    ]
    if summary.pauses:
        pause = _pause_beats(beats, summary)
        runs.append(pause)
        labels.append(
            f"Longest pause: {summary.longest_pause_sec:.2f} s, ending at {format_time(pause[-1].time, start_time)}"
        )
    if summary.fastest_run is not None:
        runs.append(next(r for r in pvc_runs(beats) if r[0].time == summary.fastest_run.start_time))
        labels.append(f"Fastest run: {_run_text(summary.fastest_run, start_time)}")
    return StripSection(heading=EXTREMES_TITLE, runs=runs, labels=labels)


def build_content(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> ReportContent:
    """Assemble the report text: summary, reference ranges, and the strip
    sections (heart-rate extremes, then flagged multi-beat runs, then
    isolated PVCs), the latter two capped at MAX_STRIPS_PER_SECTION with the
    cap stated in the heading. Event times are wall-clock labels when
    start_time is known."""
    # The last beat is the only end-of-recording marker available on every
    # path (no samples on the report-only path); it is within seconds of the
    # true end.
    duration_sec = beats[-1].time if beats else 0.0
    sections = [
        _extremes_section(beats, summary, start_time),
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
