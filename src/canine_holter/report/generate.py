import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.common import (
    DISCLAIMER,
    EVENTS_TITLE,
    ISOLATED_TITLE,
    MAX_STRIPS_PER_SECTION,
    REPORT_TITLE,
    event_line,
    flagged_runs,
    format_time,
    isolated_pvcs,
    pvc_line,
    run_center_time,
    section_heading,
    select_evenly,
)
from canine_holter.report.pdf import write_pdf
from canine_holter.report.reference import format_duration, pvc_per_24h_line, reference_lines
from canine_holter.report.strip import draw_strip
from canine_holter.report.timeline import plot_timeline


def _plot_strip(
    samples: np.ndarray,
    sample_rate: float,
    center_time: float,
    out_path: str,
    title: str,
    mark_times: list[float] = (),
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    draw_strip(ax, samples, sample_rate, center_time, mark_times=mark_times)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _summary_lines(summary: ArrhythmiaSummary, start_time: datetime | None, duration_sec: float) -> list[str]:
    """The Summary bullet lines, shared by the markdown and the PDF."""
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


def _strip_section(
    lines: list[str],
    title: str,
    runs: list[list[Beat]],
    labeler,
    file_stem: str,
    md_alt: str,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
    start_time: datetime | None,
) -> None:
    """Append one markdown section listing (up to the cap) the given PVC
    runs, writing a strip PNG per run when waveform data is available."""
    if not runs:
        return
    shown = select_evenly(runs, MAX_STRIPS_PER_SECTION)
    lines.append(f"## {section_heading(title, len(shown), len(runs))}")
    for i, run in enumerate(shown):
        label = labeler(i, run, start_time)
        lines.append(f"- {label}")
        if samples is not None and sample_rate is not None:
            plot_path = os.path.join(out_dir, f"{file_stem}_{i + 1}_strip.png")
            title_text = f"Rhythm strip around {format_time(run_center_time(run), start_time)}"
            _plot_strip(
                samples, sample_rate, run_center_time(run), plot_path,
                title=title_text, mark_times=[b.time for b in run],
            )
            lines.append(f"  ![{md_alt} {i + 1}]({os.path.basename(plot_path)})")
    lines.append("")


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
    start_time: datetime | None = None,
) -> str:
    """Write the report: report.pdf (the primary artifact - summary text,
    reference ranges, timeline, and rhythm strips in one file), plus
    report.md, timeline.png, and (if waveform data is provided) a strip PNG
    per flagged multi-beat PVC run and per isolated PVC, each section capped
    at MAX_STRIPS_PER_SECTION. Event times are wall-clock labels when
    start_time is known. Returns the path to the PDF."""
    os.makedirs(out_dir, exist_ok=True)

    # The last beat is the only end-of-recording marker available on every
    # path (no samples on the report-only path); it is within seconds of the
    # true end.
    duration_sec = beats[-1].time if beats else 0.0
    summary_lines = _summary_lines(summary, start_time, duration_sec)
    ref_lines = reference_lines(duration_sec)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"**{DISCLAIMER}**",
        "",
        "## Summary",
        *summary_lines,
        "",
        "## Reference ranges",
        *ref_lines,
        "",
    ]

    plot_timeline(beats, summary, start_time, os.path.join(out_dir, "timeline.png"))
    lines += ["## Timeline", "![timeline](timeline.png)", ""]

    _strip_section(
        lines, EVENTS_TITLE, flagged_runs(beats), event_line, "event", "event",
        out_dir, samples, sample_rate, start_time,
    )
    _strip_section(
        lines, ISOLATED_TITLE, isolated_pvcs(beats), pvc_line, "pvc", "pvc",
        out_dir, samples, sample_rate, start_time,
    )

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))

    pdf_path = os.path.join(out_dir, "report.pdf")
    write_pdf(
        pdf_path,
        summary_lines=summary_lines,
        reference_lines=ref_lines,
        beats=beats,
        summary=summary,
        start_time=start_time,
        samples=samples,
        sample_rate=sample_rate,
    )
    return pdf_path
