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
    REPORT_TITLE,
    event_line,
    flagged_runs,
    format_time,
    run_center_time,
)
from canine_holter.report.pdf import write_pdf
from canine_holter.report.strip import draw_strip
from canine_holter.report.timeline import plot_timeline


def _plot_strip(
    samples: np.ndarray, sample_rate: float, center_time: float, out_path: str, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    draw_strip(ax, samples, sample_rate, center_time)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _summary_lines(summary: ArrhythmiaSummary, start_time: datetime | None, duration_sec: float) -> list[str]:
    """The Summary bullet lines, shared by the markdown and the PDF."""
    hours, rem = divmod(int(duration_sec), 3600)
    start_text = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "unknown"
    return [
        f"- Recording start: {start_text}",
        f"- Duration: {hours}h {rem // 60}m",
        f"- Total beats: {summary.total_beats}",
        f"- PVC count: {summary.pvc_count}",
        f"- PVC burden: {summary.pvc_burden_pct:.2f}%",
        f"- Couplets: {summary.couplets}",
        f"- Triplets: {summary.triplets}",
        f"- VT runs (4+ consecutive PVCs): {summary.vtach_runs}",
        f"- Pauses (>= threshold): {len(summary.pauses)}",
        f"- Sustained bradycardia events: {len(summary.bradycardia_events)}",
        f"- Sustained tachycardia events: {len(summary.tachycardia_events)}",
    ]


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
    start_time: datetime | None = None,
) -> str:
    """Write the report: report.pdf (the primary artifact - summary text,
    timeline, and rhythm strips in one file), plus report.md, timeline.png,
    and (if waveform data is provided) a strip PNG per flagged multi-beat
    PVC run. Event times are wall-clock labels when start_time is known.
    Returns the path to the PDF."""
    os.makedirs(out_dir, exist_ok=True)

    # The last beat is the only end-of-recording marker available on every
    # path (no samples on the report-only path); it is within seconds of the
    # true end.
    duration_sec = beats[-1].time if beats else 0.0
    summary_lines = _summary_lines(summary, start_time, duration_sec)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"**{DISCLAIMER}**",
        "",
        "## Summary",
        *summary_lines,
        "",
    ]

    plot_timeline(beats, summary, start_time, os.path.join(out_dir, "timeline.png"))
    lines += ["## Timeline", "![timeline](timeline.png)", ""]

    flagged = flagged_runs(beats)
    if flagged:
        lines.append("## Flagged events (couplets, triplets, VT runs)")
        for i, run in enumerate(flagged):
            lines.append(f"- {event_line(i, run, start_time)}")
            if samples is not None and sample_rate is not None:
                plot_path = os.path.join(out_dir, f"event_{i + 1}_strip.png")
                title = f"Rhythm strip around {format_time(run_center_time(run), start_time)}"
                _plot_strip(samples, sample_rate, run_center_time(run), plot_path, title=title)
                lines.append(f"  ![event {i + 1}]({os.path.basename(plot_path)})")
        lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))

    pdf_path = os.path.join(out_dir, "report.pdf")
    write_pdf(
        pdf_path,
        summary_lines=summary_lines,
        beats=beats,
        summary=summary,
        start_time=start_time,
        samples=samples,
        sample_rate=sample_rate,
    )
    return pdf_path
