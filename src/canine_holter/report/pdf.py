"""Single-file PDF report: summary, reference ranges, and timeline on page
1, then rhythm strips for flagged events and isolated PVCs on the pages
after (or, with no waveform, one text page per section)."""
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.common import (
    DISCLAIMER,
    EVENTS_TITLE,
    ISOLATED_TITLE,
    MAX_STRIPS_PER_SECTION,
    REPORT_TITLE,
    event_line,
    flagged_runs,
    isolated_pvcs,
    pvc_line,
    run_center_time,
    section_heading,
    select_evenly,
)
from canine_holter.report.strip import draw_strip
from canine_holter.report.timeline import draw_timeline
from canine_holter.types import Beat

PAGE_SIZE_IN = (8.5, 11)  # US Letter, portrait
STRIPS_PER_PAGE = 3
_LEFT = 0.09
_LINE_STEP = 0.021  # fraction of page height per text line at 10 pt


def _text_block(fig: Figure, y: float, lines: list[str], **kw) -> float:
    """Write lines top-down from y (figure fraction); return the next y."""
    for line in lines:
        fig.text(_LEFT, y, line, va="top", **kw)
        y -= _LINE_STEP
    return y


def _summary_page(
    summary_lines: list[str],
    reference_lines: list[str],
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    y = 0.95
    fig.text(_LEFT, y, REPORT_TITLE, va="top", fontsize=16, fontweight="bold")
    y -= 0.035
    fig.text(_LEFT, y, DISCLAIMER, va="top", fontsize=10, fontstyle="italic")
    y -= 0.04
    fig.text(_LEFT, y, "Summary", va="top", fontsize=12, fontweight="bold")
    y -= 0.03
    y = _text_block(fig, y, [line.lstrip("- ") for line in summary_lines], fontsize=10)
    y -= 0.01
    fig.text(_LEFT, y, "Reference ranges", va="top", fontsize=12, fontweight="bold")
    y -= 0.03
    y = _text_block(fig, y, [line.lstrip("- ") for line in reference_lines], fontsize=9)
    gs = GridSpec(1, 1, figure=fig, top=y - 0.04, bottom=0.06, left=_LEFT, right=0.97)
    draw_timeline(fig, gs[0], beats, summary, start_time)
    return fig


def _strip_page(
    heading: str,
    runs: list[list[Beat]],
    labels: list[str],
    samples: np.ndarray,
    sample_rate: float,
) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, heading, va="top", fontsize=12, fontweight="bold")
    gs = GridSpec(STRIPS_PER_PAGE, 1, figure=fig, top=0.90, bottom=0.06, left=_LEFT, right=0.97, hspace=0.7)
    for row, (run, label) in enumerate(zip(runs, labels)):
        ax = fig.add_subplot(gs[row])
        draw_strip(ax, samples, sample_rate, run_center_time(run), mark_times=[b.time for b in run])
        ax.set_title(label, loc="left", fontsize=10)
    return fig


def _text_page(heading: str, lines: list[str]) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, heading, va="top", fontsize=12, fontweight="bold")
    _text_block(fig, 0.91, lines, fontsize=10)
    return fig


def _write_section(pdf, title, runs, labeler, start_time, samples, sample_rate) -> None:
    if not runs:
        return
    shown = select_evenly(runs, MAX_STRIPS_PER_SECTION)
    heading = section_heading(title, len(shown), len(runs))
    labels = [labeler(i, run, start_time) for i, run in enumerate(shown)]
    if samples is None or sample_rate is None:
        fig = _text_page(heading, labels)
        pdf.savefig(fig)
        plt.close(fig)
        return
    for start in range(0, len(shown), STRIPS_PER_PAGE):
        stop = start + STRIPS_PER_PAGE
        fig = _strip_page(heading, shown[start:stop], labels[start:stop], samples, sample_rate)
        pdf.savefig(fig)
        plt.close(fig)


def write_pdf(
    out_path: str,
    *,
    summary_lines: list[str],
    reference_lines: list[str],
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
    samples: np.ndarray | None,
    sample_rate: float | None,
) -> None:
    """Write the multi-page PDF report. With no waveform samples each
    section's events are listed as text on a page of their own."""
    with PdfPages(out_path) as pdf:
        fig = _summary_page(summary_lines, reference_lines, beats, summary, start_time)
        pdf.savefig(fig)
        plt.close(fig)
        _write_section(pdf, EVENTS_TITLE, flagged_runs(beats), event_line, start_time, samples, sample_rate)
        _write_section(pdf, ISOLATED_TITLE, isolated_pvcs(beats), pvc_line, start_time, samples, sample_rate)
