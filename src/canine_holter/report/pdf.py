"""Renders ReportContent as the PDF: summary, reference ranges, and
timeline on page 1, then rhythm strips per section on the pages after (or,
with no waveform, one text page per section)."""
from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.common import DISCLAIMER, REPORT_TITLE, run_center_time
from canine_holter.report.strip import draw_strip
from canine_holter.report.timeline import draw_timeline
from canine_holter.types import Beat

if TYPE_CHECKING:  # generate imports pdf; only the type is needed here
    from canine_holter.report.generate import ReportContent, StripSection

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


def _write_section(pdf, section: StripSection, samples, sample_rate) -> None:
    if samples is None or sample_rate is None:
        fig = _text_page(section.heading, section.labels)
        pdf.savefig(fig)
        plt.close(fig)
        return
    for start in range(0, len(section.runs), STRIPS_PER_PAGE):
        stop = start + STRIPS_PER_PAGE
        fig = _strip_page(
            section.heading, section.runs[start:stop], section.labels[start:stop], samples, sample_rate
        )
        pdf.savefig(fig)
        plt.close(fig)


def write_pdf(
    out_path: str,
    *,
    content: ReportContent,
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
    samples: np.ndarray | None,
    sample_rate: float | None,
) -> None:
    """Write the multi-page PDF report. With no waveform samples each
    section's events are listed as text on a page of their own."""
    with PdfPages(out_path) as pdf:
        fig = _summary_page(content.summary_lines, content.reference_lines, beats, summary, start_time)
        pdf.savefig(fig)
        plt.close(fig)
        for section in content.sections:
            _write_section(pdf, section, samples, sample_rate)
