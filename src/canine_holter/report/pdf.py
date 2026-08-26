"""Renders ReportContent as the PDF: the four summary panels on page 1,
the timeline with the hourly table on page 2 (the table continues on
pages of its own for recordings longer than a day), then rhythm strips per
section on the pages after (or, with no waveform, one text page per
section)."""
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
    from canine_holter.report.generate import ReportContent, StripSection, SummaryGroup

PAGE_SIZE_IN = (8.5, 11)  # US Letter, portrait
STRIPS_PER_PAGE = 3
TABLE_ROWS_ON_TIMELINE_PAGE = 26  # a full day plus its partial hour fits under the timeline
TABLE_ROWS_PER_PAGE = 40
_TABLE_HEADING = "Hourly summary"
_LEFT = 0.09
_LINE_STEP = 0.021  # fraction of page height per text line at 10 pt

STATUS_COLORS = {"ok": "#2e7d32", "caution": "#b26a00", "alert": "#c62828"}
LABEL_COLOR = "#52514e"
REFERENCE_COLOR = "#6f6e6b"
_PANEL_X = (_LEFT, 0.53)  # left edge of each panel column
_PANEL_W = 0.42
_VALUE_DX = 0.12  # value column offset inside a panel
_PANEL_TOP = (0.80, 0.60)  # top of each panel row


def _text_block(fig: Figure, y: float, lines: list[str], **kw) -> float:
    """Write lines top-down from y (figure fraction); return the next y."""
    for line in lines:
        fig.text(_LEFT, y, line, va="top", **kw)
        y -= _LINE_STEP
    return y


def _draw_group(fig: Figure, x: float, y: float, group: SummaryGroup) -> float:
    """One panel: title, then label / value / reference per row. Returns the
    y below the last row."""
    fig.text(x, y, group.title.upper(), va="top", fontsize=9, fontweight="bold")
    y -= 0.025
    for row in group.rows:
        fig.text(x, y, row.label, va="top", fontsize=9, color=LABEL_COLOR)
        fig.text(
            x + _VALUE_DX, y, row.value, va="top", fontsize=9,
            color=STATUS_COLORS.get(row.status or "", "black"),
            fontweight="bold" if row.status else "normal",
        )
        if row.reference:
            fig.text(x + _PANEL_W, y, row.reference, va="top", ha="right", fontsize=7.5, color=REFERENCE_COLOR)
        y -= _LINE_STEP
    return y


def _summary_page(groups: list[SummaryGroup], footer_lines: list[str]) -> Figure:
    """Title, disclaimer, the four panels in a 2x2 grid, then the legend and
    source lines."""
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    y = 0.95
    fig.text(_LEFT, y, REPORT_TITLE, va="top", fontsize=16, fontweight="bold")
    y -= 0.035
    fig.text(_LEFT, y, DISCLAIMER, va="top", fontsize=10, fontstyle="italic")
    bottom = 1.0
    for i, group in enumerate(groups):
        bottom = min(bottom, _draw_group(fig, _PANEL_X[i % 2], _PANEL_TOP[i // 2], group))
    _text_block(fig, bottom - 0.02, footer_lines, fontsize=8, color=REFERENCE_COLOR)
    return fig


def _draw_table(fig: Figure, header: list[str], rows: list[list[str]], top: float) -> None:
    """The hourly table, headed, growing downward from top (figure fraction)."""
    fig.text(_LEFT, top, _TABLE_HEADING, va="top", fontsize=12, fontweight="bold")
    ax = fig.add_axes((_LEFT, 0.04, 0.97 - _LEFT, top - 0.07))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=header,
        colWidths=[0.16] + [0.105] * (len(header) - 1),
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)


def _timeline_page(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
    header: list[str],
    rows: list[list[str]],
) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, "Heart-rate timeline and events", va="top", fontsize=12, fontweight="bold")
    gs = GridSpec(1, 1, figure=fig, top=0.90, bottom=0.66, left=_LEFT, right=0.97)
    draw_timeline(fig, gs[0], beats, summary, start_time)
    if summary.excluded:
        fig.text(
            _LEFT, 0.62, "Hatched grey bands: excluded from analysis (artifact / off-body).",
            va="top", fontsize=8, color=REFERENCE_COLOR,
        )
    if rows:
        _draw_table(fig, header, rows, top=0.58)
    return fig


def _table_page(header: list[str], rows: list[list[str]]) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    _draw_table(fig, header, rows, top=0.95)
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
    header, rows = content.hourly_header, content.hourly_rows
    first, rest = rows[:TABLE_ROWS_ON_TIMELINE_PAGE], rows[TABLE_ROWS_ON_TIMELINE_PAGE:]
    with PdfPages(out_path) as pdf:
        pages = [
            _summary_page(content.summary_groups, content.footer_lines),
            _timeline_page(beats, summary, start_time, header, first),
        ]
        pages += [
            _table_page(header, rest[i : i + TABLE_ROWS_PER_PAGE])
            for i in range(0, len(rest), TABLE_ROWS_PER_PAGE)
        ]
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
        for section in content.sections:
            _write_section(pdf, section, samples, sample_rate)
