"""Renders ReportContent as the PDF: the four summary panels on page 1,
the timeline with the hourly table on page 2 (the table continues on
pages of its own for recordings longer than a day), then a one-page primer
on reading strips, then two three-lead strips per page for each section
(or, with no waveform, one text page per section)."""
from __future__ import annotations
import textwrap
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
from canine_holter.report.common import (
    DISCLAIMER,
    HOW_TO_READ_STRIPS,
    HOW_TO_READ_TITLE,
    REPORT_TITLE,
    run_center_time,
)
from canine_holter.report.strip import CHANNEL_HEIGHT_MM, STRIP_WIDTH_MM, draw_strip, scale_label
from canine_holter.report.timeline import draw_timeline
from canine_holter.types import Beat

if TYPE_CHECKING:  # generate imports pdf; only the types are needed here
    from canine_holter.report.generate import ReportContent, StripCaption, StripSection, SummaryGroup

PAGE_SIZE_IN = (8.5, 11)  # US Letter, portrait
_PAGE_MM = (215.9, 279.4)
STRIPS_PER_PAGE = 2
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
_PANEL_TOP = (0.86, 0.68)  # top of each panel row

_STRIP_W = STRIP_WIDTH_MM / _PAGE_MM[0]  # 6 s at 25 mm/s, as a figure fraction
_CHANNEL_H = CHANNEL_HEIGHT_MM / _PAGE_MM[1]  # 3 mV at 10 mm/mV, per lead
_STRIP_SLOT = 0.45  # figure fraction per strip (caption + leads); two per page
_CAPTION_STEP = 0.017
_CAPTION_WRAP = 105


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
        colWidths=[0.14, 0.11] + [0.75 / (len(header) - 2)] * (len(header) - 2),
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


def _primer_page() -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, HOW_TO_READ_TITLE, va="top", fontsize=12, fontweight="bold")
    _text_block(fig, 0.91, HOW_TO_READ_STRIPS, fontsize=10)
    return fig


def _caption_lines(caption: StripCaption) -> list[str]:
    return textwrap.wrap(caption.what, _CAPTION_WRAP)


def _strip_page(
    heading: str,
    runs: list[list[Beat]],
    captions: list[StripCaption],
    pauses: list[tuple[float, float] | None],
    channels: np.ndarray,
    channel_names: tuple[str, ...],
    sample_rate: float,
    beats: list[Beat],
) -> Figure:
    """Up to STRIPS_PER_PAGE strips, each a caption block (title, what,
    significance) above its stacked leads drawn at true scale."""
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, heading, va="top", fontsize=12, fontweight="bold")
    for slot, (run, caption, pause) in enumerate(zip(runs, captions, pauses)):
        y = 0.90 - slot * _STRIP_SLOT
        fig.text(_LEFT, y, caption.title, va="top", fontsize=10, fontweight="bold")
        y -= 0.024
        what = _caption_lines(caption)
        for line in what:
            fig.text(_LEFT, y, line, va="top", fontsize=8.5)
            y -= _CAPTION_STEP
        fig.text(
            _LEFT, y, "● " + caption.significance, va="top", fontsize=8.5,
            color=STATUS_COLORS.get(caption.status or "", "black"),
            fontweight="bold" if caption.status else "normal",
        )
        y -= _CAPTION_STEP + 0.035  # room for the beat-label row above the top lead
        height = _CHANNEL_H * channels.shape[0]
        gs = GridSpec(1, 1, figure=fig, left=_LEFT, right=_LEFT + _STRIP_W, top=y, bottom=y - height)
        axes = draw_strip(
            fig, gs[0], channels, channel_names, sample_rate, run_center_time(run), beats,
            mark_times=[b.time for b in run], pause=pause,
        )
        window = axes[0].get_xlim()[1]
        range_mv = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
        fig.text(
            _LEFT + _STRIP_W, 0.90 - slot * _STRIP_SLOT, scale_label(window, range_mv),
            va="top", ha="right", fontsize=7.5, color=REFERENCE_COLOR,
        )
    return fig


def _text_page(heading: str, captions: list[StripCaption]) -> Figure:
    """The no-waveform fallback: every caption as text."""
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, heading, va="top", fontsize=12, fontweight="bold")
    y = 0.91
    for caption in captions:
        fig.text(_LEFT, y, caption.title, va="top", fontsize=10, fontweight="bold")
        y -= _LINE_STEP
        y = _text_block(fig, y, _caption_lines(caption) + [caption.significance], fontsize=9)
        y -= 0.01
    return fig


def _write_section(pdf, section: StripSection, channels, channel_names, sample_rate, beats) -> None:
    if channels is None or sample_rate is None:
        fig = _text_page(section.heading, section.captions)
        pdf.savefig(fig)
        plt.close(fig)
        return
    pauses = section.pauses or [None] * len(section.runs)
    for start in range(0, len(section.runs), STRIPS_PER_PAGE):
        stop = start + STRIPS_PER_PAGE
        fig = _strip_page(
            section.heading, section.runs[start:stop], section.captions[start:stop], pauses[start:stop],
            channels, channel_names, sample_rate, beats,
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
    channels: np.ndarray | None,
    channel_names: tuple[str, ...],
    sample_rate: float | None,
) -> None:
    """Write the multi-page PDF report. channels holds every lead to draw,
    (n_channels, n_samples); with none, each section's captions are listed
    as text on a page of their own and the primer page is skipped."""
    header, rows = content.hourly_header, content.hourly_rows
    first, rest = rows[:TABLE_ROWS_ON_TIMELINE_PAGE], rows[TABLE_ROWS_ON_TIMELINE_PAGE:]
    draw_strips = channels is not None and sample_rate is not None and bool(content.sections)
    with PdfPages(out_path) as pdf:
        pages = [
            _summary_page(content.summary_groups, content.footer_lines),
            _timeline_page(beats, summary, start_time, header, first),
        ]
        pages += [
            _table_page(header, rest[i : i + TABLE_ROWS_PER_PAGE])
            for i in range(0, len(rest), TABLE_ROWS_PER_PAGE)
        ]
        if draw_strips:
            pages.append(_primer_page())
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
        for section in content.sections:
            _write_section(pdf, section, channels, channel_names, sample_rate, beats)
