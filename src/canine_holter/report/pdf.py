"""Renders ReportContent as the PDF: the six summary panels on page 1,
the timeline with the hourly table on page 2 (the table continues on
pages of its own for recordings longer than a day), then a one-page primer
on reading strips, then two three-lead strips per page for each section
(or, with no waveform, one text page per section)."""
import os
import textwrap
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.generate import (
    ReportContent,
    StripCaption,
    StripItem,
    StripSection,
    SummaryGroup,
    build_content,
)
from canine_holter.report.strip import CHANNEL_HEIGHT_MM, STRIP_WIDTH_MM, draw_strip, scale_label
from canine_holter.report.timeline import draw_timeline
from canine_holter.types import Beat

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
_PANEL_TOP = (0.86, 0.68, 0.50)  # top of each panel row; the tallest panel (seven rows) fits the 0.18 pitch

_STRIP_LEFT = 0.17  # leaves room for the lead names left of the strip
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


def _summary_page(content: ReportContent) -> Figure:
    """Title, disclaimer, the six panels in a 3x2 grid, then the legend and
    source lines."""
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    y = 0.95
    fig.text(_LEFT, y, content.title, va="top", fontsize=16, fontweight="bold")
    y -= 0.035
    fig.text(_LEFT, y, content.disclaimer, va="top", fontsize=10, fontstyle="italic")
    bottom = 1.0
    for i, group in enumerate(content.summary_groups):
        bottom = min(bottom, _draw_group(fig, _PANEL_X[i % 2], _PANEL_TOP[i // 2], group))
    _text_block(fig, bottom - 0.02, content.footer_lines, fontsize=8, color=REFERENCE_COLOR)
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


def _primer_page(content: ReportContent) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, content.primer_title, va="top", fontsize=12, fontweight="bold")
    _text_block(fig, 0.91, content.primer_lines, fontsize=10)
    return fig


def _caption_lines(caption: StripCaption) -> list[str]:
    return textwrap.wrap(caption.what, _CAPTION_WRAP)


def _significance_lines(caption: StripCaption) -> list[str]:
    if not caption.significance:
        return []
    return textwrap.wrap("\u25cf " + caption.significance, _CAPTION_WRAP)


def _strip_page(
    heading: str,
    items: list[StripItem],
    channels: np.ndarray,
    channel_names: tuple[str, ...],
    sample_rate: float,
    beats: list[Beat],
) -> Figure:
    """Up to STRIPS_PER_PAGE strips, each a caption block (title, what,
    significance) above its stacked leads drawn at true scale."""
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, heading, va="top", fontsize=12, fontweight="bold")
    for slot, item in enumerate(items):
        run, caption, pause = item.run, item.caption, item.pause
        y = 0.90 - slot * _STRIP_SLOT
        fig.text(_LEFT, y, caption.title, va="top", fontsize=10, fontweight="bold")
        y -= 0.024
        what = _caption_lines(caption)
        for line in what:
            fig.text(_LEFT, y, line, va="top", fontsize=8.5)
            y -= _CAPTION_STEP
        for line in _significance_lines(caption):
            fig.text(
                _LEFT, y, line, va="top", fontsize=8.5,
                color=STATUS_COLORS.get(caption.status or "", "black"),
                fontweight="bold" if caption.status else "normal",
            )
            y -= _CAPTION_STEP
        y -= 0.04  # room for the beat-label and RR rows above the top lead
        height = _CHANNEL_H * channels.shape[0]
        gs = GridSpec(1, 1, figure=fig, left=_STRIP_LEFT, right=_STRIP_LEFT + _STRIP_W, top=y, bottom=y - height)
        axes = draw_strip(
            fig, gs[0], channels, channel_names, sample_rate, item.center_time, beats,
            mark_times=[b.time for b in run], pause=pause,
        )
        window = axes[0].get_xlim()[1]
        range_mv = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
        fig.text(
            _STRIP_LEFT + _STRIP_W, 0.90 - slot * _STRIP_SLOT, scale_label(window, range_mv),
            va="top", ha="right", fontsize=7.5, color=REFERENCE_COLOR,
        )
    return fig


def _text_page(heading: str, items: list[StripItem]) -> Figure:
    """The no-waveform fallback: every caption as text."""
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    fig.text(_LEFT, 0.95, heading, va="top", fontsize=12, fontweight="bold")
    y = 0.91
    for item in items:
        caption = item.caption
        fig.text(_LEFT, y, caption.title, va="top", fontsize=10, fontweight="bold")
        y -= _LINE_STEP
        y = _text_block(fig, y, _caption_lines(caption) + _significance_lines(caption), fontsize=9)
        y -= 0.01
    return fig


def _write_section(pdf, section: StripSection, channels, channel_names, sample_rate, beats) -> None:
    if channels is None or sample_rate is None:
        fig = _text_page(section.heading, section.items)
        pdf.savefig(fig)
        plt.close(fig)
        return
    for start in range(0, len(section.items), STRIPS_PER_PAGE):
        stop = start + STRIPS_PER_PAGE
        fig = _strip_page(
            section.heading,
            section.items[start:stop],
            channels,
            channel_names,
            sample_rate,
            beats,
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
            _summary_page(content),
            _timeline_page(beats, summary, start_time, header, first),
        ]
        pages += [
            _table_page(header, rest[i : i + TABLE_ROWS_PER_PAGE])
            for i in range(0, len(rest), TABLE_ROWS_PER_PAGE)
        ]
        if draw_strips:
            pages.append(_primer_page(content))
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)
        for section in content.sections:
            _write_section(pdf, section, channels, channel_names, sample_rate, beats)


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
    start_time: datetime | None = None,
    channels: np.ndarray | None = None,
    channel_names: tuple[str, ...] = (),
) -> str:
    """Write report.pdf into out_dir and return its path. Nothing else is
    written. Strips show every lead in channels, or the one lead in samples
    when that is all there is; without any waveform they are listed as
    text."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")
    if channels is None and samples is not None:
        channels, channel_names = samples[None, :], ("ECG",)
    write_pdf(
        pdf_path,
        content=build_content(beats, summary, start_time),
        beats=beats,
        summary=summary,
        start_time=start_time,
        channels=channels,
        channel_names=channel_names,
        sample_rate=sample_rate,
    )
    return pdf_path
