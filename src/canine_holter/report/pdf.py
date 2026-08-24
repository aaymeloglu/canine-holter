"""Single-file PDF report: summary text and timeline on page 1, rhythm
strips for flagged events on the pages after."""
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.common import DISCLAIMER, REPORT_TITLE, event_line, flagged_runs, run_center_time
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
    event_lines: list[str],
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
    if event_lines:
        y -= 0.01
        fig.text(_LEFT, y, "Flagged events (couplets, triplets, VT runs)", va="top", fontsize=12, fontweight="bold")
        y -= 0.03
        y = _text_block(fig, y, event_lines, fontsize=10)
    timeline_top = min(0.50, y - 0.03)
    gs = GridSpec(1, 1, figure=fig, top=timeline_top, bottom=0.06, left=_LEFT, right=0.97)
    draw_timeline(fig, gs[0], beats, summary, start_time)
    return fig


def _strip_page(
    runs: list[list[Beat]],
    labels: list[str],
    samples: np.ndarray,
    sample_rate: float,
) -> Figure:
    fig = plt.figure(figsize=PAGE_SIZE_IN)
    gs = GridSpec(STRIPS_PER_PAGE, 1, figure=fig, top=0.94, bottom=0.06, left=_LEFT, right=0.97, hspace=0.7)
    for row, (run, label) in enumerate(zip(runs, labels)):
        ax = fig.add_subplot(gs[row])
        draw_strip(ax, samples, sample_rate, run_center_time(run))
        ax.set_title(label, loc="left", fontsize=10)
    return fig


def write_pdf(
    out_path: str,
    *,
    summary_lines: list[str],
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
    samples: np.ndarray | None,
    sample_rate: float | None,
) -> None:
    """Write the multi-page PDF report. With no waveform samples the flagged
    events are listed as text on page 1 instead of plotted."""
    runs = flagged_runs(beats)
    labels = [event_line(i, run, start_time) for i, run in enumerate(runs)]
    have_waveform = samples is not None and sample_rate is not None
    with PdfPages(out_path) as pdf:
        fig = _summary_page(summary_lines, [] if have_waveform else labels, beats, summary, start_time)
        pdf.savefig(fig)
        plt.close(fig)
        if have_waveform:
            for start in range(0, len(runs), STRIPS_PER_PAGE):
                stop = start + STRIPS_PER_PAGE
                fig = _strip_page(runs[start:stop], labels[start:stop], samples, sample_rate)
                pdf.savefig(fig)
                plt.close(fig)
