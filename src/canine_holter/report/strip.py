"""Rhythm-strip drawing: every recorded lead stacked on a standard ECG grid
at clinical scale (25 mm/s, 10 mm/mV), with a label over each beat, RR
intervals around the flagged beats, and the flagged beats shaded across all
leads. The scale actually used is always printed (see scale_label), so a
strip widened for a long pause or squeezed for a tall signal never passes
for a standard one."""
import math
from collections.abc import Sequence
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import blended_transform_factory
from canine_holter.types import Beat

MM_PER_SEC = 25.0  # standard ECG paper speed
MM_PER_MV = 10.0  # standard ECG gain
STRIP_WINDOW_SEC = 6.0  # seconds of context shown around each flagged run
MARK_MARGIN_SEC = 1.0  # kept either side of the marks when they span more than the window
CHANNEL_RANGE_MV = 3.0  # each lead panel shows this much, centred on its window; grows for taller signals
STRIP_WIDTH_MM = STRIP_WINDOW_SEC * MM_PER_SEC  # 150 mm: fixed on the page
CHANNEL_HEIGHT_MM = CHANNEL_RANGE_MV * MM_PER_MV  # 30 mm per lead: fixed on the page

MARK_COLOR = "#c62828"  # flagged beats: matches the summary's alert colour
LABEL_COLOR = "#6f6e6b"
TRACE_COLOR = "#1a1a1a"
GRID_MINOR_COLOR = "#f6d5d5"  # 1 mm squares
GRID_MAJOR_COLOR = "#e8a3a3"  # 5 mm squares
_BAND_BEFORE_SEC, _BAND_AFTER_SEC = 0.08, 0.12  # shading around a flagged R-peak: covers the QRS
_LABEL_Y, _RR_Y = 1.10, 1.02  # axes-fraction rows above the top panel


def strip_window(center_time: float, mark_times: Sequence[float], duration_sec: float) -> tuple[float, float]:
    """(start, end) seconds of the strip: STRIP_WINDOW_SEC centred on
    center_time, widened to keep every mark on the waveform with a margin,
    clamped to the recording."""
    span = (max(mark_times) - min(mark_times)) if mark_times else 0.0
    half = max(STRIP_WINDOW_SEC, span + 2 * MARK_MARGIN_SEC) / 2
    start = max(0.0, center_time - half)
    return start, min(duration_sec, start + 2 * half)


def scale_label(window_sec: float, range_mv: float) -> str:
    """The paper speed and gain the strip is actually drawn at."""
    return f"{STRIP_WIDTH_MM / window_sec:.0f} mm/s · {CHANNEL_HEIGHT_MM / range_mv:.0f} mm/mV"


def channel_range_mv(segments: Sequence[np.ndarray]) -> float:
    """CHANNEL_RANGE_MV, or the next whole millivolt above the tallest
    lead's swing so no lead is clipped."""
    ptp = max((float(s.max() - s.min()) for s in segments if len(s)), default=0.0)
    return max(CHANNEL_RANGE_MV, float(math.ceil(ptp + 0.2)))


def _beat_label(beat: Beat) -> str:
    return beat.label if beat.label in ("N", "V") else "?"


def _grid(ax: Axes) -> None:
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.4)
    ax.grid(True, which="major", color=GRID_MAJOR_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID_MAJOR_COLOR)


def draw_strip(
    fig: Figure,
    subplot_spec: SubplotSpec,
    channels: np.ndarray,
    channel_names: Sequence[str],
    sample_rate: float,
    center_time: float,
    beats: Sequence[Beat],
    mark_times: Sequence[float] = (),
    pause: tuple[float, float] | None = None,
    analysis_channel: int = 0,
) -> list[Axes]:
    """Draw one strip (all leads) into a region of fig and return its axes,
    top lead first. beats are the recording's beats; those inside the window
    get labels. mark_times are the flagged beats; pause is the (start, end)
    of a gap to bracket."""
    n_channels = channels.shape[0]
    duration = channels.shape[1] / sample_rate
    start, end = strip_window(center_time, mark_times, duration)
    window = end - start
    s0, s1 = int(start * sample_rate), int(end * sample_rate)
    segments = [channels[i, s0:s1] - np.median(channels[i, s0:s1]) for i in range(n_channels)]
    range_mv = channel_range_mv(segments)
    t = np.arange(s1 - s0) / sample_rate

    inner = subplot_spec.subgridspec(n_channels, 1, hspace=0)
    axes: list[Axes] = []
    for i, segment in enumerate(segments):
        ax = fig.add_subplot(inner[i], sharex=axes[0] if axes else None)
        ax.plot(t, segment, color=TRACE_COLOR, linewidth=0.8)
        mid = (float(segment.max()) + float(segment.min())) / 2 if len(segment) else 0.0
        ax.set_xlim(0, window)
        ax.set_ylim(mid - range_mv / 2, mid + range_mv / 2)
        _grid(ax)
        for mark in mark_times:
            ax.axvspan(mark - start - _BAND_BEFORE_SEC, mark - start + _BAND_AFTER_SEC, color=MARK_COLOR, alpha=0.15, linewidth=0)
        name = channel_names[i] + (" (analysis)" if i == analysis_channel else "")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8, labelpad=6)
        last = i == n_channels - 1
        ax.tick_params(labelleft=False, left=False, labelbottom=last, bottom=last, which="both", labelsize=7)
        axes.append(ax)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} s" if abs(v - round(v)) < 1e-6 else ""))

    top = axes[0]
    trans = blended_transform_factory(top.transData, top.transAxes)
    in_window = [b for b in beats if start <= b.time <= end]
    for beat in in_window:
        label = _beat_label(beat)
        top.text(
            beat.time - start, _LABEL_Y, label, transform=trans, ha="center", va="bottom", fontsize=8,
            fontweight="bold" if label == "V" else "normal", color=MARK_COLOR if label == "V" else LABEL_COLOR,
        )
    marked = {i for i, b in enumerate(in_window) for m in mark_times if abs(b.time - m) < 1e-6}
    for j in sorted({j for i in marked for j in (i, i + 1) if 0 < j < len(in_window)}):
        rr = in_window[j].rr_interval
        if rr:
            x = (in_window[j - 1].time + in_window[j].time) / 2 - start
            top.text(x, _RR_Y, f"{rr:.2f} s", transform=trans, ha="center", va="bottom", fontsize=6.5, color=LABEL_COLOR)
    if pause is not None:
        gap_start, gap_end = pause
        y = sum(top.get_ylim()) / 2
        top.annotate(
            "", xy=(gap_end - start, y), xytext=(gap_start - start, y),
            arrowprops=dict(arrowstyle="<->", color=MARK_COLOR, linewidth=1.0),
        )
        top.text(
            (gap_start + gap_end) / 2 - start, y + range_mv * 0.12, f"{gap_end - gap_start:.2f} s gap",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=MARK_COLOR,
        )
    return axes
