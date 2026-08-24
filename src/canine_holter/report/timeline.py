"""Whole-recording timeline: heart-rate trend plus lanes marking where PVCs,
pauses, and sustained brady/tachycardia occur, so clusters are visible at a
glance (e.g. a pile of "pauses" in the minutes after the vest slipped off)."""
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.types import Beat

HR_BIN_SEC = 60.0
MIN_SPAN_SEC = 5.0  # so a sub-second brady/tachy event is still visible at 2.5 h scale

# Categorical slots 1-4 of the dataviz reference palette (validated adjacent
# CVD-safe). Each lane is also named on the y-axis, so identity never relies
# on color alone.
LANES = [  # (label, color)
    ("PVC", "#2a78d6"),
    ("Pause", "#eb6834"),
    ("Brady", "#1baf7a"),
    ("Tachy", "#eda100"),
]
HR_COLOR = "#52514e"
GRID_COLOR = "#e5e4e0"
_TICK_TARGET = 6  # aim for roughly this many x-axis labels
_TICK_CHOICES_MIN = (1, 5, 10, 15, 30, 60, 120, 240)


def _tick_interval_minutes(span_sec: float) -> int:
    """Pick the coarsest of the standard minute intervals that still yields
    at least _TICK_TARGET labels, so a 25 s clip and a 24 h recording both
    get a handful of readable :00/:30-aligned ticks."""
    for minutes in reversed(_TICK_CHOICES_MIN):
        if span_sec / (minutes * 60) >= _TICK_TARGET:
            return minutes
    return _TICK_CHOICES_MIN[0]


def _recording_span_sec(beats: list[Beat], summary: ArrhythmiaSummary) -> float:
    """End of the recording as far as the timeline can know it: the last
    beat or event. Floored at one HR bin so an empty recording still has a
    non-degenerate axis."""
    ends = [b.time for b in beats] + list(summary.pauses)
    ends += [end for _, end in summary.bradycardia_events + summary.tachycardia_events]
    return max([HR_BIN_SEC] + ends)


def _heart_rate_trend(beats: list[Beat]) -> tuple[np.ndarray, np.ndarray]:
    """Median-RR heart rate per HR_BIN_SEC bin. Bins with fewer than two RR
    intervals are NaN so they draw as gaps rather than dropping to zero."""
    timed = [(b.time, b.rr_interval) for b in beats if b.rr_interval]
    if not timed:
        return np.array([]), np.array([])
    times = np.array([t for t, _ in timed])
    rr = np.array([r for _, r in timed])
    idx = (times // HR_BIN_SEC).astype(int)
    n_bins = int(idx.max()) + 1
    bpm = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = rr[idx == i]
        if len(sel) >= 2:
            bpm[i] = 60.0 / np.median(sel)
    centers = (np.arange(n_bins) + 0.5) * HR_BIN_SEC
    return centers, bpm


def plot_timeline(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
    out_path: str,
) -> None:
    """Write a two-panel PNG: heart rate over the recording on top, event
    lanes below. X-axis is time of day when start_time is known, otherwise
    minutes from the start of the recording."""
    if start_time is None:
        to_x = lambda sec: sec / 60.0  # noqa: E731
        to_width = lambda sec: sec / 60.0  # noqa: E731
    else:
        to_x = lambda sec: mdates.date2num(start_time + timedelta(seconds=float(sec)))  # noqa: E731
        to_width = lambda sec: sec / 86400.0  # noqa: E731  (matplotlib date units are days)

    fig, (ax_hr, ax_ev) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]}
    )

    centers, bpm = _heart_rate_trend(beats)
    # A line through a single point draws nothing; give short recordings a dot.
    marker = "o" if np.count_nonzero(~np.isnan(bpm)) == 1 else None
    ax_hr.plot([to_x(c) for c in centers], bpm, color=HR_COLOR, linewidth=1.5, marker=marker)
    ax_hr.set_ylabel("Heart rate (bpm)")
    ax_hr.set_title("Recording timeline")
    ax_hr.grid(axis="y", color=GRID_COLOR, linewidth=0.8)

    lane_items = [
        [b.time for b in beats if b.label == "V"],
        list(summary.pauses),
        list(summary.bradycardia_events),
        list(summary.tachycardia_events),
    ]
    for lane, ((_, color), items) in enumerate(zip(LANES, lane_items)):
        y = len(LANES) - 1 - lane  # first lane drawn at the top
        for item in items:
            if isinstance(item, tuple):
                start, end = item
                width = to_width(max(end - start, MIN_SPAN_SEC))
                ax_ev.broken_barh([(to_x(start), width)], (y - 0.35, 0.7), color=color)
            else:
                ax_ev.vlines(to_x(item), y - 0.35, y + 0.35, color=color, linewidth=1.2)
    ax_ev.set_yticks(range(len(LANES)))
    ax_ev.set_yticklabels([label for label, _ in reversed(LANES)])
    ax_ev.set_ylim(-0.6, len(LANES) - 0.4)
    ax_ev.grid(axis="x", color=GRID_COLOR, linewidth=0.8)

    for ax in (ax_hr, ax_ev):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Pin the axis to the recording. Left to autoscale, a near-degenerate
    # span in date units expands to +/-5% of the *date number* - years - and
    # the minute locator then tries to place tens of thousands of ticks.
    span = _recording_span_sec(beats, summary)
    pad = 0.01 * span
    ax_ev.set_xlim(to_x(-pad), to_x(span + pad))
    if start_time is None:
        ax_ev.set_xlabel("minutes from start")
    else:
        interval = _tick_interval_minutes(span)
        if interval >= 60:
            locator = mdates.HourLocator(interval=interval // 60)
        else:
            locator = mdates.MinuteLocator(byminute=range(0, 60, interval))
        ax_ev.xaxis.set_major_locator(locator)
        ax_ev.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_ev.set_xlabel("time of day")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
