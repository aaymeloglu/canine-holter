"""Rhythm-strip drawing shared by the PNG and PDF reports."""
from collections.abc import Sequence
import numpy as np

STRIP_WINDOW_SEC = 6.0  # seconds of context shown around each flagged run
MARK_MARGIN_SEC = 1.0  # kept either side of the marks when they span more than the window


MARK_COLOR = "#2a78d6"  # matches the timeline's PVC lane


def draw_strip(
    ax,
    samples: np.ndarray,
    sample_rate: float,
    center_time: float,
    mark_times: Sequence[float] = (),
) -> None:
    """Draw STRIP_WINDOW_SEC of waveform centred on center_time into ax,
    with a faint vertical line at each of mark_times (recording seconds) so
    the flagged beats can be found without knowing their morphology. Marks
    spanning more than the window (a long pause) widen it to keep them all
    on the waveform."""
    span = (max(mark_times) - min(mark_times)) if mark_times else 0.0
    half_window = max(STRIP_WINDOW_SEC, span + 2 * MARK_MARGIN_SEC) / 2
    start_sample = max(0, int((center_time - half_window) * sample_rate))
    end_sample = min(len(samples), int((center_time + half_window) * sample_rate))
    segment = samples[start_sample:end_sample]
    t = np.arange(len(segment)) / sample_rate
    ax.plot(t, segment, linewidth=0.8)
    for mark in mark_times:
        ax.axvline(mark - start_sample / sample_rate, color=MARK_COLOR, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("seconds")
    ax.set_ylabel("mV")
