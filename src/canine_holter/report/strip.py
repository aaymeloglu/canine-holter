"""Rhythm-strip drawing shared by the PNG and PDF reports."""
import numpy as np

STRIP_WINDOW_SEC = 6.0  # seconds of context shown around each flagged run


def draw_strip(ax, samples: np.ndarray, sample_rate: float, center_time: float) -> None:
    """Draw STRIP_WINDOW_SEC of waveform centred on center_time into ax."""
    half_window = STRIP_WINDOW_SEC / 2
    start_sample = max(0, int((center_time - half_window) * sample_rate))
    end_sample = min(len(samples), int((center_time + half_window) * sample_rate))
    segment = samples[start_sample:end_sample]
    t = np.arange(len(segment)) / sample_rate
    ax.plot(t, segment, linewidth=0.8)
    ax.set_xlabel("seconds")
    ax.set_ylabel("mV")
