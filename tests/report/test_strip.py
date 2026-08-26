import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from canine_holter.report.strip import draw_strip


def test_mark_times_draw_one_vertical_line_per_time_at_strip_offset():
    samples = np.zeros(2000)
    fig, ax = plt.subplots()
    draw_strip(ax, samples, 100.0, center_time=10.0, mark_times=[9.5, 10.0])
    xs = sorted(line.get_xdata()[0] for line in ax.lines[1:])
    assert xs == [2.5, 3.0]
    plt.close(fig)


def test_no_mark_times_draws_only_the_waveform():
    fig, ax = plt.subplots()
    draw_strip(ax, np.zeros(2000), 100.0, center_time=10.0)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_draw_strip_clamps_near_start_without_wraparound():
    """For an event within half a window of the start, (center_time -
    half_window) * sample_rate goes negative; unclamped, Python would slice
    from the end. Verify the strip starts at sample 0 and pulls in no tail."""
    samples = np.zeros(1000)
    samples[:50] = 1.0
    samples[-50:] = -1.0
    fig, ax = plt.subplots()
    draw_strip(ax, samples, 100.0, center_time=0.5)
    ydata = ax.lines[0].get_ydata()
    assert len(ydata) > 0
    assert 1.0 in ydata
    assert -1.0 not in ydata
    plt.close(fig)


def test_marks_wider_than_the_window_widen_the_strip_to_cover_them():
    """An 18 s pause is longer than the 6 s window; the strip grows so both
    edges of the gap are on the waveform, with a margin either side."""
    samples = np.zeros(10000)
    fig, ax = plt.subplots()
    draw_strip(ax, samples, 100.0, center_time=50.0, mark_times=[41.0, 59.0])
    t = ax.lines[0].get_xdata()
    assert abs(t[-1] - t[0] - 20.0) < 0.05  # 18 s span + 1 s margin each side
    xs = sorted(line.get_xdata()[0] for line in ax.lines[1:])
    assert xs == [1.0, 19.0]
    plt.close(fig)
