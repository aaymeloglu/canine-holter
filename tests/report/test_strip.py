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
