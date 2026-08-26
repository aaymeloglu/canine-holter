import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from canine_holter.report.strip import channel_range_mv, draw_strip, scale_label, strip_window
from canine_holter.types import Beat

FS = 100.0


def _beat(time, rr, label="N"):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def _steady(n, rr=0.8):
    return [_beat(i * rr, rr if i else None) for i in range(n)]


def _channels(n_channels=3, seconds=100.0):
    t = np.arange(0, seconds, 1 / FS)
    return np.stack([np.sin(2 * np.pi * 1.25 * t) * (k + 1) * 0.3 + 13.0 for k in range(n_channels)])


def _draw(channels, beats, **kw):
    fig = plt.figure(figsize=(8.5, 11))
    axes = draw_strip(fig, GridSpec(1, 1, figure=fig)[0], channels, ["Ch 1", "Ch 2", "Ch 3"][: channels.shape[0]], FS, beats=beats, **kw)
    return fig, axes


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_strip_window_is_six_seconds_centred_and_clamped_at_the_start():
    assert strip_window(10.0, [10.0], 100.0) == (7.0, 13.0)
    assert strip_window(0.5, [0.5], 100.0) == (0.0, 6.0)
    assert strip_window(99.0, [99.0], 100.0) == (96.0, 100.0)


def test_strip_window_widens_for_marks_further_apart_than_the_window():
    assert strip_window(50.0, [41.0, 59.0], 100.0) == (40.0, 60.0)  # 18 s span + 1 s margin each side


def test_scale_label_states_the_speed_and_gain_actually_used():
    assert scale_label(6.0, 3.0) == "25 mm/s · 10 mm/mV"
    assert scale_label(20.0, 3.0) == "8 mm/s · 10 mm/mV"
    assert scale_label(6.0, 5.0) == "25 mm/s · 6 mm/mV"


def test_channel_range_grows_to_the_next_millivolt_above_the_tallest_lead():
    assert channel_range_mv([np.array([0.0, 1.0]), np.array([-0.5, 0.5])]) == 3.0
    assert channel_range_mv([np.array([0.0, 3.5])]) == 4.0


def test_one_panel_per_lead_with_the_analysis_lead_named():
    fig, axes = _draw(_channels(3), _steady(100), center_time=10.0, mark_times=[10.4])
    assert len(axes) == 3
    assert axes[0].get_ylabel() == "Ch 1 (analysis)"
    assert axes[1].get_ylabel() == "Ch 2"
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[2])
    plt.close(fig)


def test_grid_is_ecg_paper_and_the_trace_is_baseline_corrected():
    fig, axes = _draw(_channels(1), _steady(100), center_time=10.0)
    ax = axes[0]
    # Minor ticks that coincide with a major tick are dropped by matplotlib,
    # so check the union: 1 mm (0.04 s) squares with 5 mm (0.2 s) majors.
    ticks = np.unique(np.concatenate([ax.xaxis.get_minorticklocs(), ax.xaxis.get_majorticklocs()]))
    ticks = ticks[(ticks >= 0) & (ticks <= 6)]
    assert np.allclose(np.diff(ticks), 0.04)
    assert np.allclose(np.diff(ax.xaxis.get_majorticklocs()), 0.2)
    assert np.allclose(np.diff(ax.yaxis.get_majorticklocs()), 0.5)
    y = ax.lines[0].get_ydata()
    assert abs(np.median(y)) < 0.01  # the 13 mV offset is gone
    assert ax.get_ylim()[1] - ax.get_ylim()[0] == 3.0
    plt.close(fig)


def test_beats_in_the_window_get_labels_and_flagged_beats_get_bands_on_every_lead():
    beats = _steady(100)
    beats[13] = _beat(beats[13].time, 0.8, "V")
    beats[14] = _beat(beats[14].time, 0.8, None)
    fig, axes = _draw(_channels(3), beats, center_time=10.4, mark_times=[10.4])
    labels = [t for t in _texts(axes[0]) if t in ("N", "V", "?")]
    assert labels.count("V") == 1 and labels.count("?") == 1
    assert 5 <= labels.count("N") <= 7  # the 6 s window holds ~7 beats at 0.8 s
    assert all(len(ax.patches) == 1 for ax in axes)
    plt.close(fig)


def test_rr_intervals_are_printed_around_the_flagged_beat():
    beats = _steady(100)
    beats[13] = _beat(beats[13].time, 0.8, "V")
    fig, axes = _draw(_channels(1), beats, center_time=10.4, mark_times=[10.4])
    assert _texts(axes[0]).count("0.80 s") == 2  # the RR into the beat and the RR out of it
    plt.close(fig)


def test_pause_is_bracketed_with_its_length():
    beats = _steady(40) + [_beat(31.2 + 3.0, 3.0), _beat(35.0, 0.8)]
    fig, axes = _draw(_channels(2), beats, center_time=32.7, mark_times=[31.2, 34.2], pause=(31.2, 34.2))
    assert "3.00 s gap" in _texts(axes[0])
    plt.close(fig)


def test_widened_strip_keeps_every_mark_on_the_waveform():
    fig, axes = _draw(_channels(1), _steady(100), center_time=50.0, mark_times=[41.0, 59.0])
    t = axes[0].lines[0].get_xdata()
    assert abs(t[-1] - t[0] - 20.0) < 0.05
    xs = sorted(p.get_x() for p in axes[0].patches)
    assert abs(xs[0] - (1.0 - 0.08)) < 1e-6 and abs(xs[1] - (19.0 - 0.08)) < 1e-6
    plt.close(fig)
