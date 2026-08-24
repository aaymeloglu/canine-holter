import os
import tempfile
from datetime import datetime
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import ArrhythmiaSummary
from canine_holter.report.timeline import plot_timeline


def _beat(time, rr, label):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def _summary(**kw):
    base = dict(
        total_beats=0, pvc_count=0, pvc_burden_pct=0.0, couplets=0, triplets=0,
        vtach_runs=0, bradycardia_events=[], tachycardia_events=[], pauses=[],
    )
    base.update(kw)
    return ArrhythmiaSummary(**base)


def _render(beats, summary, start_time):
    with tempfile.TemporaryDirectory() as out_dir:
        out = os.path.join(out_dir, "timeline.png")
        plot_timeline(beats, summary, start_time, out)
        assert os.path.getsize(out) > 0


def test_renders_with_every_event_type_and_wall_clock():
    beats = [
        _beat(i * 0.8, 0.8 if i else None, "V" if i % 10 == 0 else "N") for i in range(300)
    ]
    summary = _summary(
        bradycardia_events=[(10.0, 30.0)],
        tachycardia_events=[(100.0, 100.5)],
        pauses=[50.0, 60.0],
    )
    _render(beats, summary, datetime(2026, 8, 23, 15, 33, 8))


def test_renders_without_start_time():
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(300)]
    _render(beats, _summary(), None)


def test_renders_with_no_events_and_no_beats():
    _render([], _summary(), None)


def test_heart_rate_trend_bins_by_minute_and_leaves_sparse_bins_as_gaps():
    from canine_holter.report.timeline import _heart_rate_trend
    import numpy as np

    # 75 bpm steady for the first minute, nothing in minute 2, one lone beat
    # in minute 3 (too few RR intervals to call a rate).
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(75)]
    beats.append(_beat(150.0, 0.8, "N"))
    centers, bpm = _heart_rate_trend(beats)
    assert list(centers) == [30.0, 90.0, 150.0]
    assert bpm[0] == 75.0
    assert np.isnan(bpm[1]) and np.isnan(bpm[2])


def test_single_bin_recording_draws_a_visible_point(monkeypatch):
    """A recording shorter than one HR bin yields one trend point; a bare
    line through one point renders nothing, so it must carry a marker."""
    import matplotlib.pyplot as plt

    captured = {}
    orig_close = plt.close

    def capture_then_close(fig):
        captured["marker"] = fig.axes[0].lines[0].get_marker()
        orig_close(fig)

    monkeypatch.setattr(plt, "close", capture_then_close)
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(50)]
    _render(beats, _summary(), None)
    assert captured["marker"] not in (None, "None", "")


def test_tick_interval_scales_with_recording_span():
    from canine_holter.report.timeline import _tick_interval_minutes

    assert _tick_interval_minutes(25) == 1
    assert _tick_interval_minutes(8 * 60) == 1
    assert _tick_interval_minutes(45 * 60) == 5
    assert _tick_interval_minutes(2.5 * 3600) == 15
    assert _tick_interval_minutes(8 * 3600) == 60
    assert _tick_interval_minutes(24 * 3600) == 240


def test_short_wall_clock_recording_keeps_axis_to_its_own_span(monkeypatch):
    """Regression: a 25 s recording with a date axis used to autoscale to
    +/- 5% of the *date number* (years), and the minute locator then tried
    to lay out tens of thousands of ticks - minutes of CPU per report."""
    import matplotlib.pyplot as plt

    captured = {}
    orig_close = plt.close

    def capture_then_close(fig):
        ax = fig.axes[1]
        captured["xlim_days"] = ax.get_xlim()[1] - ax.get_xlim()[0]
        captured["n_ticks"] = len(ax.get_xticks())
        orig_close(fig)

    monkeypatch.setattr(plt, "close", capture_then_close)
    beats = [_beat(i * 0.5, 0.5 if i else None, "N") for i in range(50)]  # 25 s
    _render(beats, _summary(), datetime(2010, 7, 8, 11, 12, 50))
    assert captured["xlim_days"] < 1.0 / 24  # well under an hour
    assert captured["n_ticks"] < 20


def test_draw_timeline_draws_into_given_figure_region():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from canine_holter.report.timeline import draw_timeline

    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(2, 1, figure=fig)
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(300)]
    ax_hr, ax_ev = draw_timeline(fig, gs[1], beats, _summary(), None)
    assert ax_hr.figure is fig and ax_ev.figure is fig
    assert len(fig.axes) == 2
    plt.close(fig)
