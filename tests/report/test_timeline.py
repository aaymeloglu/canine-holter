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
