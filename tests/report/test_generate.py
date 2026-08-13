import os
import tempfile
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.generate import write_report


def _beat(time, rr, label, qrs=0.08):
    return Beat(time=time, rr_interval=rr, qrs_duration=qrs, label=label)


def test_writes_markdown_report_with_summary_stats():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")

    with tempfile.TemporaryDirectory() as out_dir:
        report_path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        assert os.path.exists(report_path)
        content = open(report_path).read()
        assert "PVC" in content
        assert "1" in content  # pvc_count appears somewhere in the stats


def test_generates_strip_plot_for_each_pvc_run():
    beats = [_beat(0.0, None, "N")] + [
        _beat(i * 0.8, 0.8, "V") for i in range(1, 4)
    ]  # a triplet
    summary = summarize(beats, dog_weight_class="medium")
    import numpy as np
    samples = np.sin(np.linspace(0, 20, 2000))  # dummy waveform, just needs a shape

    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=samples, sample_rate=100.0)
        plot_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(plot_files) >= 1


def test_report_with_zero_pvc_beats_has_no_flagged_section_and_no_plots():
    """No PVCs at all: report should still write cleanly, with no "Flagged
    events" section and no plot files, rather than crashing or emitting an
    empty section."""
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.6, 0.8, "N")]
    summary = summarize(beats, dog_weight_class="medium")

    with tempfile.TemporaryDirectory() as out_dir:
        report_path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        assert os.path.exists(report_path)
        content = open(report_path).read()
        assert "Flagged events" not in content
        plot_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert plot_files == []


def test_isolated_single_pvc_is_not_a_flagged_event():
    """A lone PVC (not part of a run of 2+) is counted in the summary stats
    but _flagged_runs explicitly excludes it - confirm the report actually
    honors that filter rather than listing every single PVC as an event."""
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.pvc_count == 1

    with tempfile.TemporaryDirectory() as out_dir:
        report_path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        content = open(report_path).read()
        assert "Flagged events" not in content
        assert "Event 1" not in content


def test_no_plots_written_when_samples_missing_even_with_multibeat_run():
    """samples=None/sample_rate=None must be a genuine report-only mode: a
    flagged multi-beat run is still listed in the markdown text, but zero
    PNGs get written - not a crash, and not an empty/broken PNG."""
    beats = [_beat(0.0, None, "N")] + [
        _beat(i * 0.8, 0.8, "V") for i in range(1, 4)
    ]  # a triplet
    summary = summarize(beats, dog_weight_class="medium")

    with tempfile.TemporaryDirectory() as out_dir:
        report_path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        content = open(report_path).read()
        assert "Flagged events" in content
        assert "Event 1" in content
        plot_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert plot_files == []


def test_plot_strip_clamps_near_start_without_wraparound(monkeypatch):
    """Mirrors the off-by-one class of bug already found and fixed in
    detect.py's _qrs_width search-window clamping: for an event very close
    to the start of the recording, (center_time - half_window) * sample_rate
    goes negative. If that negative value isn't clamped to 0 before slicing,
    Python interprets a moderately negative start index as "count from the
    end", silently pulling in unrelated data from the tail of the recording
    instead of an error. Verify the clamp actually prevents that."""
    import numpy as np
    import matplotlib.pyplot as plt
    from canine_holter.report.generate import _plot_strip

    sample_rate = 100.0
    n = 1000  # 10 second recording
    samples = np.zeros(n)
    samples[:50] = 1.0  # distinct signal right at the start of the recording
    samples[-50:] = -1.0  # distinct signal at the very end of the recording

    captured = {}
    orig_close = plt.close

    def capture_then_close(fig):
        line = fig.axes[0].lines[0]
        captured["ydata"] = line.get_ydata()
        orig_close(fig)

    monkeypatch.setattr(plt, "close", capture_then_close)

    with tempfile.TemporaryDirectory() as out_dir:
        _plot_strip(
            samples,
            sample_rate,
            center_time=0.5,  # near the start; half-window is 3s
            out_path=os.path.join(out_dir, "strip.png"),
        )

    ydata = captured["ydata"]
    assert len(ydata) > 0
    assert 1.0 in ydata  # the near-start signal should be included
    assert -1.0 not in ydata  # must not wrap around and pull in end-of-recording data
