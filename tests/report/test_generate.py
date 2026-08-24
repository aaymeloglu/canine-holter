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
        assert report_path == os.path.join(out_dir, "report.pdf")
        assert os.path.exists(report_path)
        content = open(os.path.join(out_dir, "report.md")).read()
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
        content = open(os.path.join(out_dir, "report.md")).read()
        assert "Flagged events" not in content
        strip_files = [f for f in os.listdir(out_dir) if f.startswith("event_")]
        assert strip_files == []


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
        write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        content = open(os.path.join(out_dir, "report.md")).read()
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
        write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        content = open(os.path.join(out_dir, "report.md")).read()
        assert "Flagged events" in content
        assert "Event 1" in content
        strip_files = [f for f in os.listdir(out_dir) if f.startswith("event_")]
        assert strip_files == []


def test_plot_strip_clamps_near_start_without_wraparound(monkeypatch):
    """Mirrors the off-by-one class of bug already found and fixed in
    detect.py's _qrs_width search-window clamping: for an event very close
    to the start of the recording, (center_time - half_window) * sample_rate
    goes negative. If that negative value isn't clamped to 0 before slicing,
    Python interprets it as "count from the end" rather than raising an
    error. For a short strip window on a long recording (this test's case),
    the normalized negative start lands past the (also-clamped) stop index,
    so unclamped code would silently produce an empty plot rather than an
    error - not, in general, a guarantee of pulling in real tail data,
    though a window comparable to the recording length could produce that
    too. Either way, verify the max(0, ...) clamp prevents the empty-plot
    failure mode and the strip genuinely starts at sample 0."""
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
            title="strip",
        )

    ydata = captured["ydata"]
    assert len(ydata) > 0
    assert 1.0 in ydata  # the near-start signal should be included
    assert -1.0 not in ydata  # must not wrap around and pull in end-of-recording data


# --- wall-clock labels -------------------------------------------------------
from datetime import datetime
from canine_holter.report.generate import format_time


def test_format_time_with_start_gives_wall_clock_and_elapsed():
    start = datetime(2026, 8, 23, 15, 33, 8)
    assert format_time(8232.8, start) == "17:50:20 (t=8232.8s)"


def test_format_time_without_start_gives_elapsed_only():
    assert format_time(8232.8, None) == "t=8232.8s"


def test_report_summary_includes_start_and_duration():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(150.0, 0.8, "N")]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=None, sample_rate=None, start_time=start)
        content = open(os.path.join(out_dir, "report.md")).read()
        assert "- Recording start: 2026-08-23 15:33:08" in content
        assert "- Duration: 0h 2m" in content


def test_report_summary_without_start_says_unknown():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        assert "- Recording start: unknown" in open(os.path.join(out_dir, "report.md")).read()


def test_flagged_event_uses_wall_clock_label_when_start_known():
    beats = [_beat(0.0, None, "N")] + [_beat(i * 0.8, 0.8, "V") for i in range(1, 3)]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=None, sample_rate=None, start_time=start)
        assert "Event 1: 2 consecutive PVCs at ~15:33:09 (t=1.2s)" in open(os.path.join(out_dir, "report.md")).read()


def test_report_links_timeline_png():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        content = open(os.path.join(out_dir, "report.md")).read()
        assert "## Timeline" in content
        assert "![timeline](timeline.png)" in content
        assert os.path.exists(os.path.join(out_dir, "timeline.png"))


def test_summary_lines_match_markdown_summary_block():
    from canine_holter.report.generate import _summary_lines

    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    lines = _summary_lines(summary, start, duration_sec=0.8)
    assert lines[0] == "- Recording start: 2026-08-23 15:33:08"
    assert lines[1] == "- Duration: 0h 0m"
    assert "- Total beats: 2" in lines
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=None, sample_rate=None, start_time=start)
        content = open(os.path.join(out_dir, "report.md")).read()
    for line in lines:
        assert line in content
