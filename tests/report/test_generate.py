import os
import tempfile
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.common import EVENTS_TITLE, EXTREMES_TITLE, ISOLATED_TITLE
from canine_holter.report.generate import _summary_lines, build_content, write_report


def _beat(time, rr, label, qrs=0.08):
    return Beat(time=time, rr_interval=rr, qrs_duration=qrs, label=label)


def _couplet_and_single():
    return [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "V"),
        _beat(3.2, 0.8, "N"),
        _beat(4.0, 0.8, "V"),
        _beat(4.8, 0.8, "N"),
    ]


def test_write_report_writes_only_the_pdf():
    beats = _couplet_and_single()
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = write_report(beats, summary, out_dir, samples=np.zeros(1000), sample_rate=100.0)
        assert report_path == os.path.join(out_dir, "report.pdf")
        assert os.path.getsize(report_path) > 1000
        assert os.listdir(out_dir) == ["report.pdf"]


def test_write_report_without_samples_still_writes_only_the_pdf():
    beats = _couplet_and_single()
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summarize(beats), out_dir, samples=None, sample_rate=None)
        assert os.listdir(out_dir) == ["report.pdf"]


def test_content_summary_has_the_stats():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.6, 0.8, "V"), _beat(2.4, 0.8, "N")]
    content = build_content(beats, summarize(beats), None)
    assert "- Total beats: 4" in content.summary_lines
    assert "- PVC count: 1" in content.summary_lines


def test_isolated_single_pvc_goes_in_its_own_section_not_flagged_events():
    """A lone PVC is not a flagged event, but it is where classifier errors
    live, so it gets its own section and strip."""
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.6, 0.8, "V"), _beat(2.4, 0.8, "N")]
    content = build_content(beats, summarize(beats), None)
    assert [s.heading for s in content.sections] == [ISOLATED_TITLE]
    assert content.sections[0].labels == ["PVC 1: isolated PVC at ~t=1.6s"]
    assert [[b.time for b in run] for run in content.sections[0].runs] == [[1.6]]


def test_flagged_events_come_before_isolated_pvcs():
    content = build_content(_couplet_and_single(), summarize(_couplet_and_single()), None)
    assert [s.heading for s in content.sections] == [EXTREMES_TITLE, EVENTS_TITLE, ISOLATED_TITLE]
    assert content.sections[1].labels == ["Event 1: 2 consecutive PVCs at ~t=2.0s"]


def test_zero_pvcs_yields_no_sections():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.6, 0.8, "N")]
    assert build_content(beats, summarize(beats), None).sections == []


def test_isolated_pvc_strips_are_capped_and_the_heading_says_so():
    beats = [_beat(0.0, None, "N")]
    for i in range(1, 61):  # 30 isolated PVCs, each between normals
        beats.append(_beat(i * 0.8, 0.8, "V" if i % 2 else "N"))
    summary = summarize(beats)
    assert summary.pvc_count == 30
    section = build_content(beats, summary, None).sections[-1]
    assert section.heading == "Isolated PVCs (24 of 30 shown, evenly spaced through the recording)"
    assert len(section.runs) == 24
    assert len(section.labels) == 24


def test_flagged_event_uses_wall_clock_label_when_start_known():
    beats = [_beat(0.0, None, "N")] + [_beat(i * 0.8, 0.8, "V") for i in range(1, 3)]
    start = datetime(2026, 8, 23, 15, 33, 8)
    content = build_content(beats, summarize(beats), start)
    assert content.sections[0].labels == ["Event 1: 2 consecutive PVCs at ~15:33:09 (t=1.2s)"]


def test_content_summary_includes_start_and_duration():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(150.0, 0.8, "N")]
    content = build_content(beats, summarize(beats), datetime(2026, 8, 23, 15, 33, 8))
    assert "- Recording start: 2026-08-23 15:33:08" in content.summary_lines
    assert "- Duration: 0h 2m" in content.summary_lines


def test_content_summary_without_start_says_unknown():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    assert "- Recording start: unknown" in build_content(beats, summarize(beats), None).summary_lines


def test_summary_has_longest_pause_and_24h_line_and_reference_lines():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(3.77, 2.97, "N")]
    content = build_content(beats, summarize(beats), None)
    assert "- Longest pause: 2.97 s" in content.summary_lines
    assert "- PVCs per 24 h: not computed (recording is 0h 0m; needs >= 20 h)" in content.summary_lines
    assert any("under 50" in line for line in content.reference_lines)


def test_content_summary_lines_come_from_the_shared_helper():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    lines = _summary_lines(summary, start, duration_sec=0.8)
    assert lines[0] == "- Recording start: 2026-08-23 15:33:08"
    assert lines[1] == "- Duration: 0h 0m"
    assert build_content(beats, summary, start).summary_lines == lines


def _steady(n, rr):
    return [_beat(i * rr, rr if i else None, "N") for i in range(n)]


def _with_run(beats, start_index, n, rr):
    out = list(beats)
    t = beats[start_index].time
    for k in range(n):
        out[start_index + k] = _beat(t + k * rr, beats[start_index].rr_interval if k == 0 else rr, "V", qrs=0.12)
    return out


def test_summary_has_heart_rate_lines_with_times():
    beats = _steady(10, 0.5)  # 120 bpm
    start = datetime(2026, 8, 23, 15, 33, 8)
    lines = build_content(beats, summarize(beats), start).summary_lines
    assert "- Mean heart rate: 120 bpm" in lines
    assert any(l.startswith("- Slowest heart rate (5-beat median): 120 bpm at 15:33:") for l in lines)
    assert any(l.startswith("- Fastest heart rate (5-beat median): 120 bpm at 15:33:") for l in lines)


def test_summary_heart_rate_lines_say_when_not_computed():
    beats = _steady(3, 0.5)
    lines = build_content(beats, summarize(beats), None).summary_lines
    assert "- Heart rate: not computed (fewer than 5 beats with an RR interval)" in lines


def test_summary_has_longest_and_fastest_run_lines():
    beats = _with_run(_with_run(_steady(100, 0.5), 10, 5, 0.3), 50, 3, 0.25)
    lines = build_content(beats, summarize(beats), None).summary_lines
    assert "- Longest run: 5 PVCs at 200 bpm, starting t=5.0s" in lines
    assert "- Fastest run: 3 PVCs at 240 bpm, starting t=25.0s" in lines


def test_summary_run_lines_say_none():
    beats = _steady(10, 0.5)
    lines = build_content(beats, summarize(beats), None).summary_lines
    assert "- Longest run: none (no runs of 3+ PVCs)" in lines
    assert "- Fastest run: none (no runs of 3+ PVCs)" in lines


def test_reference_lines_distinguish_vt_from_slower_runs():
    lines = build_content(_steady(10, 0.5), summarize(_steady(10, 0.5)), None).reference_lines
    assert any("idioventricular" in l for l in lines)


def test_extremes_section_comes_first_with_max_min_pause_and_fastest_run():
    beats = _steady(100, 0.5)
    beats[40] = _beat(beats[40].time, 0.5, "N")
    # A 3 s pause: shift everything after beat 60 by 2.5 s.
    beats = beats[:61] + [_beat(b.time + 2.5, 3.0 if i == 0 else b.rr_interval, b.label) for i, b in enumerate(beats[61:])]
    beats = _with_run(beats, 20, 4, 0.25)
    summary = summarize(beats)
    content = build_content(beats, summary, None)
    section = content.sections[0]
    assert section.heading == EXTREMES_TITLE
    assert [l.split(":")[0] for l in section.labels] == [
        "Fastest heart rate", "Slowest heart rate", "Longest pause", "Fastest run"
    ]
    assert section.labels[2] == "Longest pause: 3.00 s, ending at t=33.0s"
    assert section.labels[3] == "Fastest run: 4 PVCs at 240 bpm, starting t=10.0s"
    # The pause strip marks both beats bracketing the gap; the run strip marks every PVC.
    assert [b.time for b in section.runs[2]] == [30.0, 33.0]
    assert [b.time for b in section.runs[3]] == [10.0, 10.25, 10.5, 10.75]
    assert [s.heading for s in content.sections[1:]] == [EVENTS_TITLE]


def test_extremes_section_omits_what_is_not_there():
    beats = _steady(10, 0.5)  # no pause above threshold, no runs
    section = build_content(beats, summarize(beats), None).sections[0]
    assert [l.split(":")[0] for l in section.labels] == ["Fastest heart rate", "Slowest heart rate"]


def test_no_extremes_section_when_heart_rate_not_computed():
    beats = _steady(3, 0.5)
    assert build_content(beats, summarize(beats), None).sections == []
