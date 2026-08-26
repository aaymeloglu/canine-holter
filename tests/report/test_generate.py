import os
import tempfile
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.common import EVENTS_TITLE, EXTREMES_TITLE, ISOLATED_TITLE
from canine_holter.quality.gate import SignalQuality
from canine_holter.report.generate import SummaryRow, build_content, write_report


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


def _steady(n, rr):
    return [_beat(i * rr, rr if i else None, "N") for i in range(n)]


def _with_run(beats, start_index, n, rr):
    out = list(beats)
    t = beats[start_index].time
    for k in range(n):
        out[start_index + k] = _beat(t + k * rr, beats[start_index].rr_interval if k == 0 else rr, "V", qrs=0.12)
    return out


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
    assert section.labels[3] == "Fastest run: 4 beats, 240 bpm, t=10s"
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


HOURLY_HEADER = [
    "Hour", "Analyzed (min)", "Beats", "Min HR", "Mean HR", "Max HR", "PVCs", "Couplets", "Runs (3+)", "Pauses",
]


def test_hourly_table_rows_with_wall_clock_labels():
    beats = _with_run(_steady(int(1.5 * 3600 / 0.5) + 1, 0.5), 100, 3, 0.25)  # 1.5 h at 120 bpm
    start = datetime(2026, 8, 23, 15, 33, 8)
    content = build_content(beats, summarize(beats), start)
    assert content.hourly_header == HOURLY_HEADER
    assert content.hourly_rows[0] == ["15:33-16:33", "60.0", "7200", "120", "120", "120", "3", "0", "1", "0"]
    assert content.hourly_rows[1][:3] == ["16:33-17:03", "30.0", "3601"]


def test_hourly_table_uses_elapsed_labels_without_a_start_time():
    beats = _steady(int(1.5 * 3600 / 0.5) + 1, 0.5)
    rows = build_content(beats, summarize(beats), None).hourly_rows
    assert [r[0] for r in rows] == ["0:00-1:00", "1:00-1:30"]


def test_hourly_table_blanks_rates_for_an_hour_with_too_few_beats():
    beats = _steady(7200, 0.5) + [_beat(3700.0, 100.5, "N")]
    rows = build_content(beats, summarize(beats), None).hourly_rows
    assert rows[1] == ["1:00-1:01", "1.7", "1", "-", "-", "-", "0", "0", "0", "1"]


def test_hourly_table_empty_for_no_beats():
    content = build_content([], summarize([]), None)
    assert content.hourly_header == HOURLY_HEADER
    assert content.hourly_rows == []


# --- summary panels -----------------------------------------------------------
def _rows(content, title):
    group = next(g for g in content.summary_groups if g.title == title)
    return {r.label: r for r in group.rows}


def test_content_has_four_summary_groups_in_order_and_a_footer():
    beats = _couplet_and_single()
    content = build_content(beats, summarize(beats), None)
    assert [g.title for g in content.summary_groups] == ["Recording", "Heart rate", "Ventricular ectopy", "Pauses"]
    assert any("not a diagnosis" in line for line in content.footer_lines)


def test_recording_group_reports_duration_analyzed_and_excluded():
    beats = _couplet_and_single()
    q = SignalQuality(7200.0, ((0.0, 60.0), (7140.0, 7200.0)))
    rows = _rows(build_content(beats, summarize(beats, quality=q), datetime(2026, 8, 23, 15, 33, 8)), "Recording")
    assert rows["Start"].value == "2026-08-23 15:33:08"
    assert rows["Duration"].value == "2h 0m"
    assert rows["Analyzed"] == SummaryRow("Analyzed", "1h 58m (98%)", ">= 20 h", "caution")
    assert rows["Excluded"] == SummaryRow("Excluded", "0h 2m", "artifact / off-body")
    assert rows["Total beats"].value == "7"


def test_recording_group_without_start_says_unknown():
    beats = _couplet_and_single()
    assert _rows(build_content(beats, summarize(beats), None), "Recording")["Start"].value == "unknown"


def test_ectopy_group_values_references_and_statuses():
    beats = _couplet_and_single()
    rows = _rows(build_content(beats, summarize(beats), None), "Ventricular ectopy")
    assert rows["PVCs"] == SummaryRow("PVCs", "3 (42.86%)")
    assert rows["PVCs per 24 h"] == SummaryRow("PVCs per 24 h", "n/a", "needs >= 20 h analyzed")
    assert rows["Couplets"] == SummaryRow("Couplets", "1", "0", "alert")
    assert rows["Triplets"] == SummaryRow("Triplets", "0", "0", "ok")
    assert rows["VT runs (4+)"] == SummaryRow("VT runs (4+)", "0", "0", "ok")
    assert rows["Longest run"] == SummaryRow("Longest run", "none")
    assert rows["Fastest run"] == SummaryRow("Fastest run", "none", "<180 bpm", "ok")


def test_ectopy_group_scales_pvcs_by_analyzed_time_and_colors_the_band():
    beats = _steady(24 * 3600 * 2, 0.5)  # 24 h at 120 bpm
    beats = [_beat(b.time, b.rr_interval, "V" if i % 100 == 0 else "N") for i, b in enumerate(beats)]
    q = SignalQuality(24 * 3600.0, ((0.0, 4 * 3600.0),))  # 20 h analyzed
    rows = _rows(build_content(beats, summarize(beats, quality=q), None), "Ventricular ectopy")
    assert rows["PVCs per 24 h"].reference == "<50 | 50-300 | >300"
    assert rows["PVCs per 24 h"].value == "2074 (scaled from 20h 0m analyzed)"
    assert rows["PVCs per 24 h"].status == "alert"


def test_run_rows_show_beats_rate_and_time():
    beats = _with_run(_with_run(_steady(100, 0.5), 10, 5, 0.3), 50, 3, 0.25)
    rows = _rows(build_content(beats, summarize(beats), datetime(2026, 8, 23, 15, 33, 8)), "Ventricular ectopy")
    assert rows["Longest run"] == SummaryRow("Longest run", "5 beats, 200 bpm, 15:33:13")
    assert rows["Fastest run"] == SummaryRow("Fastest run", "3 beats, 240 bpm, 15:33:33", "<180 bpm", "alert")


def test_heart_rate_group_has_mean_and_timed_extremes():
    beats = _steady(10, 0.5)  # 120 bpm
    rows = _rows(build_content(beats, summarize(beats), datetime(2026, 8, 23, 15, 33, 8)), "Heart rate")
    assert rows["Mean"] == SummaryRow("Mean", "120 bpm")
    assert rows["Slowest"].value.startswith("120 bpm at 15:33:") and rows["Slowest"].reference == "5-beat median"
    assert rows["Fastest"].value.startswith("120 bpm at 15:33:")
    assert rows["Brady events"] == SummaryRow("Brady events", "0")
    assert rows["Tachy events"] == SummaryRow("Tachy events", "0")


def test_heart_rate_group_says_when_not_computed():
    beats = _steady(3, 0.5)
    rows = _rows(build_content(beats, summarize(beats), None), "Heart rate")
    assert rows["Heart rate"].value == "not computed (fewer than 5 beats with an RR)"


def test_pause_group_counts_and_colors_the_longest():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(3.77, 2.97, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "Pauses")
    assert rows["Pauses"] == SummaryRow("Pauses", "1", ">= 2.5 s")
    assert rows["Longest"] == SummaryRow("Longest", "2.97 s", "<2.5 | 2.5-5 | >5 s", "caution")


def test_hourly_header_and_rows_carry_analyzed_minutes():
    beats = _couplet_and_single()
    q = SignalQuality(4000.0, ((0.0, 60.0), (3940.0, 4000.0)))
    content = build_content(beats, summarize(beats, quality=q), None)
    assert content.hourly_header[:2] == ["Hour", "Analyzed (min)"]
    assert [row[1] for row in content.hourly_rows] == ["59.0", "5.7"]
