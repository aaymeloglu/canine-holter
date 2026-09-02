import os
import pytest
import tempfile
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.quality.gate import SignalQuality
from canine_holter.report.generate import (
    ESCAPE_TITLE,
    EVENTS_TITLE,
    EXTREMES_TITLE,
    HOURLY_TITLE,
    HOW_TO_READ_STRIPS,
    ISOLATED_TITLE,
    MAX_HOURLY_STRIPS,
    SummaryRow,
    build_content,
    select_evenly,
    short_time,
)
from canine_holter.report.pdf import write_report


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
    assert [s.heading for s in content.sections] == [ISOLATED_TITLE, HOURLY_TITLE]
    assert [item.caption.title for item in content.sections[0].items] == ["Isolated PVC 1 · t=2s"]
    assert [[b.time for b in item.run] for item in content.sections[0].items] == [[1.6]]


def test_flagged_events_come_before_isolated_pvcs():
    content = build_content(_couplet_and_single(), summarize(_couplet_and_single()), None)
    assert [s.heading for s in content.sections] == [EXTREMES_TITLE, EVENTS_TITLE, ISOLATED_TITLE, HOURLY_TITLE]
    assert [item.caption.title for item in content.sections[1].items] == ["Event 1 · t=2s"]


def test_zero_pvcs_yields_only_the_hourly_strips():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.6, 0.8, "N")]
    assert [s.heading for s in build_content(beats, summarize(beats), None).sections] == [HOURLY_TITLE]


def test_isolated_pvc_strips_are_capped_and_the_heading_says_so():
    beats = [_beat(0.0, None, "N")]
    for i in range(1, 61):  # 30 isolated PVCs, each between normals
        beats.append(_beat(i * 0.8, 0.8, "V" if i % 2 else "N"))
    summary = summarize(beats)
    assert summary.pvc_count == 30
    section = build_content(beats, summary, None).sections[-2]  # the hourly strips come after
    assert section.heading == "Isolated PVCs (24 of 30 shown, evenly spaced through the recording)"
    assert len(section.items) == 24


def test_flagged_event_uses_wall_clock_label_when_start_known():
    beats = [_beat(0.0, None, "N")] + [_beat(i * 0.8, 0.8, "V") for i in range(1, 3)]
    start = datetime(2026, 8, 23, 15, 33, 8)
    content = build_content(beats, summarize(beats), start)
    assert [item.caption.title for item in content.sections[0].items] == ["Event 1 · 15:33:08"]


def test_select_evenly_spreads_picks_across_a_long_list():
    assert select_evenly([1, 2, 3], 24) == [1, 2, 3]
    assert select_evenly(list(range(30)), 4) == [0, 10, 19, 29]


def test_short_time_is_clock_with_start_and_elapsed_without():
    assert short_time(8183.2, datetime(2026, 8, 23, 15, 33, 8)) == "17:49:31"
    assert short_time(8183.2, None) == "t=8183s"


def test_primer_covers_the_grid_the_leads_and_the_caveat_in_page_width_lines():
    text = "\n".join(HOW_TO_READ_STRIPS)
    assert "0.2 s" in text and "three" in text and "provisional" in text
    assert all(len(line) <= 100 for line in HOW_TO_READ_STRIPS)


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
    assert [item.caption.title.split(" · ")[0] for item in section.items] == [
        "Fastest heart rate", "Slowest heart rate", "Longest pause", "Fastest run"
    ]
    assert section.items[2].caption.title == "Longest pause · t=33s"
    assert section.items[3].caption.title == "Fastest run · t=10s"
    # The pause strip marks both beats bracketing the gap; the run strip marks every PVC.
    assert [b.time for b in section.items[2].run] == [30.0, 33.0]
    assert [b.time for b in section.items[3].run] == [10.0, 10.25, 10.5, 10.75]
    assert [s.heading for s in content.sections[1:]] == [EVENTS_TITLE, HOURLY_TITLE]


def test_extremes_section_omits_what_is_not_there():
    beats = _steady(10, 0.5)  # no pause above threshold, no runs
    section = build_content(beats, summarize(beats), None).sections[0]
    assert [item.caption.title.split(" · ")[0] for item in section.items] == ["Fastest heart rate", "Slowest heart rate"]


def test_no_extremes_section_when_heart_rate_not_computed():
    beats = _steady(3, 0.5)
    assert [s.heading for s in build_content(beats, summarize(beats), None).sections] == [HOURLY_TITLE]


HOURLY_HEADER = [
    "Hour", "Analyzed (min)", "Beats", "Min HR", "Mean HR", "Max HR", "PVCs", "Couplets", "Runs (3+)", "Escapes",
    "Pauses",
]


def test_hourly_table_rows_with_wall_clock_labels():
    beats = _with_run(_steady(int(1.5 * 3600 / 0.5) + 1, 0.5), 100, 3, 0.25)  # 1.5 h at 120 bpm
    start = datetime(2026, 8, 23, 15, 33, 8)
    content = build_content(beats, summarize(beats), start)
    assert content.hourly_header == HOURLY_HEADER
    assert content.hourly_rows[0] == ["15:33-16:33", "60.0", "7200", "120", "120", "120", "3", "0", "1", "0", "0"]
    assert content.hourly_rows[1][:3] == ["16:33-17:03", "30.0", "3601"]


def test_hourly_table_uses_elapsed_labels_without_a_start_time():
    beats = _steady(int(1.5 * 3600 / 0.5) + 1, 0.5)
    rows = build_content(beats, summarize(beats), None).hourly_rows
    assert [r[0] for r in rows] == ["0:00-1:00", "1:00-1:30"]


def test_hourly_table_blanks_rates_for_an_hour_with_too_few_beats():
    beats = _steady(7200, 0.5) + [_beat(3700.0, 100.5, "N")]
    rows = build_content(beats, summarize(beats), None).hourly_rows
    assert rows[1] == ["1:00-1:01", "1.7", "1", "-", "-", "-", "0", "0", "0", "0", "1"]


def test_hourly_table_empty_for_no_beats():
    content = build_content([], summarize([]), None)
    assert content.hourly_header == HOURLY_HEADER
    assert content.hourly_rows == []


# --- summary panels -----------------------------------------------------------
def _rows(content, title):
    group = next(g for g in content.summary_groups if g.title == title)
    return {r.label: r for r in group.rows}


def test_content_has_six_summary_groups_in_order_and_a_footer():
    beats = _couplet_and_single()
    content = build_content(beats, summarize(beats), None)
    assert [g.title for g in content.summary_groups] == [
        "Recording", "Heart rate", "Ventricular ectopy", "Supraventricular ectopy", "Pauses", "RR variability",
    ]
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


def test_recording_group_says_how_long_the_recorder_ran_when_a_tail_was_trimmed():
    beats = _couplet_and_single()
    q = SignalQuality(7200.0, ((0.0, 60.0), (7140.0, 7200.0)), trimmed_sec=360000.0)
    rows = _rows(build_content(beats, summarize(beats, quality=q), None), "Recording")
    assert rows["Duration"].value == "2h 0m"
    assert rows["Recorder ran"] == SummaryRow("Recorder ran", "102h 0m", "off-body tail trimmed")


def test_recording_group_without_start_says_unknown():
    beats = _couplet_and_single()
    rows = _rows(build_content(beats, summarize(beats), None), "Recording")
    assert rows["Start"].value == "unknown"
    assert "Recorder ran" not in rows


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
    assert rows["PVCs per 24 h"].value == "2074"
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
    assert rows["Brady events"].value == "0" and rows["Tachy events"].value == "0"


def test_heart_rate_group_shares_and_event_rules_name_the_thresholds():
    beats = _steady(10, 0.5)  # 120 bpm: 6 windows, none slow or fast for a medium dog
    rows = _rows(build_content(beats, summarize(beats), None), "Heart rate")
    assert rows["Under 50 bpm"] == SummaryRow("Under 50 bpm", "0 (0%)", "5-beat median")
    assert rows["Over 160 bpm"] == SummaryRow("Over 160 bpm", "0 (0%)", "5-beat median")
    assert rows["Brady events"] == SummaryRow("Brady events", "0", "3+ beats < 50 bpm")
    assert rows["Tachy events"] == SummaryRow("Tachy events", "0", "3+ beats > 160 bpm")


def test_heart_rate_shares_show_count_and_percent():
    beats = _steady(10, 0.3)  # 200 bpm: every window is fast
    rows = _rows(build_content(beats, summarize(beats), None), "Heart rate")
    assert rows["Over 160 bpm"].value == "5 (100%)"


def test_supraventricular_group_says_not_assessed():
    beats = _couplet_and_single()
    rows = _rows(build_content(beats, summarize(beats), None), "Supraventricular ectopy")
    assert rows["SVPBs"] == SummaryRow("SVPBs", "not assessed", "needs P-wave analysis")


def test_pause_group_counts_pauses_over_five_seconds():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(3.77, 2.97, "N"), _beat(9.77, 6.0, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "Pauses")
    assert rows["Pauses"].value == "2"
    assert rows["Pauses > 5 s"] == SummaryRow("Pauses > 5 s", "1")
    assert rows["Longest"].status == "alert"


def test_variability_group_rounds_and_counts_nn_intervals():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(1.7, 0.9, "N"), _beat(2.5, 0.8, "N"), _beat(3.5, 1.0, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "RR variability")
    assert rows["SDNN"] == SummaryRow("SDNN", "83 ms", "4 NN intervals")
    assert rows["RMSSD"] == SummaryRow("RMSSD", "141 ms")
    assert rows["pNN50"] == SummaryRow("pNN50", "100%")


def test_variability_group_says_when_not_computed():
    beats = _steady(3, 0.5)
    rows = _rows(build_content(beats, summarize(beats), None), "RR variability")
    assert rows["RR variability"].value == "not computed (fewer than 2 successive NN differences)"


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


# --- strip captions -----------------------------------------------------------
def _captions(content, heading):
    section = next(s for s in content.sections if s.heading == heading)
    return [item.caption for item in section.items]


def test_isolated_pvc_caption_quotes_the_measurements_behind_the_label():
    beats = _steady(20, 0.8)
    beats[10] = _beat(8.0, 0.4, "V", qrs=0.12)
    caption = _captions(build_content(beats, summarize(beats), None), ISOLATED_TITLE)[0]
    assert caption.title == "Isolated PVC 1 · t=8s"
    assert caption.what == (
        "The marked beat arrived 0.40 s after the beat before it (typical here 0.80 s) and its QRS"
        " lasts 0.12 s (typical 0.08 s): early and wide is what makes it a PVC."
    )
    assert caption.significance == ""  # said once in the primer, not under every strip
    assert caption.status is None


def test_couplet_caption_names_both_beats_and_is_an_alert():
    beats = _with_run(_steady(100, 0.5), 50, 2, 0.25)
    caption = _captions(build_content(beats, summarize(beats), datetime(2026, 8, 23, 15, 33, 8)), EVENTS_TITLE)[0]
    assert caption.title.startswith("Event 1 · 15:33:")
    assert caption.what.startswith("The marked beats arrived 0.50 s and 0.25 s after the beat before them (typical here 0.50 s)")
    assert caption.significance == "Any couplet is worth a cardiologist's review, whatever the PVC count."
    assert caption.status == "alert"


def test_run_captions_are_timed_at_the_first_beat_like_the_summary_row():
    # A run's clock time must match page 1 (RunStats.start_time) wherever it
    # is shown; the run here straddles a second boundary so the centre reads
    # one second later than the first beat.
    beats = _with_run(_steady(100, 0.5), 50, 4, 0.4)  # V beats at 25.0-26.2 s
    start = datetime(2026, 8, 23, 15, 33, 8, 500000)
    content = build_content(beats, summarize(beats), start)
    assert _rows(content, "Ventricular ectopy")["Fastest run"].value == "4 beats, 150 bpm, 15:33:33"
    assert _captions(content, EXTREMES_TITLE)[-1].title == "Fastest run · 15:33:33"
    assert _captions(content, EVENTS_TITLE)[0].title == "Event 1 · 15:33:33"


def test_run_caption_status_follows_the_vt_rate_line():
    fast = _with_run(_steady(100, 0.5), 10, 4, 60 / 180)
    slow = _with_run(_steady(100, 0.5), 10, 4, 60 / 179)
    fast_caption = _captions(build_content(fast, summarize(fast), None), EVENTS_TITLE)[0]
    slow_caption = _captions(build_content(slow, summarize(slow), None), EVENTS_TITLE)[0]
    assert fast_caption.significance == "4 PVCs in a row at 180 bpm is ventricular tachycardia."
    assert fast_caption.status == "alert"
    assert slow_caption.significance.startswith("4 PVCs in a row at 179 bpm: an accelerated idioventricular rhythm")
    assert slow_caption.status == "caution"


def _with_pause(gap):
    beats = _steady(40, 0.5)
    t = beats[-1].time
    return beats + [_beat(t + gap, gap, "N")] + [_beat(t + gap + k * 0.5, 0.5, "N") for k in range(1, 10)]


def test_pause_caption_status_and_text_follow_the_pause_band():
    for gap, status in ((2.5, "caution"), (5.0, "caution"), (5.01, "alert")):
        beats = _with_pause(gap)
        captions = _captions(build_content(beats, summarize(beats), None), EXTREMES_TITLE)
        pause = next(c for c in captions if c.title.startswith("Longest pause"))
        assert pause.what == f"No beat for {gap:.2f} s."
        assert pause.status == status, gap
    assert "worth a cardiologist's review" in pause.significance


def test_heart_rate_extreme_captions_are_uncoloured_and_explain_context():
    beats = _steady(20, 0.5)
    captions = _captions(build_content(beats, summarize(beats), None), EXTREMES_TITLE)
    assert captions[0].title.startswith("Fastest heart rate · t=")
    assert captions[0].what == "120 bpm averaged over 5 beats."
    assert "play or excitement" in captions[0].significance and captions[0].status is None
    assert captions[1].title.startswith("Slowest heart rate · t=")
    assert "sinus arrhythmia" in captions[1].significance and captions[1].status is None


# --- one strip per hour --------------------------------------------------------


def test_hourly_strips_come_last_one_per_hour_at_the_hours_first_beat():
    beats = _steady(int(2.5 * 3600 / 0.5), 0.5)  # 2.5 h at 120 bpm
    content = build_content(beats, summarize(beats, start_time=datetime(2026, 8, 27, 10, 18, 49)), datetime(2026, 8, 27, 10, 18, 49))
    section = content.sections[-1]
    assert section.heading == HOURLY_TITLE
    assert [item.caption.title for item in section.items] == [
        "Hour 10:18-11:00 · 10:18:49", "Hour 11:00-12:00 · 11:00:00", "Hour 12:00-12:48 · 12:00:00",
    ]
    assert [item.run[0].time for item in section.items] == [0.0, 2471.0, 6071.0]
    assert section.items[0].caption.what == "The first beats of the hour. This hour: 120-120 bpm, mean 120."
    assert section.items[0].caption.significance == ""
    assert not section.items[0].mark  # nothing in a rhythm sample is flagged, so nothing is shaded


def test_hourly_strips_skip_an_hour_without_beats_and_say_when_rates_are_missing():
    beats = [_beat(0.0, None, "N"), _beat(1.0, 1.0, "N"), _beat(7300.0, None, "N")]
    section = build_content(beats, summarize(beats), None).sections[-1]
    assert [item.caption.title for item in section.items] == ["Hour 0:00-1:00 · t=0s", "Hour 2:00-2:01 · t=7300s"]
    assert section.items[0].caption.what == "The first beats of the hour. Too few beats this hour for a rate."


def test_hourly_strips_are_capped_and_the_heading_says_so():
    beats = _steady(50 * 6, 600.0)  # 50 h, a beat every 10 min
    section = build_content(beats, summarize(beats), None).sections[-1]
    assert len(section.items) == MAX_HOURLY_STRIPS
    assert section.heading == f"{HOURLY_TITLE} ({MAX_HOURLY_STRIPS} of 50 shown, evenly spaced through the recording)"


def test_primer_explains_the_hourly_strips():
    assert any("every hour" in line for line in HOW_TO_READ_STRIPS)


# --- ventricular escape beats -------------------------------------------------


def _with_escape():
    """8 normals at 0.8 s, then a wide beat after 2.0 s, then a normal."""
    beats = _steady(8, 0.8)
    t = beats[-1].time
    return beats + [_beat(t + 2.0, 2.0, "E", qrs=0.14), _beat(t + 2.8, 0.8, "N")]


def test_ectopy_panel_counts_escape_beats_apart_from_pvcs():
    beats = _with_escape()
    rows = _rows(build_content(beats, summarize(beats), None), "Ventricular ectopy")
    assert rows["PVCs"].value == "0 (0.00%)"
    assert rows["Escape beats"] == SummaryRow("Escape beats", "1", "wide, RR >= 1.5x local")


def test_hourly_header_has_an_escapes_column_after_runs():
    beats = _with_escape()
    content = build_content(beats, summarize(beats), None)
    assert content.hourly_header[content.hourly_header.index("Runs (3+)") + 1] == "Escapes"
    assert content.hourly_rows[0][content.hourly_header.index("Escapes")] == "1"


def test_escape_section_comes_after_isolated_pvcs_and_before_the_hourly_strips():
    beats = _with_escape()
    beats = _with_run(beats, 3, 1, 0.5)
    content = build_content(beats, summarize(beats), None)
    assert [s.heading for s in content.sections] == [EXTREMES_TITLE, ISOLATED_TITLE, ESCAPE_TITLE, HOURLY_TITLE]
    item = content.sections[2].items[0]
    assert item.caption.title == "Escape beat 1 · t=8s"
    assert [b.time for b in item.run] == [pytest.approx(7.6)]
    assert item.caption.what == (
        "The marked beat arrived 2.00 s after the beat before it (typical here 0.80 s) and its QRS lasts"
        " 0.14 s (typical 0.08 s): wide and late is what makes it a ventricular escape beat."
    )
    assert "gap" in item.caption.significance and "cardiologist" in item.caption.significance
    assert len(item.caption.significance) <= 105  # one caption line; a third line pushes the strip into the next slot
    assert item.caption.status is None


def test_escape_strips_are_capped_and_the_heading_says_so():
    beats = _steady(8, 0.8)
    t = beats[-1].time
    for i in range(30):
        beats.append(_beat(t + 2.0, 2.0, "E", qrs=0.14))
        beats.append(_beat(t + 2.8, 0.8, "N"))
        t += 2.8
    section = next(s for s in build_content(beats, summarize(beats), None).sections if s.heading.startswith(ESCAPE_TITLE))
    assert section.heading == f"{ESCAPE_TITLE} (24 of 30 shown, evenly spaced through the recording)"


def test_primer_explains_the_escape_label():
    assert any("E for" in line for line in HOW_TO_READ_STRIPS)


# --- interpretation-report follow-ups -------------------------------------------


def _bridged_arrest():
    """8 normals at 0.8 s, an escape beat 2.0 s later, a normal 1.0 s after it, then normals."""
    beats = _steady(8, 0.8)
    t = beats[-1].time
    return beats + [_beat(t + 2.0, 2.0, "E", qrs=0.14), _beat(t + 3.0, 1.0, "N"), _beat(t + 3.8, 0.8, "N")]


def test_heart_rate_rows_show_the_60_bpm_line_before_the_class_threshold():
    beats = _steady(10, 1.0)  # 60 bpm: under neither line
    rows = build_content(beats, summarize(beats, dog_weight_class="large"), None).summary_groups[1].rows
    labels = [r.label for r in rows]
    assert labels.index("Under 60 bpm") < labels.index("Under 45 bpm") < labels.index("Over 150 bpm")


def test_class_threshold_row_is_omitted_when_it_is_the_60_bpm_line():
    beats = _steady(10, 1.0)
    rows = _rows(build_content(beats, summarize(beats, dog_weight_class="small"), None), "Heart rate")
    assert "Under 60 bpm" in rows and [k for k in rows if k.startswith("Under")] == ["Under 60 bpm"]


def test_pause_rows_report_the_longest_sinus_interval_and_the_arrest_count():
    beats = _bridged_arrest()
    rows = _rows(build_content(beats, summarize(beats), None), "Pauses")
    assert rows["Longest"].value == "2.00 s"
    assert rows["Longest sinus interval"] == SummaryRow(
        "Longest sinus interval", "3.00 s", "1 escape beat inside", "caution"
    )
    assert rows["Sinus arrests"] == SummaryRow("Sinus arrests", "1", "bridged by escape beats")


def test_longest_sinus_interval_row_says_when_no_escape_beat_bridged_it():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(3.77, 2.97, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "Pauses")
    assert rows["Longest sinus interval"] == SummaryRow("Longest sinus interval", "2.97 s", "the longest pause", "caution")


def test_extremes_get_a_bridged_arrest_strip_marking_the_escape_beat():
    beats = _bridged_arrest()
    section = build_content(beats, summarize(beats), None).sections[0]
    item = next(i for i in section.items if i.caption.title.startswith("Longest sinus interval"))
    assert item.caption.title == "Longest sinus interval · t=9s"
    assert item.pause == (pytest.approx(5.6), pytest.approx(8.6))
    assert [b.label for b in item.run] == ["E"]
    assert item.caption.what == "No sinus beat for 3.00 s; 1 ventricular escape beat filled the gap."
    assert item.caption.status == "caution"


def test_no_bridged_arrest_strip_when_the_longest_sinus_interval_is_a_plain_pause():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N"), _beat(3.77, 2.97, "N")] + [_beat(3.77 + 0.8 * k, 0.8, "N") for k in range(1, 8)]
    section = build_content(beats, summarize(beats), None).sections[0]
    assert not any(i.caption.title.startswith("Longest sinus interval") for i in section.items)


def test_ectopy_rows_count_escape_couplets_and_runs():
    beats = _steady(8, 0.8)
    t = beats[-1].time
    beats += [_beat(t + 2.0, 2.0, "E", qrs=0.14), _beat(t + 4.0, 2.0, "E", qrs=0.14), _beat(t + 5.0, 1.0, "N")]
    rows = _rows(build_content(beats, summarize(beats), None), "Ventricular ectopy")
    assert rows["Escape couplets"] == SummaryRow("Escape couplets", "1")
    assert rows["Escape runs (3+)"] == SummaryRow("Escape runs (3+)", "0")


def test_escape_run_is_one_strip_with_every_beat_marked_and_its_rate():
    beats = _steady(8, 0.8)
    t = beats[-1].time
    beats += [_beat(t + 2.0, 2.0, "E", qrs=0.14), _beat(t + 4.0, 2.0, "E", qrs=0.14), _beat(t + 6.0, 2.0, "E", qrs=0.14), _beat(t + 7.0, 1.0, "N")]
    section = next(s for s in build_content(beats, summarize(beats), None).sections if s.heading == ESCAPE_TITLE)
    assert len(section.items) == 1
    item = section.items[0]
    assert [b.label for b in item.run] == ["E", "E", "E"]
    assert item.caption.title == "Escape run 1 · t=8s"
    assert item.caption.what == "3 escape beats in a row at 30 bpm: an idioventricular rhythm."
