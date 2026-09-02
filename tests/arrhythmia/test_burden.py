from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize, pvc_runs


def _beat(time, rr, label):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def test_counts_total_and_pvc_beats():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.total_beats == 4
    assert summary.pvc_count == 1
    assert summary.pvc_burden_pct == 25.0


def test_detects_couplet_and_triplet():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "V"),
        _beat(1.6, 0.8, "V"),
        _beat(2.4, 0.8, "N"),
        _beat(3.2, 0.8, "V"),
        _beat(4.0, 0.8, "V"),
        _beat(4.8, 0.8, "V"),
        _beat(5.6, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.couplets == 1
    assert summary.triplets == 1
    assert summary.vtach_runs == 0


def test_detects_vtach_run_of_four_or_more():
    beats = [_beat(0.0, None, "N")] + [
        _beat(i * 0.8, 0.8, "V") for i in range(1, 5)
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.vtach_runs == 1
    assert summary.triplets == 0  # a run of 4 is not double-counted as a triplet


def test_flags_pause_above_threshold():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(3.5, 2.7, "N"),  # 2.7s gap - a real pause for a dog
        _beat(4.3, 0.8, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert len(summary.pauses) == 1
    assert summary.pauses[0] == 3.5


def test_brady_run_of_two_is_not_flagged():
    # Two consecutive slow beats is below SUSTAINED_EVENT_MIN_BEATS (3) and
    # must not be reported as a sustained bradycardia event.
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(2.3, 1.5, "N"),  # hr = 40 bpm, below medium-dog brady threshold (50)
        _beat(3.8, 1.5, "N"),  # hr = 40 bpm - second consecutive slow beat
        _beat(4.6, 0.8, "N"),  # back to normal rate
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.bradycardia_events == []


def test_brady_run_of_exactly_three_is_flagged():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(2.3, 1.5, "N"),  # hr = 40 bpm
        _beat(3.8, 1.5, "N"),  # hr = 40 bpm
        _beat(5.3, 1.5, "N"),  # hr = 40 bpm - third consecutive slow beat
        _beat(6.1, 0.8, "N"),  # back to normal
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert len(summary.bradycardia_events) == 1
    start, end = summary.bradycardia_events[0]
    assert start == 2.3 - 1.5
    assert end == 5.3


def test_tachycardia_run_of_three_is_flagged():
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(1.1, 0.3, "N"),  # hr = 200 bpm, above medium-dog tachy threshold (160)
        _beat(1.4, 0.3, "N"),  # hr = 200 bpm
        _beat(1.7, 0.3, "N"),  # hr = 200 bpm - third consecutive fast beat
        _beat(2.5, 0.8, "N"),  # back to normal
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert len(summary.tachycardia_events) == 1
    start, end = summary.tachycardia_events[0]
    assert start == 1.1 - 0.3
    assert end == 1.7


def test_hr_exactly_at_tachycardia_threshold_is_not_flagged():
    # medium-dog tachy threshold is 160 bpm; hr == threshold must not count
    # as "above" it (strict inequality), even sustained for 4 beats.
    rr = 60.0 / 160
    beats = [
        _beat(0.0, None, "N"),
        _beat(rr, rr, "N"),
        _beat(2 * rr, rr, "N"),
        _beat(3 * rr, rr, "N"),
        _beat(4 * rr, rr, "N"),
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.tachycardia_events == []


def test_summarize_empty_beats_returns_zeroed_summary():
    summary = summarize([], dog_weight_class="medium")
    assert summary.total_beats == 0
    assert summary.pvc_count == 0
    assert summary.pvc_burden_pct == 0.0
    assert summary.couplets == 0
    assert summary.triplets == 0
    assert summary.vtach_runs == 0
    assert summary.bradycardia_events == []
    assert summary.tachycardia_events == []
    assert summary.pauses == []


def test_pvc_runs_all_pvc_beats_form_one_run_starting_at_index_zero():
    beats = [_beat(i * 0.5, 0.5 if i > 0 else None, "V") for i in range(4)]
    runs = pvc_runs(beats)
    assert len(runs) == 1
    assert runs[0] == beats


def test_bradycardia_run_still_open_at_end_of_recording_is_flagged():
    # The recording ends mid-run (no trailing normal-rate beat to close it
    # via the mid-loop branch) - this must still be caught by the post-loop
    # closing check, not silently dropped because the loop ran out of beats.
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(2.3, 1.5, "N"),  # hr = 40 bpm
        _beat(3.8, 1.5, "N"),  # hr = 40 bpm
        _beat(5.3, 1.5, "N"),  # hr = 40 bpm - recording ends here, still slow
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert len(summary.bradycardia_events) == 1
    start, end = summary.bradycardia_events[0]
    assert start == 2.3 - 1.5
    assert end == 5.3


def test_zero_rr_interval_does_not_crash_and_breaks_run():
    # rr_interval == 0 must not raise ZeroDivisionError in the hr calc, and
    # must reset any in-progress brady/tachy run rather than joining it.
    beats = [
        _beat(0.0, None, "N"),
        _beat(0.8, 0.8, "N"),
        _beat(2.3, 1.5, "N"),  # hr = 40 bpm, slow beat 1
        _beat(3.1, 0.0, "N"),  # rr_interval = 0 - must not crash, resets run
        _beat(4.6, 1.5, "N"),  # hr = 40 bpm, slow beat (only 2 in a row now)
        _beat(6.1, 1.5, "N"),  # hr = 40 bpm, slow beat
    ]
    summary = summarize(beats, dog_weight_class="medium")
    assert summary.bradycardia_events == []


def test_longest_pause_is_the_longest_rr_interval():
    beats = [
        Beat(time=0.0, rr_interval=None, qrs_duration=0.08, label="N"),
        Beat(time=0.8, rr_interval=0.8, qrs_duration=0.08, label="N"),
        Beat(time=2.7, rr_interval=1.9, qrs_duration=0.08, label="N"),
        Beat(time=3.5, rr_interval=0.8, qrs_duration=0.08, label="N"),
    ]
    assert summarize(beats).longest_pause_sec == 1.9


def test_longest_pause_is_none_when_no_beat_has_an_rr():
    beats = [Beat(time=0.0, rr_interval=None, qrs_duration=0.08, label="N")]
    assert summarize(beats).longest_pause_sec is None
    assert summarize([]).longest_pause_sec is None


def _steady(n, rr, label="N"):
    return [Beat(time=i * rr, rr_interval=rr if i else None, qrs_duration=0.08, label=label) for i in range(n)]


def test_heart_rate_min_mean_max_with_times():
    # 10 beats at 0.5 s (120 bpm), then 10 at 1.0 s (60 bpm), then 10 at 0.25 s (240 bpm).
    beats = _steady(10, 0.5)
    t = beats[-1].time
    for i in range(10):
        t += 1.0
        beats.append(Beat(time=t, rr_interval=1.0, qrs_duration=0.08, label="N"))
    for i in range(10):
        t += 0.25
        beats.append(Beat(time=t, rr_interval=0.25, qrs_duration=0.08, label="N"))
    hr = summarize(beats).heart_rate
    assert hr.min_bpm == 60.0
    assert hr.max_bpm == 240.0
    # Mean is over the whole recording: 29 RR intervals spanning 4.5 + 10 + 2.5 s.
    assert abs(hr.mean_bpm - 60.0 * 29 / 17.0) < 1e-9
    # Extremes sit inside their plateau, not at its boundary beats.
    assert 4.5 + 1.0 <= hr.min_time <= 14.5
    assert 14.5 + 0.25 <= hr.max_time <= 17.0


def test_heart_rate_extremes_are_five_beat_medians_so_one_phantom_beat_cannot_set_them():
    beats = _steady(30, 0.5)  # 120 bpm throughout
    # One phantom beat splits an RR into 0.1 + 0.4 s (600 bpm instantaneous).
    phantom = Beat(time=10.1, rr_interval=0.1, qrs_duration=0.08, label="U")
    following = Beat(time=10.5, rr_interval=0.4, qrs_duration=0.08, label="N")
    beats = [b for b in beats if b.time != 10.5] + [phantom, following]
    beats.sort(key=lambda b: b.time)
    hr = summarize(beats).heart_rate
    assert hr.max_bpm == 120.0
    assert hr.min_bpm == 120.0


def test_heart_rate_is_none_when_fewer_than_five_rr_intervals():
    assert summarize(_steady(5, 0.5)).heart_rate is None  # 4 RR intervals
    assert summarize([]).heart_rate is None
    assert summarize(_steady(6, 0.5)).heart_rate is not None  # 5 RR intervals


def _with_run(beats, start_index, n, rr):
    """Replace beats from start_index with an n-beat PVC run at rr within the run."""
    out = list(beats)
    t = beats[start_index].time
    for k in range(n):
        out[start_index + k] = Beat(time=t + k * rr, rr_interval=beats[start_index].rr_interval if k == 0 else rr,
                                    qrs_duration=0.12, label="V")
    return out


def test_longest_and_fastest_runs():
    beats = _steady(100, 0.5)
    beats = _with_run(beats, 10, 3, 0.4)   # triplet at 150 bpm
    beats = _with_run(beats, 30, 5, 0.3)   # 5-beat run at 200 bpm
    beats = _with_run(beats, 60, 4, 0.25)  # 4-beat run at 240 bpm
    s = summarize(beats)
    assert s.longest_run.beats == 5
    assert abs(s.longest_run.bpm - 200.0) < 1e-9
    assert s.longest_run.start_time == beats[30].time
    assert s.fastest_run.beats == 4
    assert abs(s.fastest_run.bpm - 240.0) < 1e-9
    assert s.fastest_run.start_time == beats[60].time


def test_run_rate_uses_only_the_rr_intervals_inside_the_run():
    # The first beat's RR is the coupling interval to the preceding normal beat
    # and says nothing about the run's own rate.
    beats = _with_run(_steady(20, 0.5), 5, 3, 0.2)
    assert abs(summarize(beats).fastest_run.bpm - 300.0) < 1e-9


def test_couplets_and_singles_do_not_count_as_runs():
    beats = _with_run(_steady(20, 0.5), 5, 2, 0.2)
    s = summarize(beats)
    assert s.longest_run is None and s.fastest_run is None
    assert summarize([]).longest_run is None


def _hours_of_beats(hours, rr):
    return _steady(int(hours * 3600 / rr) + 1, rr)


def test_hourly_rows_bin_from_recording_start_and_keep_the_partial_last_hour():
    beats = _hours_of_beats(2.5, 0.5)  # beats at 0, 0.5, ..., 9000.0
    rows = summarize(beats).hourly
    assert [r.start_sec for r in rows] == [0.0, 3600.0, 7200.0]
    assert [r.end_sec for r in rows] == [3600.0, 7200.0, 9000.0]
    # A beat on the boundary belongs to the hour it starts.
    assert [r.beats for r in rows] == [7200, 7200, 3601]


def test_hourly_rows_count_pvcs_couplets_runs_and_pauses_in_their_hour():
    beats = _hours_of_beats(2.0, 0.5)
    beats = _with_run(beats, 100, 1, 0.3)          # isolated PVC, hour 0
    beats = _with_run(beats, 200, 2, 0.3)          # couplet, hour 0
    beats = _with_run(beats, 7300, 4, 0.3)         # run, hour 1
    i = 8000
    beats[i] = Beat(time=beats[i].time, rr_interval=3.0, qrs_duration=0.08, label="N")  # pause, hour 1
    rows = summarize(beats).hourly
    assert (rows[0].pvcs, rows[0].couplets, rows[0].runs, rows[0].pauses) == (3, 1, 0, 0)
    assert (rows[1].pvcs, rows[1].couplets, rows[1].runs, rows[1].pauses) == (4, 0, 1, 1)


def test_hourly_rows_have_min_mean_max_heart_rate_per_hour():
    fast = _steady(7200, 0.5)                  # hour 0 at 120 bpm, last beat at 3599.5
    t = fast[-1].time
    slow = [Beat(time=t + (k + 1) * 1.0, rr_interval=1.0, qrs_duration=0.08, label="N") for k in range(3600)]
    rows = summarize(fast + slow).hourly
    assert (rows[0].min_bpm, rows[0].max_bpm) == (120.0, 120.0)
    assert abs(rows[0].mean_bpm - 120.0) < 1e-9
    assert (rows[1].min_bpm, rows[1].max_bpm) == (60.0, 60.0)
    assert abs(rows[1].mean_bpm - 60.0) < 1e-9


def test_hourly_row_heart_rate_is_none_when_the_hour_has_too_few_beats():
    beats = _steady(7200, 0.5)  # hour 0 only; last beat at 3599.5
    beats.append(Beat(time=3700.0, rr_interval=100.5, qrs_duration=0.08, label="N"))
    rows = summarize(beats).hourly
    assert rows[1].beats == 1
    assert rows[1].min_bpm is None and rows[1].mean_bpm is None and rows[1].max_bpm is None


def test_hourly_rows_empty_for_no_beats():
    assert summarize([]).hourly == []


# --- duration, analyzed time, excluded spans -------------------------------
from canine_holter.quality.gate import SignalQuality  # noqa: E402


def test_summary_without_quality_uses_the_last_beat_as_duration_and_excludes_nothing():
    s = summarize(_hours_of_beats(2.5, 0.5))
    assert (s.duration_sec, s.analyzed_sec, s.excluded) == (9000.0, 9000.0, ())


def test_summary_with_quality_carries_duration_analyzed_and_excluded():
    q = SignalQuality(10000.0, ((0.0, 60.0), (9500.0, 10000.0)))
    s = summarize(_hours_of_beats(2.5, 0.5), quality=q)
    assert (s.duration_sec, s.analyzed_sec, s.excluded) == (10000.0, 9440.0, q.excluded)


def test_hourly_rows_run_to_the_recording_end_not_the_last_beat():
    q = SignalQuality(10000.0, ((9000.0, 10000.0),))
    rows = summarize(_hours_of_beats(2.5, 0.5), quality=q).hourly  # beats end at 9000
    assert [(r.start_sec, r.end_sec) for r in rows] == [(0.0, 3600.0), (3600.0, 7200.0), (7200.0, 10000.0)]
    assert [r.analyzed_sec for r in rows] == [3600.0, 3600.0, 1800.0]


def test_hourly_rows_analyzed_equals_the_hour_without_quality():
    rows = summarize(_hours_of_beats(2.5, 0.5)).hourly
    assert [r.analyzed_sec for r in rows] == [3600.0, 3600.0, 1800.0]


def test_hourly_rows_keep_a_beat_exactly_on_the_last_boundary():
    """Duration on the hour with a beat right at it: the beat's hour is listed."""
    beats = _hours_of_beats(2.0, 0.5)  # last beat at 7200.0
    rows = summarize(beats, quality=SignalQuality(7200.0, ())).hourly
    assert [(r.start_sec, r.end_sec, r.beats) for r in rows] == [
        (0.0, 3600.0, 7200), (3600.0, 7200.0, 7200), (7200.0, 7200.0, 1)
    ]


def test_summary_empty_beats_with_quality_still_reports_duration():
    s = summarize([], quality=SignalQuality(120.0, ((0.0, 120.0),)))
    assert (s.duration_sec, s.analyzed_sec, s.total_beats) == (120.0, 0.0, 0)
    assert [(r.start_sec, r.end_sec, r.beats, r.analyzed_sec) for r in s.hourly] == [(0.0, 120.0, 0, 0.0)]


# --- heart-rate variability ---------------------------------------------------
import pytest  # noqa: E402
from canine_holter.arrhythmia.burden import heart_rate_variability  # noqa: E402


def _chain(rrs, labels=None):
    """Beats at cumulative times from a list of RRs (the first beat has none)."""
    labels = labels or ["N"] * len(rrs)
    t, beats = 0.0, []
    for rr, label in zip(rrs, labels):
        t += rr or 0.0
        beats.append(_beat(t, rr, label))
    return beats


def test_hrv_sdnn_rmssd_pnn50_from_literal_nn_intervals():
    hrv = heart_rate_variability(_chain([None, 0.8, 0.9, 0.8, 1.0, 1.04]))
    # NN = 800, 900, 800, 1000, 1040 ms: mean 908, population SD sqrt(9856)
    assert hrv.nn_intervals == 5
    assert hrv.sdnn_ms == pytest.approx(99.277, abs=0.01)
    # successive differences 100, -100, 200, 40: RMS sqrt(15400); three of four over 50 ms
    assert hrv.rmssd_ms == pytest.approx(124.097, abs=0.01)
    assert hrv.pnn50_pct == pytest.approx(75.0)


def test_hrv_skips_a_pvc_its_follower_and_the_first_beat():
    beats = _chain([None, 0.8, 0.5, 1.1, 0.8, 0.8, 0.8], ["N", "N", "V", "N", "N", "N", "N"])
    hrv = heart_rate_variability(beats)
    # The 0.5 (V) and the 1.1 (after a V) are not NN; the chain restarts at the 0.8s.
    assert hrv.nn_intervals == 4
    assert (hrv.sdnn_ms, hrv.rmssd_ms, hrv.pnn50_pct) == (0.0, 0.0, 0.0)


def test_hrv_is_none_with_fewer_than_two_successive_differences():
    assert heart_rate_variability(_chain([None, 0.8, 0.9])) is None
    assert heart_rate_variability([]) is None


def test_summary_carries_hrv():
    assert summarize(_chain([None, 0.8, 0.9, 0.8])).heart_rate_variability.nn_intervals == 3


# --- rate shares and long pauses ---------------------------------------------


def test_rate_shares_count_five_beat_median_windows_against_the_class_thresholds():
    beats = _chain([None] + [0.3] * 9 + [1.5] * 9)  # 200 bpm then 40 bpm
    s = summarize(beats, dog_weight_class="medium")  # 50 / 160 bpm
    # 18 RRs make 14 windows; a window's median flips from 0.3 to 1.5 once three of five are slow.
    assert (s.fast_beats, s.slow_beats, s.rated_beats) == (7, 7, 14)
    assert (s.brady_threshold_bpm, s.tachy_threshold_bpm) == (50, 160)


def test_rate_shares_are_zero_with_too_few_beats():
    s = summarize(_chain([None, 0.3, 0.3]))
    assert (s.fast_beats, s.slow_beats, s.rated_beats) == (0, 0, 0)


def test_long_pauses_count_rrs_over_five_seconds():
    s = summarize(_chain([None, 0.8, 3.0, 5.0, 5.5, 0.8]))
    assert len(s.pauses) == 3
    assert s.long_pauses == 1
