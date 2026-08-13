from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize


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
