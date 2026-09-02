from canine_holter.types import Beat
from canine_holter.classify.rules import classify_beats


def _beat(time, rr, qrs):
    return Beat(time=time, rr_interval=rr, qrs_duration=qrs, label=None)


def test_beat_with_missing_rr_is_undetermined():
    beats = [_beat(0.0, None, 0.08)]
    labeled = classify_beats(beats)
    assert labeled[0].label == "U"


def test_first_complete_beat_seeds_the_baseline_as_normal():
    labeled = classify_beats([_beat(0.8, 0.8, 0.08)])
    assert labeled[0].label == "N"


def test_normal_regular_beats_are_labeled_normal():
    # 8 beats at a steady 0.8s RR / 0.08s QRS establish baseline, 9th matches
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(9)]
    labeled = classify_beats(beats)
    assert all(b.label in ("N", "U") for b in labeled)
    assert labeled[-1].label == "N"


def test_premature_and_wide_beat_is_labeled_pvc():
    # 8 steady normal beats to establish baseline (0.8s RR, 0.08s QRS)
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    # 9th beat: premature (RR well below 0.85 * 0.8 = 0.68) and wide
    # (QRS well above 1.25 * 0.08 = 0.10)
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, 0.14))
    labeled = classify_beats(beats)
    assert labeled[-1].label == "V"


def test_premature_but_normal_width_beat_is_not_pvc():
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, 0.08))  # premature, normal-width
    labeled = classify_beats(beats)
    assert labeled[-1].label == "N"


def test_missing_qrs_duration_is_undetermined_not_pvc():
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, None))  # premature, no QRS reading
    labeled = classify_beats(beats)
    assert labeled[-1].label == "U"


def test_rr_exactly_at_prematurity_threshold_is_not_premature():
    # Strict "<": RR exactly 0.85 x the 0.8 s baseline, with a comfortably
    # wide QRS so only the RR boundary decides.
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.68, 0.14))  # RR exactly at threshold, wide QRS
    labeled = classify_beats(beats)
    assert labeled[-1].label == "N"


def test_qrs_exactly_at_width_threshold_is_not_wide():
    # Strict ">": QRS exactly 1.25 x the 0.08 s baseline, with a comfortably
    # premature RR so only the width boundary decides.
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, 0.10))  # premature RR, QRS exactly at threshold
    labeled = classify_beats(beats)
    assert labeled[-1].label == "N"


def test_baseline_window_forgets_beats_older_than_window():
    # Eight old beats at 0.6 s RR, then eight new ones at 0.8 s: exactly
    # enough to evict the old ones from the 8-beat window. The test beat's
    # 0.6 s RR is premature against the new baseline (0.85 x 0.8 = 0.68)
    # but not against a mixed one (median 0.7, threshold 0.595).
    old_rr, old_qrs = 0.6, 0.06
    new_rr, new_qrs = 0.8, 0.08
    beats = [_beat(i * old_rr, old_rr, old_qrs) for i in range(8)]
    beats += [_beat(8 * old_rr + (i + 1) * new_rr, new_rr, new_qrs) for i in range(8)]
    beats.append(_beat(beats[-1].time + new_rr, 0.6, 0.12))

    labeled = classify_beats(beats)
    assert labeled[-1].label == "V"


def test_empty_beats_list_returns_empty_list():
    assert classify_beats([]) == []


def test_a_beat_only_a_few_samples_wider_than_baseline_is_not_a_pvc():
    # 1.28x but 17 ms wider: three samples at 180 Hz, jitter, not a wide QRS.
    beats = [_beat(i * 0.8, 0.8 if i else None, 0.061) for i in range(10)]
    beats[6] = _beat(beats[6].time - 0.3, 0.5, 0.078)
    assert classify_beats(beats)[6].label == "N"


def test_a_beat_both_proportionally_and_absolutely_wider_is_a_pvc():
    beats = [_beat(i * 0.8, 0.8 if i else None, 0.061) for i in range(10)]
    beats[6] = _beat(beats[6].time - 0.3, 0.5, 0.100)
    assert classify_beats(beats)[6].label == "V"


# --- ventricular escape beats -------------------------------------------------
from canine_holter.classify.rules import ESCAPE_RR_RATIO  # noqa: E402


def _steady_baseline(n=8, rr=0.8, qrs=0.08):
    return [_beat(i * rr, rr if i > 0 else None, qrs) for i in range(n)]


def test_wide_beat_after_a_long_gap_is_a_ventricular_escape_beat():
    beats = _steady_baseline() + [_beat(7 * 0.8 + 1.3, 1.3, 0.14)]  # over 1.5x the 0.8 s baseline, wide
    assert classify_beats(beats)[-1].label == "E"
    assert ESCAPE_RR_RATIO == 1.5


def test_wide_beat_just_under_the_escape_ratio_is_normal():
    beats = _steady_baseline() + [_beat(7 * 0.8 + 1.19, 1.19, 0.14)]
    assert classify_beats(beats)[-1].label == "N"


def test_narrow_beat_after_a_long_gap_is_normal():
    beats = _steady_baseline() + [_beat(7 * 0.8 + 2.0, 2.0, 0.08)]
    assert classify_beats(beats)[-1].label == "N"


def test_wide_premature_beat_is_still_a_pvc_not_an_escape():
    beats = _steady_baseline() + [_beat(7 * 0.8 + 0.5, 0.5, 0.14)]
    assert classify_beats(beats)[-1].label == "V"


def test_escape_beat_does_not_feed_the_baseline():
    # After an escape beat the baseline is still 0.8 s / 0.08 s, so the next
    # wide beat at 1.3 s is another escape beat, not a normal beat against a
    # baseline the escape beat would have stretched.
    beats = _steady_baseline() + [_beat(7 * 0.8 + 1.3, 1.3, 0.14), _beat(7 * 0.8 + 2.6, 1.3, 0.14)]
    assert [b.label for b in classify_beats(beats)[-2:]] == ["E", "E"]
