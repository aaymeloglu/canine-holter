from canine_holter.types import Beat
from canine_holter.classify.rules import classify_beats


def _beat(time, rr, qrs):
    return Beat(time=time, rr_interval=rr, qrs_duration=qrs, label=None)


def test_first_beats_are_undetermined_until_baseline_established():
    beats = [_beat(0.0, None, 0.08)]
    labeled = classify_beats(beats)
    assert labeled[0].label == "U"


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
    # Prematurity uses strict "<": RR interval exactly PREMATURITY_RATIO
    # (0.85) of baseline RR (0.85 * 0.8 = 0.68) is NOT below the threshold,
    # so it must not count as premature. QRS is made comfortably wide
    # (well above 1.25 * 0.08 = 0.10) so that, if the code were ever changed
    # from "<" to "<=", this beat would flip to "V" and the test would fail
    # - isolating the RR boundary specifically.
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.68, 0.14))  # RR exactly at threshold, wide QRS
    labeled = classify_beats(beats)
    assert labeled[-1].label == "N"


def test_qrs_exactly_at_width_threshold_is_not_wide():
    # Width uses strict ">": QRS duration exactly QRS_WIDTH_RATIO (1.25) of
    # baseline QRS (1.25 * 0.08 = 0.10) is NOT above the threshold, so it
    # must not count as wide. RR is made comfortably premature (well below
    # 0.85 * 0.8 = 0.68) so that, if the code were ever changed from ">" to
    # ">=", this beat would flip to "V" and the test would fail - isolating
    # the QRS boundary specifically.
    beats = [_beat(i * 0.8, 0.8 if i > 0 else None, 0.08) for i in range(8)]
    beats.append(_beat(8 * 0.8 - 0.3, 0.5, 0.10))  # premature RR, QRS exactly at threshold
    labeled = classify_beats(beats)
    assert labeled[-1].label == "N"


def test_baseline_window_forgets_beats_older_than_window():
    # BASELINE_WINDOW=8 should make the baseline reflect only the most
    # recent 8 "N" beats, forgetting older ones. Seed a *full* window of 8
    # "old" beats at a distinctly different rate (0.6s RR / 0.06s QRS), then
    # feed 8 more steady "new" beats (0.8s RR / 0.08s QRS) - exactly enough
    # to evict every old beat from the maxlen=8 deque, leaving a baseline of
    # pure new values.
    #
    # The test beat (RR=0.6s, QRS=0.12s) is chosen so its classification
    # depends on *which* baseline is used, not just on old vs. new being
    # present in some mix:
    #   - against the correctly-forgotten baseline (median 0.8/0.08):
    #     premature threshold is 0.85*0.8=0.68, so 0.6 < 0.68 -> premature.
    #   - against a baseline that failed to evict the old beats (e.g. an
    #     unbounded window mixing 8 old + 8 new, median RR 0.7): threshold
    #     is 0.85*0.7=0.595, so 0.6 > 0.595 -> NOT premature.
    # QRS=0.12 is comfortably wide under every baseline in play (well above
    # 1.25 * even the highest possible median QRS, 0.08), so "wide" is
    # never the deciding factor - only the forget behavior is.
    old_rr, old_qrs = 0.6, 0.06
    new_rr, new_qrs = 0.8, 0.08

    beats = [_beat(i * old_rr, old_rr, old_qrs) for i in range(8)]  # fills the window with old beats
    beats += [
        _beat(8 * old_rr + (i + 1) * new_rr, new_rr, new_qrs) for i in range(8)
    ]  # 8 new N beats - evicts all 8 old ones from the 8-slot window

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
