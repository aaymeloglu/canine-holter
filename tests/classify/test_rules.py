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
