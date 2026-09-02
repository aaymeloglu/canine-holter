"""Hand-counted windows of Teeny's 2026-08-25 DR200 recording - the first
canine ground truth in the suite. See scripts/extract_teeny_fixtures.py for
provenance and docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md
for why each window exists."""
import os
import numpy as np
import pytest
from canine_holter.detection.detect import detect_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "teeny_2026-08-25")
MATCH_TOLERANCE_SEC = 0.15


def _load(name):
    z = np.load(os.path.join(FIXTURES_DIR, f"{name}.npz"))
    return z["channels"].astype(float), float(z["sample_rate"]), z["beat_times"]


def _sensitivity_precision(detected, truth):
    detected = np.asarray(detected)
    hits = sum(1 for t in truth if len(detected) and np.min(np.abs(detected - t)) <= MATCH_TOLERANCE_SEC)
    return hits / len(truth), hits / max(1, len(detected))


@pytest.mark.parametrize("name, min_sensitivity, min_precision", [
    ("tachy", 0.90, 0.95),   # sinus tachycardia ~150 bpm: the search-back's reason to exist
    ("lying", 0.90, 0.95),   # small QRS spike + large T trough
    # Same posture with a wandering pacemaker: NeuroKit detects the T waves
    # (the T-wave rule's reason to exist) and, on alternate beats, the tall
    # P wave 130-170 ms before a QRS that is a 0.03-0.3 mV notch on this
    # lead but 3.5 mV on Ch 3. Those P-wave detections fall just outside the
    # 150 ms match, capping both scores until detection uses more than one
    # lead; and the slice starts cold, so its first T wave has no rhythm
    # history for the T-wave rule to judge it by.
    ("lying_t", 0.75, 0.65),
    ("quiet", 1.00, 1.00),   # 7 beats and a real 4.67 s pause; nine phantom candidates on flat baseline
])
def test_detect_beats_on_teeny_window_single_lead(name, min_sensitivity, min_precision):
    channels, sample_rate, truth = _load(name)
    detected = [b.time for b in detect_beats(channels[0], sample_rate)]
    sensitivity, precision = _sensitivity_precision(detected, truth)
    assert sensitivity >= min_sensitivity, f"{name}: found {len(detected)} of {len(truth)} beats; sensitivity {sensitivity:.2f}"
    assert precision >= min_precision, f"{name}: {len(detected)} detections for {len(truth)} beats; precision {precision:.2f}"


@pytest.mark.parametrize("name, min_sensitivity, min_precision", [
    ("tachy", 0.95, 1.00),
    ("lying", 1.00, 1.00),
    # The P-wave and T-wave detections of the single lead have no partner
    # on the other two leads; the QRS the notch hides on Ch 1 is found on
    # Ch 2 and Ch 3. This was the pending acceptance case for multi-lead
    # detection.
    ("lying_t", 1.00, 1.00),
    ("quiet", 1.00, 1.00),
])
def test_detect_beats_on_teeny_window_all_leads(name, min_sensitivity, min_precision):
    channels, sample_rate, truth = _load(name)
    detected = [b.time for b in detect_beats(channels, sample_rate)]
    sensitivity, precision = _sensitivity_precision(detected, truth)
    assert sensitivity >= min_sensitivity, f"{name}: found {len(detected)} of {len(truth)} beats; sensitivity {sensitivity:.2f}"
    assert precision >= min_precision, f"{name}: {len(detected)} detections for {len(truth)} beats; precision {precision:.2f}"
