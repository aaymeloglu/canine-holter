# tests/test_mitbih_validation.py
"""Pipeline mechanics validation against MIT-BIH ground truth.

This confirms the detect -> classify pipeline runs correctly end-to-end and
finds a reasonable fraction of real PVC beats. It is NOT a claim of canine
clinical accuracy - MIT-BIH is human data. See
docs/superpowers/specs/2026-08-13-pvc-detection-design.md for why canine
validation requires real Teeny recordings instead.
"""
import os
import wfdb
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats
from canine_holter.classify.rules import classify_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_classifier_finds_most_known_pvc_beats_in_mitbih_record_119():
    fixture_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    rec = load_local_record(fixture_path, source="mitdb_119")
    ann = wfdb.rdann(fixture_path, "atr")

    # Ground-truth PVC beat times (annotation symbol 'V'), in seconds
    ground_truth_pvc_times = {
        sample / rec.sample_rate
        for sample, symbol in zip(ann.sample, ann.symbol)
        if symbol == "V"
    }
    assert len(ground_truth_pvc_times) > 0, "fixture should contain at least one PVC"

    beats = detect_beats(rec.samples, rec.sample_rate)
    labeled = classify_beats(beats)
    detected_pvc_times = [b.time for b in labeled if b.label == "V"]

    # A detected PVC "matches" a ground-truth PVC if within 50ms of it
    TOLERANCE_SEC = 0.05
    matched = sum(
        1
        for gt_time in ground_truth_pvc_times
        if any(abs(gt_time - d_time) <= TOLERANCE_SEC for d_time in detected_pvc_times)
    )
    sensitivity = matched / len(ground_truth_pvc_times)

    # Rules-based v1 on human data is a coarse mechanics check, not a tuned
    # classifier - 50% catches "the pipeline is fundamentally working"
    # without demanding human-tuned accuracy from a canine-first design.
    assert sensitivity >= 0.5, (
        f"Only matched {matched}/{len(ground_truth_pvc_times)} "
        f"({sensitivity:.0%}) known PVC beats"
    )
