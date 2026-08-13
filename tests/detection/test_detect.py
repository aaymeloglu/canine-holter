import os
import numpy as np
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_detects_beats_in_mitbih_fixture():
    rec = load_local_record(
        os.path.join(FIXTURES_DIR, "mitdb_119", "119"), source="mitdb_119"
    )
    beats = detect_beats(rec.samples, rec.sample_rate)
    # 60s at a resting ~75bpm should be roughly 60-90 beats; a wide sanity
    # range avoids a brittle exact-count assertion
    assert 50 <= len(beats) <= 100
    assert beats[0].rr_interval is None  # first beat has no prior beat
    assert all(b.rr_interval is not None for b in beats[1:])
    assert all(b.label is None for b in beats)  # not classified yet
    # The energy-envelope QRS width measurement (replacing NeuroKit2's wave
    # delineation, which returned NaN for 19/19 ground-truth PVC beats in
    # this same fixture) should succeed for nearly every beat, not just
    # "at least one" - confirmed 65/65 (100%) on this fixture; 90% leaves
    # slack for incidental edge-of-recording beats without masking a
    # regression back toward the old near-total-failure behavior.
    non_none = sum(1 for b in beats if b.qrs_duration is not None)
    assert non_none / len(beats) >= 0.9, (
        f"only {non_none}/{len(beats)} beats had a non-None qrs_duration"
    )


def test_detects_beats_in_canine_fixture():
    rec = load_local_record(
        os.path.join(FIXTURES_DIR, "physiozoo_dog1", "Dog_01"), source="physiozoo_dog1"
    )
    beats = detect_beats(rec.samples, rec.sample_rate)
    assert len(beats) > 0
    assert beats[0].rr_interval is None  # first beat has no prior beat
    assert all(b.rr_interval is not None for b in beats[1:])
    assert all(b.label is None for b in beats)  # not classified yet


def test_returns_empty_list_for_flat_signal():
    # A flat/constant signal has no meaningful gradient for NeuroKit2's
    # R-peak detector to key off of, so it reliably yields zero R-peaks -
    # confirmed via a scratch run (0 peaks for 1000 zero samples at 360Hz),
    # exercising the len(r_peaks) < 2 early-return branch.
    flat_signal = np.zeros(1000)
    beats = detect_beats(flat_signal, sample_rate=360.0)
    assert beats == []


def _gaussian_pulse(t, center, sigma, amplitude=2.0):
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def test_wide_synthetic_beat_measures_wider_than_narrow_synthetic_beat():
    # Build a deterministic synthetic signal: a narrow Gaussian "QRS" pulse
    # (sigma=8ms, roughly a normal QRS) followed by a wide one (sigma=40ms,
    # roughly a PVC-like wide complex), at known sample positions. This
    # exercises the envelope-based _qrs_width measurement directly through
    # the public detect_beats() entry point without depending on any real
    # ECG fixture.
    sample_rate = 500.0
    duration_sec = 4.0
    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    signal = np.zeros(n)
    signal += _gaussian_pulse(t, center=1.0, sigma=0.008)  # narrow beat
    signal += _gaussian_pulse(t, center=2.0, sigma=0.040)  # wide beat

    beats = detect_beats(signal, sample_rate)

    assert len(beats) == 2
    narrow_beat, wide_beat = beats
    assert narrow_beat.qrs_duration is not None
    assert wide_beat.qrs_duration is not None
    assert wide_beat.qrs_duration > narrow_beat.qrs_duration
