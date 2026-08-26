from canine_holter.types import Recording, Beat
import numpy as np


def test_recording_holds_signal_and_metadata():
    samples = np.array([0.1, 0.2, 0.1, -0.1])
    rec = Recording(samples=samples, sample_rate=360.0, start_time=None, source="test")
    assert rec.sample_rate == 360.0
    assert len(rec.samples) == 4
    assert rec.source == "test"


def test_beat_defaults_label_to_none():
    beat = Beat(time=1.5, rr_interval=0.8, qrs_duration=0.09, label=None)
    assert beat.label is None
    assert beat.time == 1.5


def test_beat_is_immutable_and_replaceable():
    from dataclasses import replace
    beat = Beat(time=1.5, rr_interval=0.8, qrs_duration=0.09, label=None)
    labeled = replace(beat, label="V")
    assert labeled.label == "V"
    assert beat.label is None  # original unchanged


def test_recording_equality_does_not_raise_on_array_field():
    rec1 = Recording(samples=np.array([1.0, 2.0]), sample_rate=360.0, start_time=None, source="a")
    rec2 = Recording(samples=np.array([3.0, 4.0]), sample_rate=360.0, start_time=None, source="b")
    assert (rec1 == rec2) is False  # must not raise
    assert (rec1 == rec1) is True


def _rec(**kw):
    base = dict(samples=np.array([1.0, 2.0, 3.0]), sample_rate=100.0, start_time=None, source="t")
    base.update(kw)
    return Recording(**base)


def test_recording_channels_default_to_none_and_no_names():
    rec = _rec()
    assert rec.channels is None
    assert rec.channel_names == ()


def test_recording_accepts_channels_matching_samples_and_names():
    rec = _rec(channels=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), channel_names=("A", "B"))
    assert rec.channels.shape == (2, 3)


def test_recording_rejects_channels_of_a_different_length():
    import pytest
    with pytest.raises(ValueError, match="length"):
        _rec(channels=np.zeros((2, 4)), channel_names=("A", "B"))


def test_recording_rejects_non_2d_channels_and_name_count_mismatch():
    import pytest
    with pytest.raises(ValueError, match="2-D"):
        _rec(channels=np.zeros(3), channel_names=("A",))
    with pytest.raises(ValueError, match="channel_names"):
        _rec(channels=np.zeros((2, 3)), channel_names=("A",))
