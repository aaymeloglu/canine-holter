import os
from canine_holter.ingest.wfdb_loader import load_local_record

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_loads_mitbih_fixture_as_recording():
    path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    rec = load_local_record(path, source="mitdb_119")
    assert rec.sample_rate == 360.0
    assert len(rec.samples) > 0
    assert rec.source == "mitdb_119"


def test_loads_physiozoo_fixture_as_recording():
    path = os.path.join(FIXTURES_DIR, "physiozoo_dog1", "Dog_01")
    rec = load_local_record(path, source="physiozoo_dog1")
    assert rec.sample_rate > 0
    assert len(rec.samples) > 0


def test_wfdb_record_carries_every_signal_as_a_channel():
    import numpy as np
    rec = load_local_record(os.path.join(FIXTURES_DIR, "mitdb_119", "119"), source="mitdb_119")
    assert rec.channels.shape == (2, len(rec.samples))
    assert rec.channel_names == ("MLII", "V1")
    np.testing.assert_array_equal(rec.channels[0], rec.samples)
