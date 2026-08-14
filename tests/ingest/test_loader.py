import os
import struct

import numpy as np
import pytest

from canine_holter.ingest import loader
from canine_holter.ingest.loader import load_recording


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_detects_wfdb_base_path():
    path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")

    rec = load_recording(path)

    assert rec.sample_rate == 360.0
    assert rec.source == path


def test_detects_wfdb_header_path():
    header_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119.hea")

    rec = load_recording(header_path)

    assert rec.sample_rate == 360.0
    assert rec.source == os.path.splitext(header_path)[0]


@pytest.mark.parametrize("filename", ["flashc0.dat", "FLASHC2.DAT", "channel.raw"])
def test_detects_decoded_dr200_channel(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(struct.pack("<hh", 0, 80))

    rec = load_recording(str(path))

    assert rec.sample_rate == 180.0
    np.testing.assert_allclose(rec.samples, [0.0, 1.0])


def test_detects_native_flash_dat(tmp_path, monkeypatch):
    path = tmp_path / "flash.dat"
    path.write_bytes(b"native data")
    sentinel = object()
    captured = []
    monkeypatch.setattr(
        loader,
        "load_native_flash",
        lambda selected_path: captured.append(selected_path) or sentinel,
    )

    assert load_recording(str(path)) is sentinel
    assert captured == [path]


def test_rejects_unknown_input(tmp_path):
    path = tmp_path / "unknown.dat"
    path.write_bytes(b"unknown")

    with pytest.raises(ValueError, match="Unsupported recording input"):
        load_recording(str(path))
