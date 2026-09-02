import os

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


def test_detects_renamed_native_flash_by_content(tmp_path):
    from tests.native_flash_factory import data_block, metadata_block

    path = tmp_path / "flash2.dat"
    path.write_bytes(metadata_block() + data_block(np.zeros((304, 3), dtype=np.uint8)) + bytes(511))
    rec = load_recording(str(path))
    assert len(rec.samples) == 304


def test_rejects_dat_file_without_native_signature(tmp_path):
    path = tmp_path / "other.dat"
    path.write_bytes(bytes(1024))
    with pytest.raises(ValueError, match="Unsupported recording input"):
        load_recording(str(path))
