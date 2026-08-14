# tests/test_pipeline.py
import os
import re
import struct
import tempfile

import numpy as np
from canine_holter.pipeline import run_analysis

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Ground truth for tests/fixtures/mitdb_119/119 (from wfdb .atr annotations):
# 19 known PVC ('V') beats out of ~65 total detected beats. See
# test_mitbih_validation.py, which independently requires >= 50% sensitivity
# (i.e. >= 10 of 19) on this same detect -> classify path. These bounds are
# intentionally looser than the exact 18/65 currently observed, so pipeline
# tests don't fail on classifier retuning, while still catching a pipeline
# that runs "successfully" but produces empty/garbage output (e.g. 0 beats,
# 0 PVCs, or a report with no numbers at all).
MIN_EXPECTED_TOTAL_BEATS = 40
MAX_EXPECTED_TOTAL_BEATS = 90
MIN_EXPECTED_PVC_COUNT = 10  # matches the >= 50% sensitivity floor elsewhere
MAX_EXPECTED_PVC_COUNT = 19  # can't exceed ground truth PVC count


def test_run_analysis_produces_report_from_fixture():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir)
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0


def test_run_analysis_report_contains_plausible_stats_for_known_fixture():
    """Guards against a regression where the pipeline runs end-to-end without
    error but silently produces garbage or empty stats (e.g. an ingest bug
    that yields zero beats, or a wiring bug that passes the wrong beats into
    summarize/write_report)."""
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        report_path = run_analysis(input_path, out_dir)
        with open(report_path) as f:
            content = f.read()

    total_beats_match = re.search(r"Total beats:\s*(\d+)", content)
    pvc_count_match = re.search(r"PVC count:\s*(\d+)", content)
    assert total_beats_match, f"report missing 'Total beats' line:\n{content}"
    assert pvc_count_match, f"report missing 'PVC count' line:\n{content}"

    total_beats = int(total_beats_match.group(1))
    pvc_count = int(pvc_count_match.group(1))

    assert MIN_EXPECTED_TOTAL_BEATS <= total_beats <= MAX_EXPECTED_TOTAL_BEATS, (
        f"total beats {total_beats} outside plausible range for this fixture"
    )
    assert MIN_EXPECTED_PVC_COUNT <= pvc_count <= MAX_EXPECTED_PVC_COUNT, (
        f"PVC count {pvc_count} outside plausible range for this fixture"
    )


# --- End-to-end integration through the native DR200 flash.dat path ---
#
# The fixture-based tests above cover the WFDB ingest path. This builds a
# small synthetic native flash.dat (a periodic spike train) and runs the whole
# ingest(dr200) -> detect -> classify -> summarize -> report chain on it, so a
# regression that breaks the DR200 wiring (not just the decoder in isolation)
# is caught here rather than only in production.

_FLASH_CHECKSUM_TOTAL = 0x4CB31
_FLASH_SPIKE_PERIOD = 90  # 180 Hz / 90 samples = 2 Hz = 120 bpm


def _finish_flash_block(block: bytearray) -> bytes:
    struct.pack_into("<I", block, 508, _FLASH_CHECKSUM_TOTAL - sum(block[:508]))
    return bytes(block)


def _flash_metadata_block() -> bytes:
    block = bytearray(512)
    struct.pack_into("<I", block, 0, 512)
    block[4:6] = b" \n"
    meta = (
        b"SampleRate=180\nSampleStorageFormat=1\n"
        b"start_time=11:12:50\nstart_date=07/08/10\n"
    )
    block[6 : 6 + len(meta)] = meta
    return _finish_flash_block(block)


def _flash_data_block(encoded, source_position: int) -> bytes:
    nibbles = encoded.reshape(-1).astype(np.uint8)
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)
    block = bytearray(512)
    struct.pack_into("<I", block, 0, 512)
    block[4] = 0x1E
    struct.pack_into("<I", block, 6, source_position)
    block[10:466] = packed.tobytes()
    return _finish_flash_block(block)


def _spike_train_block_encoding():
    # channel 0: +70,+70 up then -70,-70 back to baseline each period; a sharp,
    # detectable QRS-like spike. Codes 7=+70, 9=-70 in the DR200 delta table.
    encoded = np.zeros((304, 3), dtype=np.uint8)
    phase = np.arange(304) % _FLASH_SPIKE_PERIOD
    encoded[phase == 0, 0] = 7
    encoded[phase == 1, 0] = 7
    encoded[phase == 2, 0] = 9
    encoded[phase == 3, 0] = 9
    return encoded


def _write_synthetic_flash(path, block_count: int = 15) -> None:
    encoded = _spike_train_block_encoding()
    data = _flash_metadata_block()
    position = 5900
    for _ in range(block_count):
        data += _flash_data_block(encoded, position)
        position += 304 * 4
    path.write_bytes(data + bytes(511))


def test_run_analysis_end_to_end_on_native_flash(tmp_path):
    flash_path = tmp_path / "flash.dat"
    _write_synthetic_flash(flash_path)
    out_dir = tmp_path / "out"

    report_path = run_analysis(str(flash_path), str(out_dir))

    assert os.path.exists(report_path)
    content = open(report_path).read()
    total_beats_match = re.search(r"Total beats:\s*(\d+)", content)
    assert total_beats_match, f"report missing 'Total beats' line:\n{content}"
    # A 25 s, 120 bpm spike train should yield roughly 50 beats; assert a
    # non-trivial count so a wiring bug that silently drops the signal fails.
    assert int(total_beats_match.group(1)) >= 20
