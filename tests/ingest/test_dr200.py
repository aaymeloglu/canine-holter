import struct
from datetime import datetime

import numpy as np
import pytest

from canine_holter.ingest.dr200 import (
    DR200_SAMPLE_RATE,
    NativeDR200FormatError,
    load_decoded_channel,
    load_native_flash,
)


_BLOCK_CHECKSUM_TOTAL = 0x4CB31
_DELTA_CODES = {
    0: 0,
    1: 1,
    3: 2,
    6: 3,
    12: 4,
    21: 5,
    38: 6,
    70: 7,
    -70: 9,
    -38: 10,
    -21: 11,
    -12: 12,
    -6: 13,
    -3: 14,
    -1: 15,
}


def _finish_block(block: bytearray) -> bytes:
    struct.pack_into("<I", block, 508, _BLOCK_CHECKSUM_TOTAL - sum(block[:508]))
    return bytes(block)


def _metadata_block(
    *, sample_rate: int = 180, storage_format: int = 1
) -> bytes:
    block = bytearray(512)
    struct.pack_into("<I", block, 0, 512)
    block[4:6] = b" \n"
    metadata = (
        f"SampleRate={sample_rate}\n"
        f"SampleStorageFormat={storage_format}\n"
        "start_time=11:12:50\n"
        "start_date=07/08/10\n"
    ).encode("ascii")
    block[6 : 6 + len(metadata)] = metadata
    return _finish_block(block)


def _data_block(encoded: np.ndarray, *, source_position: int = 5900) -> bytes:
    assert encoded.shape == (304, 3)
    nibbles = encoded.astype(np.uint8).reshape(-1)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)

    block = bytearray(512)
    struct.pack_into("<I", block, 0, 512)
    block[4] = 0x1E
    struct.pack_into("<I", block, 6, source_position)
    block[10:466] = packed.tobytes()
    return _finish_block(block)


def _write_native_flash(path, encoded: np.ndarray, **metadata_kwargs) -> None:
    path.write_bytes(
        _metadata_block(**metadata_kwargs)
        + _data_block(encoded)
        + bytes(511)
    )


def test_loads_little_endian_counts_and_scales_to_millivolts(tmp_path):
    path = tmp_path / "flashc0.dat"
    path.write_bytes(struct.pack("<hhhh", 0, 80, -80, 160))

    rec = load_decoded_channel(path, source="test-channel")

    assert rec.sample_rate == DR200_SAMPLE_RATE
    assert rec.start_time is None
    assert rec.source == "test-channel"
    np.testing.assert_allclose(rec.samples, [0.0, 1.0, -1.0, 2.0])


def test_interpolates_pacemaker_marker_instead_of_emitting_voltage_spike(tmp_path):
    path = tmp_path / "flashc1.dat"
    path.write_bytes(struct.pack("<hhh", 0, -32768, 160))

    rec = load_decoded_channel(path)

    np.testing.assert_allclose(rec.samples, [0.0, 1.0, 2.0])


def test_rejects_native_flash_dat_with_actionable_error(tmp_path):
    path = tmp_path / "flash.dat"
    path.write_bytes(b"not decoded channel data")

    with pytest.raises(NativeDR200FormatError, match="load_native_flash"):
        load_decoded_channel(path)


@pytest.mark.parametrize("contents", [b"", b"\x00"])
def test_rejects_empty_or_incomplete_channel(tmp_path, contents):
    path = tmp_path / "flashc2.dat"
    path.write_bytes(contents)

    with pytest.raises(ValueError):
        load_decoded_channel(path)


def test_rejects_non_positive_sample_rate(tmp_path):
    path = tmp_path / "channel.raw"
    path.write_bytes(struct.pack("<h", 0))

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        load_decoded_channel(path, sample_rate=0)


def test_loads_native_flash_delta_encoding_and_metadata(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[:4, 1] = [
        _DELTA_CODES[3],
        _DELTA_CODES[-3],
        _DELTA_CODES[21],
        _DELTA_CODES[-21],
    ]
    _write_native_flash(path, encoded)

    rec = load_native_flash(path, channel=1)

    assert rec.sample_rate == 180.0
    assert rec.start_time == datetime(2010, 7, 8, 11, 12, 50)
    assert rec.source == str(path)
    assert len(rec.samples) == 304
    np.testing.assert_allclose(rec.samples[:5], [0.0375, 0.0, 0.2625, 0.0, 0.0])


def test_native_flash_decodes_full_delta_table_against_unpackdc_ground_truth(tmp_path):
    # Golden vector: this exercises every difference-table entry (nibble codes
    # 1-7 for the positive deltas, 15..9 for the negative ones), written as
    # literal nibble codes rather than via the _DELTA_CODES helper, so it is
    # not circular with either the helper or the production table. The expected
    # per-sample counts were confirmed byte-for-byte against NorthEast's own
    # unpackdc decoder (mode 0) on the same nibble stream - in particular the
    # large +38/+70 jumps that make a real QRS decodable and that a naive
    # signed-4-bit model gets wrong. A regression in any table entry fails here.
    codes = [1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9]
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[: len(codes), 0] = codes
    _write_native_flash(tmp_path / "flash.dat", encoded)

    rec = load_native_flash(tmp_path / "flash.dat", channel=0)

    expected_counts = [1, 4, 10, 22, 43, 81, 151, 150, 147, 141, 129, 108, 70, 0]
    expected_mv = np.array(expected_counts, dtype=float) * 0.0125
    np.testing.assert_allclose(rec.samples[: len(codes)], expected_mv)


def test_native_flash_interpolates_simultaneous_pacemaker_marker(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[0, 0] = _DELTA_CODES[1]
    encoded[1, :] = 8
    encoded[2, 0] = _DELTA_CODES[1]
    _write_native_flash(path, encoded)

    rec = load_native_flash(path)

    np.testing.assert_allclose(rec.samples[:3], [0.0125, 0.01875, 0.025])


def test_native_flash_rejects_bad_block_checksum(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    _write_native_flash(path, encoded)
    contents = bytearray(path.read_bytes())
    contents[20] ^= 1
    path.write_bytes(contents)

    with pytest.raises(NativeDR200FormatError, match="checksum at block 0"):
        load_native_flash(path)


@pytest.mark.parametrize(
    ("metadata_kwargs", "message"),
    [
        ({"storage_format": 2}, "SampleStorageFormat=2"),
        ({"sample_rate": 360}, "sample rate 360 Hz"),
    ],
)
def test_native_flash_rejects_unsupported_recording_mode(
    tmp_path, metadata_kwargs, message
):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    _write_native_flash(path, encoded, **metadata_kwargs)

    with pytest.raises(NativeDR200FormatError, match=message):
        load_native_flash(path)


def test_native_flash_rejects_non_contiguous_data_blocks(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    path.write_bytes(
        _metadata_block()
        + _data_block(encoded, source_position=5900)
        + _data_block(encoded, source_position=9999)
    )

    with pytest.raises(NativeDR200FormatError, match="Non-contiguous"):
        load_native_flash(path)
