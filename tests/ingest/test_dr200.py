import struct
from datetime import datetime

import numpy as np
import pytest

from canine_holter.ingest.dr200 import NativeDR200FormatError, load_native_flash
from tests.native_flash_factory import (
    data_block,
    finish_block,
    metadata_block,
    write_native_flash,
)


def test_loads_native_flash_delta_encoding_and_metadata(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[:4, 1] = [2, 14, 5, 11]  # +3, -3, +21, -21 counts
    write_native_flash(path, encoded)

    rec = load_native_flash(path)

    assert rec.sample_rate == 180.0
    assert rec.start_time == datetime(2010, 7, 8, 11, 12, 50)
    assert rec.source == str(path)
    assert len(rec.samples) == 304
    np.testing.assert_allclose(rec.channels[1][:5], [0.0375, 0.0, 0.2625, 0.0, 0.0])


def test_native_flash_decodes_full_delta_table_against_unpackdc_ground_truth(tmp_path):
    # This non-circular golden vector exercises every nonzero delta. The
    # expected counts were confirmed against NorthEast's unpackdc decoder.
    codes = [1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9]
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[: len(codes), 0] = codes
    write_native_flash(tmp_path / "flash.dat", encoded)

    rec = load_native_flash(tmp_path / "flash.dat")

    expected_counts = [1, 4, 10, 22, 43, 81, 151, 150, 147, 141, 129, 108, 70, 0]
    expected_mv = np.array(expected_counts, dtype=float) * 0.0125
    np.testing.assert_allclose(rec.samples[: len(codes)], expected_mv)


def test_native_flash_interpolates_simultaneous_pacemaker_marker(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[0, 0] = 1  # +1 count
    encoded[1, :] = 8
    encoded[2, 0] = 1
    write_native_flash(path, encoded)

    rec = load_native_flash(path)

    np.testing.assert_allclose(rec.samples[:3], [0.0125, 0.01875, 0.025])


def test_native_flash_rejects_bad_block_checksum(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    write_native_flash(path, encoded)
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
    write_native_flash(path, encoded, **metadata_kwargs)

    with pytest.raises(NativeDR200FormatError, match=message):
        load_native_flash(path)


def test_native_flash_rejects_non_contiguous_data_blocks(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    path.write_bytes(
        metadata_block()
        + data_block(encoded, source_position=5900)
        + data_block(encoded, source_position=9999)
    )

    with pytest.raises(NativeDR200FormatError, match="Non-contiguous"):
        load_native_flash(path)


def test_native_flash_rejects_empty_file(tmp_path):
    path = tmp_path / "flash.dat"
    path.write_bytes(b"")
    with pytest.raises(NativeDR200FormatError, match="is empty"):
        load_native_flash(path)


def test_native_flash_rejects_truncated_block(tmp_path):
    # A partial final block that still carries nonzero bytes is corruption,
    # not the benign zero-filled preallocated tail.
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    path.write_bytes(metadata_block() + data_block(encoded) + b"\x01\x02\x03")
    with pytest.raises(NativeDR200FormatError, match="truncated block"):
        load_native_flash(path)


def test_native_flash_rejects_data_after_zero_padding(tmp_path):
    # Zero blocks mark the unused tail of a preallocated card; real data must
    # never appear after them.
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    path.write_bytes(
        metadata_block() + bytes(512) + data_block(encoded) + bytes(511)
    )
    with pytest.raises(NativeDR200FormatError, match="data after zero padding"):
        load_native_flash(path)


def test_native_flash_rejects_bad_block_length(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    block = bytearray(metadata_block())
    struct.pack_into("<I", block, 0, 256)  # claim a wrong length
    path.write_bytes(finish_block(block) + data_block(encoded) + bytes(511))
    with pytest.raises(NativeDR200FormatError, match="block length"):
        load_native_flash(path)


def test_native_flash_rejects_missing_metadata(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    path.write_bytes(data_block(encoded) + bytes(511))
    with pytest.raises(NativeDR200FormatError, match="missing recording metadata"):
        load_native_flash(path)


def test_native_flash_requires_metadata_in_first_block(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    path.write_bytes(data_block(encoded) + metadata_block() + bytes(511))

    with pytest.raises(NativeDR200FormatError, match="missing recording metadata"):
        load_native_flash(path)


def test_native_flash_rejects_recording_with_no_ecg_blocks(tmp_path):
    path = tmp_path / "flash.dat"
    path.write_bytes(metadata_block() + bytes(511))
    with pytest.raises(NativeDR200FormatError, match="no three-channel ECG blocks"):
        load_native_flash(path)


def test_native_flash_rejects_non_numeric_sample_rate(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    metadata = metadata_block(
        text=(
            "SampleRate=fast\nSampleStorageFormat=1\n"
            "start_time=11:12:50\nstart_date=07/08/10\n"
        )
    )
    path.write_bytes(metadata + data_block(encoded) + bytes(511))
    with pytest.raises(NativeDR200FormatError, match="invalid SampleRate"):
        load_native_flash(path)


def test_native_flash_rejects_misaligned_pacemaker_marker(tmp_path):
    # Nibble 8 is documented as a simultaneous marker on all three channels;
    # a marker on only one channel means the stream is misframed, so fail
    # rather than silently mis-decode it.
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[0, 0] = 8  # channel 0 only
    write_native_flash(path, encoded)
    with pytest.raises(NativeDR200FormatError, match="not aligned across all three"):
        load_native_flash(path)


def test_native_flash_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="flash.dat not found"):
        load_native_flash(tmp_path / "nope" / "flash.dat")


def test_native_flash_rejects_all_zero_file(tmp_path):
    # A card that was formatted/preallocated but never recorded is all zero
    # blocks with no metadata or data.
    path = tmp_path / "flash.dat"
    path.write_bytes(bytes(512) * 3)
    with pytest.raises(NativeDR200FormatError, match="no recording blocks"):
        load_native_flash(path)


def test_native_flash_rejects_unparseable_start_datetime(tmp_path):
    path = tmp_path / "flash.dat"
    encoded = np.zeros((304, 3), dtype=np.uint8)
    metadata = metadata_block(
        text=(
            "SampleRate=180\nSampleStorageFormat=1\n"
            "start_time=99:99:99\nstart_date=77/77/77\n"
        )
    )
    path.write_bytes(metadata + data_block(encoded) + bytes(511))
    with pytest.raises(NativeDR200FormatError, match="start date/time"):
        load_native_flash(path)


def test_native_flash_decodes_all_three_channels(tmp_path):
    """Channels 1 and 2 are checked against the documented delta table
    (1 -> +1, 2 -> +3, 3 -> +6, 15 -> -1, 14 -> -3 counts; 12.5 uV/count),
    independently of channel 0."""
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[:4, 0] = [7, 7, 9, 9]
    encoded[:3, 1] = [1, 2, 3]
    encoded[:2, 2] = [15, 14]
    path = tmp_path / "flash.dat"
    write_native_flash(path, encoded)
    rec = load_native_flash(path)
    assert rec.channel_names == ("Ch 1", "Ch 2", "Ch 3")
    assert rec.channels.shape == (3, 304)
    np.testing.assert_allclose(rec.channels[0][:4], [0.875, 1.75, 0.875, 0.0])
    np.testing.assert_allclose(rec.channels[1][:3], [0.0125, 0.05, 0.125])
    np.testing.assert_allclose(rec.channels[2][:2], [-0.0125, -0.05])
    np.testing.assert_allclose(rec.samples, rec.channels[0])


def _ramp_block(step_code: int, **kwargs) -> bytes:
    encoded = np.zeros((304, 3), dtype=np.uint8)
    encoded[:, 0] = step_code
    return data_block(encoded, **kwargs)


def test_native_flash_stops_at_stale_blocks_from_an_earlier_recording(tmp_path):
    # A reused card keeps an earlier recording's blocks after the new one.
    # They carry a different sequence number and are dropped, even when
    # their source positions happen to continue the new recording's.
    path = tmp_path / "flash.dat"
    path.write_bytes(
        metadata_block()
        + _ramp_block(1, source_position=5900, sequence=8)
        + _ramp_block(1, source_position=5900 + 1216, sequence=8)
        + _ramp_block(7, source_position=5900 + 2 * 1216, sequence=6)
        + bytes(511)
    )
    rec = load_native_flash(path)
    assert len(rec.samples) == 608
    np.testing.assert_allclose(rec.samples[-1], 608 * 0.0125)


def test_native_flash_ignores_everything_after_the_stale_boundary(tmp_path):
    # Once the boundary is found nothing after it is validated: an old
    # recording's non-contiguous block, another recorder's serial, and
    # data after zero padding are all stale card contents.
    path = tmp_path / "flash.dat"
    path.write_bytes(
        metadata_block()
        + _ramp_block(1, source_position=5900)
        + _ramp_block(1, source_position=9999, sequence=3, serial=13515)
        + bytes(512)
        + _ramp_block(1, source_position=5900, sequence=2)
        + bytes(511)
    )
    rec = load_native_flash(path)
    assert len(rec.samples) == 304


# --- diary events -------------------------------------------------------------
from canine_holter.types import DiaryEvent  # noqa: E402

_DIARY = "Manual Event^Chest Pain^Dizziness  ^Palpitation  ^Chest Pressure^Rapid Heart ^Short of Breath^Skipped Beat^"


def _flash_with_event(tmp_path, event, diary=_DIARY):
    """Three ECG blocks, the middle one carrying a diary-button press."""
    encoded = np.zeros((304, 3), dtype=np.uint8)
    text = f"SampleRate=180\nSampleStorageFormat=1\nstart_time=10:13:56\nstart_date=09/03/26\n"
    if diary is not None:
        text += f"DiaryText={diary}\n"
    blocks = [
        data_block(encoded, source_position=5900 + i * 1216, event=event if i == 1 else (0, 0))
        for i in range(3)
    ]
    path = tmp_path / "flash.dat"
    path.write_bytes(metadata_block(text=text) + b"".join(blocks) + bytes(511))
    return path


def test_native_flash_keeps_decoding_past_a_diary_event_block(tmp_path):
    # A press changes bytes 470..471 of the block being written; that is
    # not a new recording, and the blocks after it are this recording.
    rec = load_native_flash(_flash_with_event(tmp_path, event=(7, 2)))
    assert rec.samples.shape == (3 * 304,)
    assert rec.events == (DiaryEvent(time_sec=304 / 180, type_index=2, label="Chest Pain", detail=7),)


def test_diary_labels_are_trimmed_and_indexed_from_one(tmp_path):
    rec = load_native_flash(_flash_with_event(tmp_path, event=(0, 8)))
    assert rec.events[0].label == "Skipped Beat"
    rec = load_native_flash(_flash_with_event(tmp_path, event=(0, 3)))
    assert rec.events[0].label == "Dizziness"


def test_event_type_beyond_the_diary_gets_a_fallback_label(tmp_path):
    rec = load_native_flash(_flash_with_event(tmp_path, event=(0, 9)))
    assert rec.events[0].label == "Event type 9"


def test_event_without_diary_text_gets_a_fallback_label(tmp_path):
    rec = load_native_flash(_flash_with_event(tmp_path, event=(0, 2), diary=None))
    assert rec.events == (DiaryEvent(time_sec=304 / 180, type_index=2, label="Event type 2", detail=0),)


def test_no_press_means_no_events(tmp_path):
    rec = load_native_flash(_flash_with_event(tmp_path, event=(0, 0)))
    assert rec.events == ()


def test_stale_blocks_are_still_told_apart_by_sequence_and_serial(tmp_path):
    encoded = np.zeros((304, 3), dtype=np.uint8)
    blocks = [data_block(encoded, source_position=5900), data_block(encoded, source_position=5900 + 1216, sequence=3)]
    path = tmp_path / "flash.dat"
    path.write_bytes(metadata_block() + b"".join(blocks) + bytes(511))
    assert load_native_flash(path).samples.shape == (304,)
