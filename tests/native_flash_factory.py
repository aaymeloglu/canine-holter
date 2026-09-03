"""Small native DR200 records for parser and pipeline tests."""

import struct

import numpy as np


_CHECKSUM_TOTAL = 0x4CB31


def finish_block(block: bytearray) -> bytes:
    struct.pack_into("<I", block, 508, _CHECKSUM_TOTAL - sum(block[:508]))
    return bytes(block)


def metadata_block(
    *, sample_rate: int = 180, storage_format: int = 1, text: str | None = None
) -> bytes:
    if text is None:
        text = (
            f"SampleRate={sample_rate}\n"
            f"SampleStorageFormat={storage_format}\n"
            "start_time=11:12:50\n"
            "start_date=07/08/10\n"
        )

    block = bytearray(512)
    struct.pack_into("<I", block, 0, 512)
    block[4:6] = b" \n"
    encoded = text.encode("ascii")
    block[6 : 6 + len(encoded)] = encoded
    return finish_block(block)


def data_block(
    encoded: np.ndarray,
    *,
    source_position: int = 5900,
    sequence: int = 4,
    serial: int = 52837,
    event: tuple[int, int] = (0, 0),
) -> bytes:
    """One ECG block. sequence/serial are the recording sequence number and
    recorder serial number that every block of one recording repeats.
    event is (detail, type) for bytes 470..471: a diary-button press stored
    in the block being written, (0, 0) otherwise."""
    assert encoded.shape == (304, 3)
    nibbles = encoded.astype(np.uint8).reshape(-1)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)

    block = bytearray(512)
    struct.pack_into("<I", block, 0, 512)
    block[4] = 0x1E
    struct.pack_into("<I", block, 6, source_position)
    block[10:466] = packed.tobytes()
    struct.pack_into("<HHBB", block, 466, sequence, serial, *event)
    return finish_block(block)


def write_native_flash(path, encoded: np.ndarray, **metadata_kwargs) -> None:
    path.write_bytes(
        metadata_block(**metadata_kwargs) + data_block(encoded) + bytes(511)
    )
