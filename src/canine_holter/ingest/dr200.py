"""Load native and vendor-extracted NorthEast Monitoring DR200 recordings."""

import struct
from datetime import datetime
from pathlib import Path

import numpy as np

from canine_holter.types import Recording


DR200_CHANNEL_NAMES = ("Ch 1", "Ch 2", "Ch 3")  # the DR200 and HE/LX call them channels, not leads
DR200_SAMPLE_RATE = 180.0
DR200_MILLIVOLTS_PER_COUNT = 0.0125
DR200_PACEMAKER_MARKER = np.iinfo(np.int16).min

_BLOCK_SIZE = 512
_BLOCK_CHECKSUM_TOTAL = 0x4CB31
_DATA_BLOCK_TYPE = 0x1E
_DATA_PAYLOAD = slice(10, 466)
_SAMPLES_PER_BLOCK = 304
_SOURCE_BYTES_PER_SAMPLE = 4
_ZERO_BLOCK = bytes(_BLOCK_SIZE)

# SampleStorageFormat=1 stores one four-bit sample difference per channel.
# Values are ordered by their encoded nibble.  Nibble 8 is reserved for a
# simultaneous pacemaker marker on all three channels.
_DELTA_COUNTS = np.array(
    [0, 1, 3, 6, 12, 21, 38, 70, 0, -70, -38, -21, -12, -6, -3, -1],
    dtype=np.int16,
)


class NativeDR200FormatError(ValueError):
    """Raised when a native ``flash.dat`` is malformed or unsupported."""


def _replace_pacemaker_markers(samples: np.ndarray) -> np.ndarray:
    """Interpolate 0x8000 pacemaker markers from nearby ECG samples."""
    marker_indices = np.flatnonzero(samples == DR200_PACEMAKER_MARKER)
    if marker_indices.size == 0:
        return samples
    if marker_indices.size == samples.size:
        raise ValueError("DR200 channel contains pacemaker markers but no ECG samples")

    result = samples.astype(np.float64, copy=False)
    sample_indices = np.flatnonzero(samples != DR200_PACEMAKER_MARKER)
    result[marker_indices] = np.interp(
        marker_indices, sample_indices, result[sample_indices]
    )
    return result


def load_decoded_channel(
    path: str | Path,
    *,
    source: str | None = None,
    sample_rate: float = DR200_SAMPLE_RATE,
    start_time: datetime | None = None,
) -> Recording:
    """Load a vendor-extracted DR200 ``flashcN.dat``/``.raw`` channel.

    ``sample_rate`` is configurable for explicitly converted research files,
    but defaults to the 180 Hz rate documented for DR200 three-channel output.
    Native SD-card ``flash.dat`` files must be loaded with
    :func:`load_native_flash` rather than interpreted as raw int16 data.
    """
    channel_path = Path(path)
    if channel_path.name.casefold() == "flash.dat":
        raise NativeDR200FormatError(
            "flash.dat is a native DR200 recording, not a decoded channel; "
            "use load_native_flash()"
        )
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    try:
        size = channel_path.stat().st_size
    except FileNotFoundError:
        raise FileNotFoundError(f"DR200 channel file not found: {channel_path}") from None
    if size == 0:
        raise ValueError(f"DR200 channel file is empty: {channel_path}")
    if size % np.dtype("<i2").itemsize:
        raise ValueError(
            f"DR200 channel file has an incomplete 16-bit sample ({size} bytes): {channel_path}"
        )

    counts = np.fromfile(channel_path, dtype="<i2")
    reconstructed = _replace_pacemaker_markers(counts)
    samples_mv = reconstructed.astype(np.float64, copy=False) * DR200_MILLIVOLTS_PER_COUNT

    return Recording(
        samples=samples_mv,
        sample_rate=float(sample_rate),
        start_time=start_time,
        source=source if source is not None else str(channel_path),
    )


def _parse_metadata(block: bytes) -> dict[str, str]:
    text = block[6:508].split(b"\0", 1)[0].decode("ascii", errors="replace")
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def _read_start_time(metadata: dict[str, str]) -> datetime | None:
    date = metadata.get("start_date")
    time = metadata.get("start_time")
    if not date or not time:
        return None

    for date_format in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(f"{date} {time}", f"{date_format} %H:%M:%S")
        except ValueError:
            pass
    raise NativeDR200FormatError(
        f"Unsupported DR200 recording start date/time: {date} {time}"
    )


def _iter_active_blocks(path: Path):
    """Yield validated nonzero blocks, allowing a zero-filled preallocated tail."""
    saw_zero_padding = False
    with path.open("rb") as handle:
        block_index = 0
        while block := handle.read(_BLOCK_SIZE):
            if len(block) != _BLOCK_SIZE:
                if any(block):
                    raise NativeDR200FormatError(
                        f"DR200 flash.dat has a truncated block at index {block_index}"
                    )
                break

            if block == _ZERO_BLOCK:
                saw_zero_padding = True
                block_index += 1
                continue
            if saw_zero_padding:
                raise NativeDR200FormatError(
                    f"DR200 flash.dat has data after zero padding at block {block_index}"
                )

            block_length = struct.unpack_from("<I", block)[0]
            if block_length != _BLOCK_SIZE:
                raise NativeDR200FormatError(
                    f"Invalid DR200 block length {block_length} at block {block_index}"
                )

            checksum = struct.unpack_from("<I", block, 508)[0]
            if sum(block[:508]) + checksum != _BLOCK_CHECKSUM_TOTAL:
                raise NativeDR200FormatError(
                    f"Invalid DR200 checksum at block {block_index}"
                )

            yield block_index, block
            block_index += 1


def _inspect_native_flash(path: Path) -> tuple[int, datetime | None]:
    metadata: dict[str, str] = {}
    data_block_count = 0
    expected_position: int | None = None
    active_block_count = 0

    for _, block in _iter_active_blocks(path):
        active_block_count += 1
        if active_block_count == 1 and b"SampleStorageFormat=" in block:
            metadata = _parse_metadata(block)

        if block[4] != _DATA_BLOCK_TYPE or block[5] != 0:
            continue

        source_position = struct.unpack_from("<I", block, 6)[0]
        if expected_position is not None and source_position != expected_position:
            raise NativeDR200FormatError(
                "Non-contiguous DR200 ECG blocks: expected source position "
                f"{expected_position}, found {source_position}"
            )
        expected_position = (
            source_position + _SAMPLES_PER_BLOCK * _SOURCE_BYTES_PER_SAMPLE
        )
        data_block_count += 1

    if active_block_count == 0:
        raise NativeDR200FormatError("DR200 flash.dat contains no recording blocks")
    if not metadata:
        raise NativeDR200FormatError("DR200 flash.dat is missing recording metadata")
    if metadata.get("SampleStorageFormat") != "1":
        value = metadata.get("SampleStorageFormat", "missing")
        raise NativeDR200FormatError(
            f"Unsupported DR200 SampleStorageFormat={value}; only format 1 is supported"
        )
    if data_block_count == 0:
        raise NativeDR200FormatError("DR200 flash.dat contains no three-channel ECG blocks")

    try:
        sample_rate = float(metadata["SampleRate"])
    except (KeyError, ValueError):
        raise NativeDR200FormatError("DR200 flash.dat has an invalid SampleRate") from None
    if sample_rate != DR200_SAMPLE_RATE:
        raise NativeDR200FormatError(
            f"Unsupported DR200 sample rate {sample_rate:g} Hz; expected 180 Hz"
        )

    return data_block_count, _read_start_time(metadata)


def _decode_data_block(block: bytes) -> tuple[np.ndarray, np.ndarray]:
    packed = np.frombuffer(block[_DATA_PAYLOAD], dtype=np.uint8)
    nibbles = np.empty(packed.size * 2, dtype=np.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    encoded = nibbles.reshape(_SAMPLES_PER_BLOCK, 3)

    marker_rows = np.any(encoded == 8, axis=1)
    if np.any(encoded[marker_rows] != 8):
        raise NativeDR200FormatError(
            "DR200 pacemaker marker is not aligned across all three channels"
        )

    return encoded, marker_rows


def load_native_flash(
    path: str | Path,
    *,
    channel: int = 0,
    source: str | None = None,
) -> Recording:
    """Load an SD-card ``flash.dat`` recording: all three ECG channels, with
    ``channel`` selecting the analysis lead.

    DR200 SampleStorageFormat 1 stores 304 three-channel timepoints in every
    checksummed ECG block.  Each channel is encoded as a nonlinear four-bit
    difference from its previous sample.  Pacemaker markers are reconstructed
    by interpolation so they do not become artificial voltage spikes.
    """
    if channel not in (0, 1, 2):
        raise ValueError("channel must be 0, 1, or 2")

    flash_path = Path(path)
    try:
        size = flash_path.stat().st_size
    except FileNotFoundError:
        raise FileNotFoundError(f"DR200 flash.dat not found: {flash_path}") from None
    if size == 0:
        raise NativeDR200FormatError(f"DR200 flash.dat is empty: {flash_path}")

    data_block_count, start_time = _inspect_native_flash(flash_path)

    counts_by_channel = np.empty((3, data_block_count * _SAMPLES_PER_BLOCK), dtype=np.float64)
    cursor = 0
    previous_counts = np.zeros(3, dtype=np.int64)
    for _, block in _iter_active_blocks(flash_path):
        if block[4] != _DATA_BLOCK_TYPE or block[5] != 0:
            continue

        encoded, marker_rows = _decode_data_block(block)
        deltas = _DELTA_COUNTS[encoded].astype(np.int64)  # (timepoints, 3)
        deltas[marker_rows, :] = 0
        counts = previous_counts + np.cumsum(deltas, axis=0, dtype=np.int64)
        previous_counts = counts[-1].copy()
        counts[marker_rows, :] = DR200_PACEMAKER_MARKER
        counts_by_channel[:, cursor : cursor + _SAMPLES_PER_BLOCK] = counts.T
        cursor += _SAMPLES_PER_BLOCK

    channels_mv = np.stack(
        [_replace_pacemaker_markers(ch) * DR200_MILLIVOLTS_PER_COUNT for ch in counts_by_channel]
    )
    return Recording(
        samples=channels_mv[channel],
        sample_rate=DR200_SAMPLE_RATE,
        start_time=start_time,
        source=source if source is not None else str(flash_path),
        channels=channels_mv,
        channel_names=DR200_CHANNEL_NAMES,
    )
