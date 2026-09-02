"""Recording format detection and shared ingestion entry point."""

from pathlib import Path

from canine_holter.ingest.dr200 import load_native_flash
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.types import Recording


_NATIVE_BLOCK_LENGTH = (512).to_bytes(4, "little")
_NATIVE_SIGNATURE = b"SampleRate="


def _looks_native(path: Path) -> bool:
    """A native flash.dat under another name: its first block carries the
    block length and the ASCII metadata."""
    try:
        with path.open("rb") as handle:
            head = handle.read(512)
    except OSError:
        return False
    return head[:4] == _NATIVE_BLOCK_LENGTH and _NATIVE_SIGNATURE in head


def load_recording(input_path: str) -> Recording:
    """Load a supported recording.

    WFDB inputs may be supplied as a base path or as their ``.hea`` file.
    Native DR200/DR400 SD-card inputs are named ``flash.dat`` or are any
    ``.dat`` file whose first block carries the recorder metadata.
    """
    path = Path(input_path)
    suffix = path.suffix.casefold()

    if suffix == ".hea":
        record_path = str(path.with_suffix(""))
        return load_local_record(record_path, source=record_path)

    if path.name.casefold() == "flash.dat" or (suffix == ".dat" and _looks_native(path)):
        return load_native_flash(path)

    if Path(f"{input_path}.hea").is_file():
        return load_local_record(input_path, source=input_path)

    raise ValueError(
        "Unsupported recording input. Select a WFDB .hea/base path or a native "
        "DR200/DR400 flash.dat (any name)."
    )
