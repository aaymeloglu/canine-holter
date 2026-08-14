"""Recording format detection and shared ingestion entry point."""

import re
from pathlib import Path

from canine_holter.ingest.dr200 import load_decoded_channel, load_native_flash
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.types import Recording


_DR200_CHANNEL_NAME = re.compile(r"flashc[0-2]\.dat", re.IGNORECASE)


def load_recording(input_path: str) -> Recording:
    """Load a supported recording, detecting WFDB and decoded DR200 inputs.

    WFDB inputs may be supplied as a base path or as their ``.hea`` file.
    Native DR200 SD-card inputs are named ``flash.dat``. Decoded DR200 inputs
    are recognized as ``flashc0.dat`` through ``flashc2.dat`` or any file with
    a ``.raw`` suffix.
    """
    path = Path(input_path)
    suffix = path.suffix.casefold()

    if suffix == ".hea":
        record_path = str(path.with_suffix(""))
        return load_local_record(record_path, source=record_path)

    if suffix == ".raw" or _DR200_CHANNEL_NAME.fullmatch(path.name):
        return load_decoded_channel(path)

    if path.name.casefold() == "flash.dat":
        return load_native_flash(path)

    if Path(f"{input_path}.hea").is_file():
        return load_local_record(input_path, source=input_path)

    raise ValueError(
        "Unsupported recording input. Select a WFDB .hea/base path, native DR200 "
        "flash.dat, or decoded flashc0.dat, flashc1.dat, flashc2.dat, or .raw channel."
    )
