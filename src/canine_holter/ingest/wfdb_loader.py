import numpy as np
import wfdb
from canine_holter.types import Recording


def load_local_record(record_path: str, source: str) -> Recording:
    """Load a local WFDB record (a .dat/.hea pair sharing `record_path` as
    their base path, e.g. 'tests/fixtures/mitdb_119/119')."""
    record = wfdb.rdrecord(record_path)
    channels = np.ascontiguousarray(record.p_signal.T, dtype=np.float64)
    return Recording(
        samples=channels[0],
        sample_rate=float(record.fs),
        start_time=None,
        source=source,
        channels=channels,
        channel_names=tuple(record.sig_name),
    )
