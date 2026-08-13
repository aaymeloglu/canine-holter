import wfdb
from canine_holter.types import Recording


def load_local_record(record_path: str, source: str, channel: int = 0) -> Recording:
    """Load a local WFDB record (a .dat/.hea pair sharing `record_path` as
    their base path, e.g. 'tests/fixtures/mitdb_119/119') into a Recording.

    Uses the first signal channel by default (channel=0) since this
    pipeline is single-lead.
    """
    record = wfdb.rdrecord(record_path)
    samples = record.p_signal[:, channel]
    return Recording(
        samples=samples,
        sample_rate=float(record.fs),
        start_time=None,
        source=source,
    )
