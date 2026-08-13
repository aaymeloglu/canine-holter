from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True, eq=False)
class Recording:
    """A single-lead ECG recording, in millivolts, with metadata.

    Equality is identity-based (eq=False): the default dataclass eq would
    compare field tuples, which calls bool() on the samples array comparison
    and raises ValueError, since np.ndarray.__eq__ returns an array rather
    than a bool. Identity comparison is the sane default for objects
    wrapping large sample buffers.

    Note: frozen=True only prevents reassigning fields (e.g. rec.source = "x"
    raises FrozenInstanceError). It does not make the samples array itself
    immutable - numpy arrays are mutable in place regardless of Python-level
    frozen semantics, so callers must not mutate .samples after construction.

    samples: the raw ECG signal, in millivolts
    sample_rate: samples per second
    start_time: wall-clock time the recording began, if known
    source: identifies where this recording came from (e.g. a fixture name or file path)
    """
    samples: np.ndarray
    sample_rate: float
    start_time: datetime | None
    source: str


@dataclass(frozen=True)
class Beat:
    """A single detected heartbeat.

    time: seconds from the start of the recording
    rr_interval: seconds since the previous beat; None for the first beat
    qrs_duration: seconds; None if QRS delineation failed for this beat
    label: "N" (normal), "V" (PVC), "U" (undetermined), or None (not yet classified)
    """
    time: float
    rr_interval: float | None
    qrs_duration: float | None
    label: str | None
