from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class Recording:
    """A single-lead ECG recording, in millivolts, with metadata."""
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
