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

    samples: the analysis lead, in millivolts (1-D)
    sample_rate: samples per second
    start_time: wall-clock time the recording began, if known
    source: identifies where this recording came from (e.g. a fixture name or file path)
    channels: every recorded lead, shape (n_channels, n_samples), in
        millivolts and recorder order, for display; None when the input
        carried a single lead. Only the report reads it - analysis stays on
        `samples`.
    channel_names: one name per channels row
    """
    samples: np.ndarray
    sample_rate: float
    start_time: datetime | None
    source: str
    channels: np.ndarray | None = None
    channel_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Fail closed: a lead set that does not line up with the analysis
        # lead must not be drawn as if it did.
        if self.channels is None:
            return
        if self.channels.ndim != 2:
            raise ValueError(f"channels must be 2-D (n_channels, n_samples), got shape {self.channels.shape}")
        if self.channels.shape[1] != len(self.samples):
            raise ValueError(
                f"channels length {self.channels.shape[1]} differs from samples length {len(self.samples)}"
            )
        if len(self.channel_names) != self.channels.shape[0]:
            raise ValueError(
                f"channel_names has {len(self.channel_names)} entries for {self.channels.shape[0]} channels"
            )


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
