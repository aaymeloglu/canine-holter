from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass(frozen=True)
class DiaryEvent:
    """A press of the recorder's event button.

    time_sec: seconds from the start of the recording, to the block the
        press was stored in (1.7 s at 180 Hz)
    type_index: the recorder's 1-based index into its diary list
    label: the diary entry's text, or "Event type <n>" when the recording
        carries no entry for it
    detail: a recorder byte stored with the press whose meaning is unknown
    """
    time_sec: float
    type_index: int
    label: str
    detail: int


@dataclass(frozen=True, eq=False)
class Recording:
    """An ECG recording in millivolts.

    eq=False: the default dataclass equality would compare the sample
    arrays and raise. frozen=True stops field reassignment only; the
    arrays themselves must not be mutated after construction.

    samples: the lead quality gating judges and, for a single-lead input,
        the lead beats are detected on (1-D)
    sample_rate: samples per second
    start_time: wall-clock time the recording began, if known
    source: where the recording came from (a fixture name or file path)
    channels: every recorded lead, shape (n_channels, n_samples), in
        recorder order; None when the input carried a single lead. Beat
        detection runs on every lead and keeps the beats they agree on;
        the report draws them all.
    channel_names: one name per channels row
    events: the recorder's diary-button presses, in time order; empty for
        formats that carry none
    """
    samples: np.ndarray
    sample_rate: float
    start_time: datetime | None
    source: str
    channels: np.ndarray | None = None
    channel_names: tuple[str, ...] = ()
    events: tuple[DiaryEvent, ...] = ()

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
    label: "N" (normal), "V" (PVC), "E" (ventricular escape beat: wide and
        late), "U" (undetermined), or None (not yet classified)
    """
    time: float
    rr_interval: float | None
    qrs_duration: float | None
    label: str | None
