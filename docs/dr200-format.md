# DR200 recording format research

Research date: 2026-08-13

## What is on the SD card

The DR200 writes a preallocated file named `flash.dat`. It is a proprietary NorthEast Monitoring container, not an ISHNE Holter file. The native file mixes recorder metadata, ECG blocks, recorder diagnostics, and zero-filled unused space.

NorthEast's [DR200 operator manual](https://www.nemon.com/supportfiles/NEMM019-Rev-T_DR200_manual.pdf) documents the supported vendor conversion, but not the native container. After HE/LX analyzes `flash.dat`, its `unpackdc` utility converts `datacard.dat` into:

- `flashc0.dat`, `flashc1.dat`, and `flashc2.dat`;
- signed 16-bit little-endian samples;
- 180 samples/second per channel;
- 12.5 microvolts per count;
- `0x8000` in all channels for a detected pacemaker pulse.

Those values agree with NorthEast's [DR200 specifications](https://www.nemon.com/products-holter-event-recorders/dr200-he-holter-and-event-recorder/). Paul Bourke independently documented the [same converted format and published a `datacard.dat` fixture with its three decoded channels](https://paulbourke.net/dataformats/holter/).

## Native `flash.dat` layout

The supported native mode identifies itself as `SampleStorageFormat=1` and `SampleRate=180`. The file consists of 512-byte blocks followed, when the card was preallocated beyond the recording, by zero blocks or a final partial run of zero bytes.

Every active block satisfies:

```text
sum(block[0:508]) + little_endian_u32(block[508:512]) == 0x4CB31
```

The first active block contains newline-delimited ASCII metadata, including `SampleRate`, `SampleStorageFormat`, `start_date`, and `start_time`. Three-channel ECG blocks have this layout:

| Offset | Length | Meaning |
| ---: | ---: | --- |
| 0 | 4 | Little-endian block length, always 512 |
| 4 | 1 | Block type `0x1e` |
| 5 | 1 | Zero for this recording mode |
| 6 | 4 | Source position; advances by 1,216 bytes per ECG block |
| 10 | 456 | 304 timepoints × 3 channels × 4-bit differences |
| 466 | 42 | Recorder diagnostic/reserved bytes |
| 508 | 4 | Checksum complement described above |

The payload is read low nibble first. Nibbles are interleaved `channel 0, channel 1, channel 2` for each timepoint and accumulated continuously across blocks. The difference table, indexed by nibble, is:

```text
0, +1, +3, +6, +12, +21, +38, +70,
pace, -70, -38, -21, -12, -6, -3, -1
```

Nibble 8 occurs simultaneously on all three channels and represents the documented pacemaker marker. The loader interpolates these locations rather than introducing a false -409.6 mV spike.

## Evidence and validation

NorthEast publishes the [HE/LX Analysis download and manuals](https://www.nemon.com/support-technical-support-training-videos-2/). The official 6.0e installer contains both a demo `flash.dat` and the `procfl.exe` converter. The layout above was derived from that recording and confirmed against the converter's block checks, sample loop, and difference lookup table without bypassing its license check.

Fixture fingerprints:

| Fixture | Size | SHA-256 |
| --- | ---: | --- |
| Official HE/LX 6.0e installer | 178,829,040 bytes | `43bbb6a2f34e9a1ff8f3cde4b759315f995723f425bf3f7f60e0277a39ca4955` |
| Bundled demo `flash.dat` | 26,263,039 bytes | `cd65a6965a752d639166ca01b868eb3bb4d9d1044cfaf24e1f65628243a6459b` |
| Bourke `datacard.dat` | 23,639,496 bytes | `35f392a0247ec9b27f5219f39f768c1b097ec6c4a51c34aa32e2cfb76d2fa9e3` |

The native parser reads 15,583,040 samples from the official demo: 24.0479 hours at 180 Hz, beginning at the embedded timestamp `2010-07-08 11:12:50`. Channel 0 reconstructs to a bounded ECG signal with a range of -12.875 to 12.7375 mV. The public converted fixture contains 15,552,300 samples in each 31,104,600-byte raw channel, which also exercises the decoded-channel loader.

## Current limits

- Native parsing intentionally rejects sample rates other than 180 Hz and `SampleStorageFormat` values other than 1.
- The pipeline currently analyzes channel 0. The native parser accepts an explicit channel index in Python, but channel selection is not exposed in the CLI yet.
- Only one native recording was publicly obtainable. Block length, checksum, sequence, marker alignment, sampling mode, and storage format are validated so an unfamiliar variant fails rather than silently producing a wrong signal.
- The format research establishes mechanical decoding, not clinical accuracy. Reports remain screening/triage aids and require veterinary interpretation.
