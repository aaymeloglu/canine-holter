# Diary events from the native file

**Date:** 2026-09-03
**Status:** approved design.

## Evidence

On 2026-09-03 Andy wore the DR200 for 13.7 minutes off a dog and pressed
the event button once for each of its eight diary entries. The card's
`flash.dat` (SHA-256 `9acdcd5c...`, copied to `~/Downloads/flash-events.dat`)
shows how a press is stored:

- The metadata block lists the diary in `DiaryText`, caret-separated:
  `Manual Event^Chest Pain^Dizziness^Palpitation^Chest Pressure^Rapid
  Heart^Short of Breath^Skipped Beat^`.
- There is no event block type. A press is stored *in the ECG block being
  written at the time*: bytes 470 and 471, zero in every other block of
  every recording we hold (51,913 blocks of 08-25, 266,289 of 08-27), are
  non-zero in exactly eight blocks. Byte 471 takes each value 1 to 8 once,
  one per press: the 1-based index into `DiaryText`. Byte 470 takes
  0, 5, 7 or 8 and is not understood.
- The recorder's rolling debug log in bytes 472..508 confirms the reading
  independently: lines of the form `EV <source position> 5484 <n> <type>`
  name the marked block's source position and the same type index for
  three of the eight presses.
- The ECG payload, checksum and source position of a marked block are
  ordinary; the block is not repeated or skipped.

The consequence for the parser is the important part. `dr200.py` reads
bytes 466..472 as the session key (u16 sequence number, u32 serial) and
stops at the first block whose key differs, which is how a reused card's
stale blocks are excluded. A press changes bytes 470..471, so the parser
stops at the first press: it returns 219 s of this 13.7 min recording.
Had Teeny's owner pressed the button during a collapse on 08-27, the
recording as parsed would have ended there, silently.

The serial's last five digits fit a u16 (52837, 27226), and the real
files carry zeros in 470..471 throughout, so the key is 466..470 (u16
sequence, u16 serial) and 470..471 is the event field.

## Design

`ingest/dr200.py`:

- The session key becomes bytes 466..470. Bytes 470..471 are read from
  every ECG block; a non-zero byte 471 is a diary event of that type at
  the block's first sample. Byte 470 is kept with the event as an
  unexplained value (`detail`), not interpreted.
- `_parse_metadata` gains the diary: `DiaryText` split on `^`, empty
  trailing entry dropped. A type index beyond the diary is not an error
  (the recording is still ECG); the event's label is `Event type <n>`.
- `Recording` gains `events: tuple[DiaryEvent, ...] = ()`, with
  `DiaryEvent(time_sec, type_index, label, detail)` frozen in `types.py`.
  WFDB records carry no events.

`report/`:

- Recording panel: `Diary events` with the count, reference `button
  presses` (absent as a row when there are none? No: printed as `0`, so
  the reader knows the file was read for them).
- Timeline: an `Event` lane, marks at each event.
- A strip section `Diary events`, after the extremes and before the PVC
  sections, one strip per event centred on the press, all leads,
  nothing shaded, captioned with the diary label and the heart rate over
  the strip; capped like the other sections. The moment someone pressed
  the button is the strip a cardiologist wants first when the dog has
  collapse episodes.
- Quality gating is unchanged: an event inside an excluded span is still
  listed, and its strip shows whatever the leads recorded.

`docs/dr200-format.md`: the event-mark section rewritten from this
evidence; the table's row for offset 466 split into sequence, serial,
and the event field.

## Testing

Parser: a fixture with an event block in the middle keeps every block
after it and reports the event with the right time, type index, label,
and detail; a type beyond the diary gets the fallback label; a file with
no diary text and an event still loads; the real-file numbers above as an
acceptance check (486 blocks, 8 events with types 2,3,4,5,6,7,8,1 in
block order). Report: the row, the lane, the section and its caption.

## Not known

What byte 470 means, whether a second press inside the same block is
kept, and whether the DR400 marks presses the same way (its firmware
writes different diagnostic bytes; the event field is presumably shared,
but no DR400 press has been recorded).
