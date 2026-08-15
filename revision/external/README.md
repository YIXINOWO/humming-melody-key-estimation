# External resource layout

This directory is intentionally documentation-only in the public release.

Expected local layout for the full external experiments:

```text
revision/external/
|-- archives/
|   `-- HumTrans-all_wav.zip
|-- checkpoints.zip
|-- extracted/
|   |-- humtrans_midi/
|   |-- humtrans_wav_valid/
|   `-- humtrans_wav_test/
`-- repos/
    |-- HumTrans-main/
    `-- ROSVOT-main/
```

`HumTrans-main/` must provide the official `valid_keys.txt`, `test_keys.txt`,
ground-truth MIDI archive, and baseline MIDI archives. `ROSVOT-main/` must be an
upstream checkout with the official checkpoint folders extracted beneath
`checkpoints/`.

Do not commit any downloaded archive, audio file, repository snapshot, or model
weight. See the repository-root `THIRD_PARTY_RESOURCES.md` for upstream routes
and hashes.

