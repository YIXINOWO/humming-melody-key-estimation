# Derived frame table

`frame_onset_offset_oof.csv.gz` is the compressed frame-level table used as the
input to the revised Molina nested-CV analysis. Pandas reads it directly; manual
decompression is not required.

The table contains recording ID, 10 ms frame time, CREPE pitch/confidence,
voicing, pitch/energy/flux/context features, temporal lags, historical labels,
and frozen historical probability columns. The revised experiment rebuilds
boundary labels from the audited source annotations and does not treat the
historical probability columns as outer-test predictions.

Uncompressed SHA-256:

```text
eda1c0664911157ba136fbe0e24e5d4712cba611788e75b0e6a04f8b2189b00f
```

Compressed-file SHA-256:

```text
48fb805c93d44f11bdb25d7151bd3bd9111e968a856993b0f6f48c3ab48ba157
```

The remaining CSV files in this directory are explicitly historical inputs
needed by diagnostic or figure-generation scripts. They are not the formal
revised nested-CV result; that result is under
`revision/results/molina_nested_cv/`.

