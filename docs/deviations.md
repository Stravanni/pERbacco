# Known deviations

This page separates verified agreement, understood implementation differences,
and open comparison gaps. Generated numeric evidence belongs in
`artifacts/deviations.json`; artifacts are intentionally not committed.

## Verified exact result: Table III

For batch size 10, the maintained bound implementation matches the paper:

| Dataset | Lower | Upper |
| --- | ---: | ---: |
| Cora | 137 | 137 |
| Camera | 2436 | 2455 |
| Funding | 1612 | 1640 |
| WDC-80 | 391 | 396 |
| Voters | 1315 | 1354 |
| Synth10k | 3514 | 3544 |

This may be labeled an exact Table III reproduction.

## Figure 4 status

The runner produces all 32 combinations of eight panels and four schedulers at
the configured `3 phi` exact budget, using a local ground-truth oracle. Those
curves are **maintained C implementation results**. Completeness of a curve set
does not establish numerical fidelity to the paper.

The paper exposes Figure 4 graphically, not as a numeric data file. The project
therefore records full maintained curves plus values at `phi`, `2 phi`, and
`3 phi`, but does not claim a pointwise `0.02` reproduction. Large residual
differences remain open.

## Known causes

### Louvain identity

The published prototype uses NetworkX Louvain. The zero-dependency library uses
its own deterministic, multilevel weighted C implementation; Table IV also
exercises seeded igraph Louvain and Leiden through external communities. The C
implementation follows the same local-move, contraction, resolution, and
modularity-threshold semantics. It is not partition-identical because its
seeded node order and tie behavior are independent. On Funding, the maintained
C path covers 97.97% of records, a validation-only NetworkX 3.6 run covers
98.36%, and the paper reports 98%.

### Stable ties

The prototype's sets can let process-specific hash order select between equal
scores. The C engine resolves ties from stable dense IDs and a local seeded RNG.
Different early batches alter later components and benefits.

### Exact budget semantics

Prototype loops using `<= max_query` may execute an additional query. The C
engine returns `PB_BUDGET_EXHAUSTED` before exceeding the configured number of
submitted partitions. `stats.finished` distinguishes a budget stop from a
terminal graph.

### Benefit and identity repairs

Prototype invalidation iterates combinations of changed old entities. With only
one changed entity, that iteration can be empty and stale benefits survive. The
maintained engine updates every affected incident edge. It also resolves current
representatives rather than trusting identities captured before contractions.
The non-match map is authoritative during heap eligibility, so migrating a
known non-match through component contraction cannot leave its old benefit edge
schedulable.

### Recursive external community mapping

The historical Leiden path converts sets to lists and crosses igraph local
indices with an unstable ordering. The maintained adapter keeps an explicit
local-to-global mapping. Restoring the identity bug merely to recover a reported
number is out of scope.

### SubOpt

Maintained SubOpt follows the truth-graph ranking and GreedyHS1000 while lazily
emitting only the best current edges. It avoids loading similarity data and does
not reproduce shortcut or off-by-one behavior from older prototype paths.

### Cache provenance

Old manifests tied a curve to configuration and a curve checksum but not to an
engine build. A checksum proves that a file is unchanged, not that current code
created it. Format-version-2 manifests include an implementation SHA-256 over
the loaded native library, result-affecting Python modules, Python runtime, and
optional graph/community dependency versions, so older caches are recomputed.

## What remains unresolved

The items above establish multiple reasons for divergence but do not isolate a
single cause for every large curve gap. The original runtime dependency
versions, unordered tie sequence, and numeric Figure 4 series are not all
available. Until a controlled side-by-side trace against a pinned prototype is
performed, the unexplained remainder must stay labeled unresolved.

No maintained result is rounded, reordered, granted an extra query, or relabeled
to resemble the paper. See [research status](research-status.md) for the concise
user-facing distinction.
