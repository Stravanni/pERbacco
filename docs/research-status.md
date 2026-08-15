# Research status and paper mapping

## Reference

Lorenzo Balzotti, Donatella Firmani, Luca Gagliardelli, and Giovanni Simonini,
“Entity Resolution via Batched Oracle Queries,” 2026.
[arXiv:2606.24407](https://arxiv.org/abs/2606.24407).

The stable arXiv record is the documentation link of record. Local paper PDFs
are deliberately excluded from the repository and documentation build.

## What is maintained

The supported library is the C99 engine in `src/perbacco.c`, the public ABI in
`include/perbacco.h`, and the wrapper in `python/perbacco`. The top-level Python
research scripts are retained as historical compatibility/reference material;
they are not the implementation used by the package or current experiments.

!!! warning "Do not conflate two implementations"

    The paper reports results from the original Python prototype and its
    dependency stack. Current experiment artifacts come from a corrected,
    deterministic C implementation. The method names are shared, but exact
    execution paths are not.

## Method mapping

| Paper method | Python enum | C enum | Maintained behavior |
| --- | --- | --- | --- |
| pERbacco | `Method.PERBACCO` | `PB_METHOD_PERBACCO` | mean benefit, temperature policy, recursive heavy communities |
| pERbac | `Method.PERBAC` | `PB_METHOD_PERBAC` | mean benefit, no community phase |
| Online | `Method.ONLINE` | `PB_METHOD_ONLINE` | maximum current edge-probability benefit |
| SubOpt | `Method.SUBOPT` | `PB_METHOD_SUBOPT` | truth-only paper ranking with GreedyHS1000 |

SubOpt is an experimental comparison, not a deployable oracle policy, because
it uses hidden truth labels while selecting each batch.

## Artifact labels

| Artifact | Status | Permitted description |
| --- | --- | --- |
| Table III | exact on all six datasets | “exact Table III reproduction” |
| Table IV | measured with corrected scheduler/external backends | “maintained implementation comparison” |
| Figure 4 | 32 ground-truth-oracle curves from C | “maintained C implementation results” |

Do **not** call the current Figure 4 an exact paper reproduction. The paper does
not publish a numeric curve series, so a rigorous point-by-point tolerance check
is unavailable without lossy digitization.

## Why the plots differ from the paper

Several concrete execution differences can move whole curves:

1. **Community implementation.** The prototype Figure 4 path uses NetworkX
   Louvain. Maintained Figure 4 uses an independent dependency-free, multilevel
   C Louvain backend with the same paper defaults. Recursive communities can
   still differ because node order and ties are independently deterministic,
   changing early queries where pERbacco gains most of its advantage.
2. **Deterministic ties.** Prototype candidate sets inherit Python set/hash
   iteration order. The C engine uses stable dense IDs and an engine-local seeded
   PRNG for explicit ties. Equal-benefit batches can therefore diverge early and
   compound after state updates.
3. **Exact budgets.** Prototype loops using `<= max_query` can admit one extra
   query. The C engine checks the cap before selection and stops at exactly
   `max_queries` submitted partitions.
4. **Benefit invalidation and representatives.** The maintained engine
   recomputes every candidate incident to a contracted or blocked component and
   resolves live representatives. The prototype can retain stale benefit rows
   when only one previous entity changes and can consult stale identities.
5. **SubOpt path.** The maintained truth-only implementation uses the paper's
   ranked truth graph and GreedyHS1000 without materializing a quadratic graph.
   It does not recreate prototype shortcut, tie, or permissive-loop behavior.
6. **Previously stale run caches.** Older local artifacts validated their curve
   checksum and configuration but did not identify the engine build. Manifests
   now include a hash of the loaded C library, result-affecting Python code,
   Python runtime, and optional backend versions; old caches are rejected.

These mechanisms explain why exact equality should not be expected. They do not
yet quantify every large curve gap—especially the largest dataset/backend
differences. Because the paper supplies plotted curves rather than source
numbers and the original dependency versions/tie order are not fully pinned,
the residual discrepancy remains unresolved. The project records it rather
than adjusting labels, budgets, or plots to force a visual match.

## Correctness choices in the C implementation

The maintained engine validates inputs and complete partitions before mutation,
uses exact budgets, invalidates all affected benefits, retains live union-find
representatives, normalizes finite non-negative weights, and snapshots a
checksummed deterministic journal. These choices are library guarantees even
when they change a published prototype trajectory.

Full technical detail is in [known deviations](deviations.md) and the
[architecture](architecture.md). Reproduction commands and artifact semantics
are in [experiments](experiments.md).
