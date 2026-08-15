# C API reference

This manual describes ABI version `PB_ABI_VERSION == 1`. The stable source of
truth is the public header; the reference is handwritten to make ownership,
capacity, and state rules explicit.

```c
#include <perbacco.h>
```

## IDs and constants

`pb_record_id` is `uint32_t`. Valid node IDs are dense integers in
`[0, node_count)`. `PB_DEFAULT_TOP_K` is `1000`.

## Status codes

| Status | Meaning |
| --- | --- |
| `PB_OK` | operation succeeded |
| `PB_DONE` | no inferable candidate batch remains |
| `PB_BUDGET_EXHAUSTED` | the exact submitted-query limit was reached |
| `PB_EINVAL` | invalid pointer, value, edge, capacity contract, or shape |
| `PB_ENOMEM` | allocation failed |
| `PB_ESTATE` | lifecycle violation, commonly an outstanding batch |
| `PB_ECONFLICT` | oracle partition contradicts known state or truth |
| `PB_EOVERFLOW` | numeric overflow or insufficient output capacity |
| `PB_EVERSION` | ABI, snapshot, graph, truth, community, or configuration identity mismatch |
| `PB_EINTERNAL` | internal invariant failed |

`PB_DONE` and `PB_BUDGET_EXHAUSTED` are not errors. The latter leaves
`pb_stats.finished` false when the graph itself is still schedulable.

## Methods and batch kinds

`pb_method` has `PB_METHOD_PERBACCO`, `PB_METHOD_PERBAC`, `PB_METHOD_ONLINE`,
and `PB_METHOD_SUBOPT`. SubOpt requires `truth_labels`, disables communities,
and ignores similarity edges.

`pb_cda` has `PB_CDA_NONE`, `PB_CDA_LOUVAIN`, and `PB_CDA_EXTERNAL`.
Community detection affects only pERbacco. `pb_batch_kind` reports
`PB_BATCH_COMMUNITY` or `PB_BATCH_CURRENT`.

## Edge

```c
typedef struct pb_edge {
    pb_record_id u;
    pb_record_id v;
    double weight;
} pb_edge;
```

Edges are undirected. Endpoints must differ and be below `node_count`; weights
must be finite and non-negative; duplicate unordered endpoint pairs are
invalid. Construction scales weights by the maximum to `[0, 1]`.

## Configuration

Initialize this structure; do not zero or partially initialize it yourself.

| Field | Type | Default | Contract |
| --- | --- | ---: | --- |
| `struct_size` | `uint32_t` | `sizeof(pb_config)` | set by `pb_config_init` |
| `abi_version` | `uint32_t` | `PB_ABI_VERSION` | set by `pb_config_init` |
| `method` | `pb_method` | `PB_METHOD_PERBACCO` | one defined enum value |
| `cda` | `pb_cda` | `PB_CDA_LOUVAIN` | one defined enum value |
| `batch_size` | `uint32_t` | `10` | `2..node_count` |
| `top_k` | `uint32_t` | `1000` | positive GreedyHS limit |
| `seed` | `uint64_t` | `42` | deterministic engine-local RNG seed |
| `max_queries` | `uint64_t` | `0` | zero is unlimited |
| `lambda_w` | `double` | `0.05` | heavy-community density in `[0, 1]` |
| `louvain_resolution` | `double` | `0.0` | zero selects `0.75` up to 20k nodes, `0.50` above |
| `louvain_threshold` | `double` | `1e-4` | non-negative stopping threshold |

```c
void pb_config_init(pb_config *config);
```

A `NULL` pointer is ignored.

## External communities

```c
typedef struct pb_communities {
    size_t community_count;
    const size_t *offsets;
    const pb_record_id *nodes;
} pb_communities;
```

`offsets` has `community_count + 1` entries, begins at zero, and is
non-decreasing. Community `i` occupies
`nodes[offsets[i]..offsets[i + 1])`. Communities must be disjoint and contain
valid IDs. They are already-final heavy communities; records omitted from them
remain in the residual graph. Constructors copy this data.

## Batch and statistics

```c
typedef struct pb_batch {
    pb_batch_kind kind;
    size_t len;
    pb_record_id *records;
} pb_batch;
```

On `PB_OK`, `records` points to the buffer passed to `pb_engine_next_batch` and
`len` is between 2 and `config.batch_size`. The caller owns the buffer.

```c
typedef struct pb_stats {
    uint64_t query_count;
    uint64_t discovered_matches;
    uint64_t candidate_edges;
    uint64_t community_batches;
    uint64_t current_batches;
    uint64_t community_records;
    double temperature;
    int finished;
} pb_stats;
```

`discovered_matches` counts record pairs implied by accepted component merges.
Optional truth labels add conflict validation but are not required for the
counter. `candidate_edges` is the current live heap size. Statistics are copied
into caller storage.

## Constructors

```c
pb_status pb_engine_create(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine);
```

Validates and copies all inputs. `edges` may be `NULL` only when `edge_count` is
zero. `truth_labels`, when present, has `node_count` entries. On failure,
`*out_engine` is `NULL`.

```c
pb_status pb_engine_create_borrowed(
    uint32_t node_count,
    pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine);
```

Normalizes and sorts `edges` in place. That writable array is borrowed until
`pb_engine_free`; all other inputs are copied. Do not reuse or mutate the edge
array concurrently.

```c
pb_status pb_engine_create_subopt(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    uint32_t batch_size,
    uint64_t seed,
    uint64_t max_queries,
    const uint32_t *truth_labels,
    pb_engine **out_engine);
```

Convenience constructor for the truth-only baseline. Edges are accepted for ABI
symmetry but ignored; pass `NULL, 0` to avoid loading them. `truth_labels` is
required.

## Selection and submission

```c
pb_status pb_engine_next_batch(
    pb_engine *engine,
    pb_record_id *records,
    size_t capacity,
    pb_batch *out_batch);
```

`capacity` must be at least configured `batch_size`. On `PB_OK`, a batch becomes
outstanding until one successful submission. On terminal statuses, no batch is
outstanding.

```c
pb_status pb_engine_submit_partition(
    pb_engine *engine,
    const uint32_t *cluster_labels,
    size_t label_count);
```

Labels align with the outstanding batch. `label_count` must equal its length.
Only equality matters; values need not be dense. Validation is transactional.

## Component members

```c
pb_status pb_engine_group_members(
    const pb_engine *engine,
    pb_record_id representative,
    pb_record_id *members,
    size_t capacity,
    size_t *out_count);
```

The representative may be any record ID; its current root is resolved. For a
two-call allocation pattern, first pass `members=NULL, capacity=0`. The function
writes the required count and returns `PB_EOVERFLOW` for a nonempty component.
Allocate that many IDs and call again. A short buffer is filled up to capacity,
reports the full requirement, and returns `PB_EOVERFLOW`.

## Statistics

```c
pb_status pb_engine_get_stats(const pb_engine *engine, pb_stats *out_stats);
```

Copies a consistent counter snapshot. Both pointers are required.

## Snapshot and restore

```c
pb_status pb_engine_snapshot_size(const pb_engine *engine, size_t *out_size);
pb_status pb_engine_snapshot(
    const pb_engine *engine,
    void *buffer,
    size_t capacity,
    size_t *out_size);
```

Both reject an outstanding batch with `PB_ESTATE`. `snapshot` reports the full
required size in `out_size`; insufficient capacity returns `PB_EOVERFLOW`.

```c
pb_status pb_engine_restore(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    const void *snapshot,
    size_t snapshot_size,
    pb_engine **out_engine);
```

Creates a new copying engine and replays its journal. All immutable inputs must
match the original. The snapshot buffer remains caller-owned.

## Diagnostics and destruction

```c
const char *pb_engine_last_error(const pb_engine *engine);
const char *pb_strerror(pb_status status);
void pb_engine_free(pb_engine *engine);
```

Both diagnostic strings are borrowed and must not be freed. The engine-specific
string is invalid after `pb_engine_free`. Freeing `NULL` is safe; using an engine
after free is undefined behavior.
