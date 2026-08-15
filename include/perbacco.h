#ifndef PERBACCO_H
#define PERBACCO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(PB_BUILD_SHARED)
#define PB_API __declspec(dllexport)
#elif defined(_WIN32)
#define PB_API __declspec(dllimport)
#else
#define PB_API __attribute__((visibility("default")))
#endif

#define PB_ABI_VERSION 1u
#define PB_DEFAULT_TOP_K 1000u

typedef uint32_t pb_record_id;

typedef enum pb_status {
    PB_OK = 0,
    PB_DONE = 1,
    PB_BUDGET_EXHAUSTED = 2,
    PB_EINVAL = -1,
    PB_ENOMEM = -2,
    PB_ESTATE = -3,
    PB_ECONFLICT = -4,
    PB_EOVERFLOW = -5,
    PB_EVERSION = -6,
    PB_EINTERNAL = -7
} pb_status;

typedef enum pb_method {
    PB_METHOD_PERBACCO = 0,
    PB_METHOD_PERBAC = 1,
    PB_METHOD_ONLINE = 2,
    PB_METHOD_SUBOPT = 3
} pb_method;

typedef enum pb_cda {
    PB_CDA_NONE = 0,
    PB_CDA_LOUVAIN = 1,
    PB_CDA_EXTERNAL = 2
} pb_cda;

typedef enum pb_batch_kind {
    PB_BATCH_COMMUNITY = 0,
    PB_BATCH_CURRENT = 1
} pb_batch_kind;

typedef struct pb_edge {
    pb_record_id u;
    pb_record_id v;
    double weight;
} pb_edge;

typedef struct pb_config {
    uint32_t struct_size;
    uint32_t abi_version;
    pb_method method;
    pb_cda cda;
    uint32_t batch_size;
    uint32_t top_k;
    uint64_t seed;
    uint64_t max_queries;
    double lambda_w;
    double louvain_resolution;
    double louvain_threshold;
} pb_config;

/* External communities are final heavy communities. They must be disjoint.
 * offsets has community_count + 1 entries and nodes has offsets[last] entries.
 * Records not listed remain in the residual graph for the final phase. */
typedef struct pb_communities {
    size_t community_count;
    const size_t *offsets;
    const pb_record_id *nodes;
} pb_communities;

typedef struct pb_batch {
    pb_batch_kind kind;
    size_t len;
    pb_record_id *records;
} pb_batch;

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

typedef struct pb_engine pb_engine;

/* Initialize every configuration field, including ABI guards and defaults.
 * Call this before overriding individual fields. A NULL pointer is ignored. */
PB_API void pb_config_init(pb_config *config);

/* Copying constructor. Edges, communities, and truth labels are validated and
 * copied; the caller may release them after success. On failure *out_engine is
 * NULL. truth_labels, when present, contains node_count entries. */
PB_API pb_status pb_engine_create(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine);

/* Memory-scaled constructor: edges is normalized and sorted in place and must
 * remain alive and unchanged until the engine is freed. The engine borrows it;
 * communities and truth labels are still copied. */
PB_API pb_status pb_engine_create_borrowed(
    uint32_t node_count,
    pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine);

/* Ground-truth-only baseline. Similarity edges are accepted for API symmetry
 * but ignored; callers may pass NULL and zero to avoid loading the graph. */
PB_API pb_status pb_engine_create_subopt(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    uint32_t batch_size,
    uint64_t seed,
    uint64_t max_queries,
    const uint32_t *truth_labels,
    pb_engine **out_engine);

/* records must have capacity of at least config.batch_size. PB_OK creates one
 * outstanding batch; PB_DONE and PB_BUDGET_EXHAUSTED are normal terminal
 * statuses and create no batch. */
PB_API pb_status pb_engine_next_batch(
    pb_engine *engine,
    pb_record_id *records,
    size_t capacity,
    pb_batch *out_batch);

/* cluster_labels is aligned with records from the outstanding batch.
 * Label values are arbitrary; only equality is significant. */
PB_API pb_status pb_engine_submit_partition(
    pb_engine *engine,
    const uint32_t *cluster_labels,
    size_t label_count);

PB_API pb_status pb_engine_group_members(
    const pb_engine *engine,
    pb_record_id representative,
    pb_record_id *members,
    size_t capacity,
    size_t *out_count);

/* Copy current counters into caller-owned storage. */
PB_API pb_status pb_engine_get_stats(const pb_engine *engine, pb_stats *out_stats);

/* Snapshots contain a deterministic query journal and exclude the immutable
 * graph. Snapshot/restore are only valid when no batch awaits submission. */
PB_API pb_status pb_engine_snapshot_size(const pb_engine *engine, size_t *out_size);
PB_API pb_status pb_engine_snapshot(const pb_engine *engine, void *buffer, size_t capacity, size_t *out_size);
PB_API pb_status pb_engine_restore(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    const void *snapshot,
    size_t snapshot_size,
    pb_engine **out_engine);

/* Diagnostic strings are borrowed. pb_engine_last_error is valid only until
 * the engine is freed; pb_strerror returns static storage. */
PB_API const char *pb_engine_last_error(const pb_engine *engine);
PB_API const char *pb_strerror(pb_status status);
/* Release an engine created or restored by this API. Passing NULL is safe. */
PB_API void pb_engine_free(pb_engine *engine);

#ifdef __cplusplus
}
#endif

#endif
