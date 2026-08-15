#include "perbacco.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PB_NONE UINT32_MAX
#define PB_SNAPSHOT_VERSION 2u
#define PB_EPSILON 1e-12

typedef struct pb_u32vec {
    uint32_t *items;
    size_t len;
    size_t cap;
} pb_u32vec;

typedef struct pb_pairmap {
    uint64_t *keys;
    uint32_t *values;
    uint8_t *states;
    size_t cap;
    size_t len;
    size_t tombstones;
} pb_pairmap;

typedef struct pb_dynedge {
    uint32_t u;
    uint32_t v;
    double sum;
    double maximum;
    uint8_t active;
    uint8_t blocked;
} pb_dynedge;

typedef struct pb_index_heap {
    uint32_t *ids;
    uint32_t *positions;
    size_t len;
    size_t cap;
} pb_index_heap;

typedef struct pb_idheap {
    uint32_t *ids;
    size_t len;
    size_t cap;
} pb_idheap;

typedef struct pb_community {
    uint32_t *nodes;
    size_t node_count;
    size_t unqueried_count;
    double weight;
    pb_idheap edges;
} pb_community;

typedef struct pb_journal_entry {
    pb_batch_kind kind;
    uint32_t len;
    uint32_t *records;
    uint32_t *labels;
} pb_journal_entry;

typedef struct pb_temp_edge {
    uint32_t u;
    uint32_t v;
    double weight;
} pb_temp_edge;

struct pb_engine {
    pb_config config;
    uint32_t node_count;
    size_t edge_count;
    pb_edge *edges;
    uint8_t owns_edges;
    uint64_t graph_hash;

    uint32_t *parent;
    uint32_t *component_size;
    uint32_t *member_head;
    uint32_t *member_tail;
    uint32_t *member_next;
    uint8_t *current;
    uint8_t *queried;
    uint32_t *truth_labels;

    pb_u32vec *incident;
    pb_u32vec *nonmatch_incident;
    pb_dynedge *dynamic_edges;
    pb_pairmap edge_map;
    pb_pairmap nonmatch_map;
    pb_index_heap current_heap;

    pb_community *communities;
    size_t community_count;
    size_t community_index;
    uint8_t community_active;
    uint8_t phase_final;
    uint8_t prefer_current;

    double temperature;
    uint64_t community_gain_sum;
    uint64_t community_gain_count;
    uint64_t query_count;
    uint64_t discovered_matches;
    uint64_t community_batches;
    uint64_t current_batches;
    uint8_t finished;

    uint8_t awaiting_submission;
    pb_batch_kind pending_kind;
    uint32_t *pending_records;
    size_t pending_len;
    uint64_t pending_matches_before;

    pb_journal_entry *journal;
    size_t journal_len;
    size_t journal_cap;

    uint64_t rng_state;
    char last_error[256];
};

static void pb_set_error(pb_engine *engine, const char *format, ...) {
    va_list arguments;
    if (engine == NULL) {
        return;
    }
    va_start(arguments, format);
    (void)vsnprintf(engine->last_error, sizeof(engine->last_error), format, arguments);
    va_end(arguments);
}

static void *pb_calloc(size_t count, size_t size) {
    if (size != 0u && count > SIZE_MAX / size) {
        return NULL;
    }
    return calloc(count, size);
}

static int pb_vec_reserve(pb_u32vec *vector, size_t needed) {
    uint32_t *replacement;
    size_t capacity;
    if (needed <= vector->cap) {
        return 1;
    }
    capacity = vector->cap == 0u ? 4u : vector->cap;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) {
            return 0;
        }
        capacity *= 2u;
    }
    replacement = (uint32_t *)realloc(vector->items, capacity * sizeof(*replacement));
    if (replacement == NULL) {
        return 0;
    }
    vector->items = replacement;
    vector->cap = capacity;
    return 1;
}

static int pb_vec_push(pb_u32vec *vector, uint32_t value) {
    if (!pb_vec_reserve(vector, vector->len + 1u)) {
        return 0;
    }
    vector->items[vector->len++] = value;
    return 1;
}

static uint64_t pb_mix64(uint64_t value) {
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31u;
    return value;
}

static uint64_t pb_rng_next(pb_engine *engine) {
    uint64_t value;
    engine->rng_state += UINT64_C(0x9e3779b97f4a7c15);
    value = engine->rng_state;
    return pb_mix64(value);
}

static size_t pb_rng_bounded(pb_engine *engine, size_t upper) {
    if (upper <= 1u) {
        return 0u;
    }
    return (size_t)(pb_rng_next(engine) % (uint64_t)upper);
}

static uint64_t pb_pair_key(uint32_t left, uint32_t right) {
    uint32_t low = left < right ? left : right;
    uint32_t high = left < right ? right : left;
    return ((uint64_t)low << 32u) | (uint64_t)high;
}

static size_t pb_next_power_of_two(size_t value) {
    size_t result = 16u;
    while (result < value) {
        if (result > SIZE_MAX / 2u) {
            return 0u;
        }
        result *= 2u;
    }
    return result;
}

static int pb_pairmap_init(pb_pairmap *map, size_t expected) {
    size_t capacity;
    size_t target;
    if (expected > SIZE_MAX - expected / 2u - 16u) {
        return 0;
    }
    target = expected + expected / 2u + 16u;
    capacity = pb_next_power_of_two(target);
    if (capacity == 0u) {
        return 0;
    }
    map->keys = (uint64_t *)pb_calloc(capacity, sizeof(*map->keys));
    map->values = (uint32_t *)pb_calloc(capacity, sizeof(*map->values));
    map->states = (uint8_t *)pb_calloc(capacity, sizeof(*map->states));
    if (map->keys == NULL || map->values == NULL || map->states == NULL) {
        free(map->keys);
        free(map->values);
        free(map->states);
        memset(map, 0, sizeof(*map));
        return 0;
    }
    map->cap = capacity;
    return 1;
}

static void pb_pairmap_destroy(pb_pairmap *map) {
    free(map->keys);
    free(map->values);
    free(map->states);
    memset(map, 0, sizeof(*map));
}

static size_t pb_pairmap_slot(const pb_pairmap *map, uint64_t key, int *found) {
    size_t mask = map->cap - 1u;
    size_t slot = (size_t)pb_mix64(key) & mask;
    size_t first_tombstone = SIZE_MAX;
    for (;;) {
        if (map->states[slot] == 0u) {
            *found = 0;
            return first_tombstone == SIZE_MAX ? slot : first_tombstone;
        }
        if (map->states[slot] == 1u && map->keys[slot] == key) {
            *found = 1;
            return slot;
        }
        if (map->states[slot] == 2u && first_tombstone == SIZE_MAX) {
            first_tombstone = slot;
        }
        slot = (slot + 1u) & mask;
    }
}

static int pb_pairmap_get(const pb_pairmap *map, uint64_t key, uint32_t *value) {
    int found;
    size_t slot;
    if (map->cap == 0u) {
        return 0;
    }
    slot = pb_pairmap_slot(map, key, &found);
    if (!found) {
        return 0;
    }
    if (value != NULL) {
        *value = map->values[slot];
    }
    return 1;
}

static int pb_pairmap_rehash(pb_pairmap *map, size_t new_capacity) {
    uint64_t *old_keys = map->keys;
    uint32_t *old_values = map->values;
    uint8_t *old_states = map->states;
    size_t old_capacity = map->cap;
    size_t index;

    map->keys = (uint64_t *)pb_calloc(new_capacity, sizeof(*map->keys));
    map->values = (uint32_t *)pb_calloc(new_capacity, sizeof(*map->values));
    map->states = (uint8_t *)pb_calloc(new_capacity, sizeof(*map->states));
    if (map->keys == NULL || map->values == NULL || map->states == NULL) {
        free(map->keys);
        free(map->values);
        free(map->states);
        map->keys = old_keys;
        map->values = old_values;
        map->states = old_states;
        return 0;
    }
    map->cap = new_capacity;
    map->len = 0u;
    map->tombstones = 0u;
    for (index = 0u; index < old_capacity; index++) {
        if (old_states[index] == 1u) {
            int found;
            size_t slot = pb_pairmap_slot(map, old_keys[index], &found);
            map->states[slot] = 1u;
            map->keys[slot] = old_keys[index];
            map->values[slot] = old_values[index];
            map->len++;
        }
    }
    free(old_keys);
    free(old_values);
    free(old_states);
    return 1;
}

static int pb_pairmap_put(pb_pairmap *map, uint64_t key, uint32_t value) {
    int found;
    size_t slot;
    if (map->cap == 0u) {
        return 0;
    }
    if (map->len + map->tombstones + 1u > (map->cap * 7u) / 10u) {
        if (map->cap > SIZE_MAX / 2u || !pb_pairmap_rehash(map, map->cap * 2u)) {
            return 0;
        }
    }
    slot = pb_pairmap_slot(map, key, &found);
    if (found) {
        map->values[slot] = value;
        return 1;
    }
    if (map->states[slot] == 2u) {
        map->tombstones--;
    }
    map->states[slot] = 1u;
    map->keys[slot] = key;
    map->values[slot] = value;
    map->len++;
    return 1;
}

static void pb_pairmap_remove(pb_pairmap *map, uint64_t key) {
    int found;
    size_t slot;
    if (map->cap == 0u) {
        return;
    }
    slot = pb_pairmap_slot(map, key, &found);
    if (found) {
        map->states[slot] = 2u;
        map->len--;
        map->tombstones++;
    }
}

static uint32_t pb_find(pb_engine *engine, uint32_t node) {
    uint32_t root = node;
    while (engine->parent[root] != root) {
        root = engine->parent[root];
    }
    while (engine->parent[node] != node) {
        uint32_t next = engine->parent[node];
        engine->parent[node] = root;
        node = next;
    }
    return root;
}

static uint32_t pb_find_const(const pb_engine *engine, uint32_t node) {
    while (engine->parent[node] != node) {
        node = engine->parent[node];
    }
    return node;
}

static double pb_edge_benefit(const pb_engine *engine, const pb_dynedge *edge) {
    if (!edge->active || edge->blocked) {
        return 0.0;
    }
    if (engine->config.method == PB_METHOD_ONLINE || engine->config.method == PB_METHOD_SUBOPT) {
        return edge->maximum * (double)engine->component_size[edge->u] *
               (double)engine->component_size[edge->v];
    }
    return edge->sum;
}

static int pb_dynamic_better(const pb_engine *engine, uint32_t left_id, uint32_t right_id) {
    const pb_dynedge *left = &engine->dynamic_edges[left_id];
    const pb_dynedge *right = &engine->dynamic_edges[right_id];
    double left_benefit = pb_edge_benefit(engine, left);
    double right_benefit = pb_edge_benefit(engine, right);
    if (left_benefit > right_benefit + PB_EPSILON) {
        return 1;
    }
    if (right_benefit > left_benefit + PB_EPSILON) {
        return 0;
    }
    return left_id < right_id;
}

static void pb_index_heap_swap(pb_index_heap *heap, size_t left, size_t right) {
    uint32_t a = heap->ids[left];
    uint32_t b = heap->ids[right];
    heap->ids[left] = b;
    heap->ids[right] = a;
    heap->positions[a] = (uint32_t)right;
    heap->positions[b] = (uint32_t)left;
}

static void pb_index_heap_up(pb_engine *engine, size_t position) {
    pb_index_heap *heap = &engine->current_heap;
    while (position > 0u) {
        size_t parent = (position - 1u) / 2u;
        if (!pb_dynamic_better(engine, heap->ids[position], heap->ids[parent])) {
            break;
        }
        pb_index_heap_swap(heap, position, parent);
        position = parent;
    }
}

static void pb_index_heap_down(pb_engine *engine, size_t position) {
    pb_index_heap *heap = &engine->current_heap;
    for (;;) {
        size_t left = position * 2u + 1u;
        size_t right = left + 1u;
        size_t best = position;
        if (left < heap->len && pb_dynamic_better(engine, heap->ids[left], heap->ids[best])) {
            best = left;
        }
        if (right < heap->len && pb_dynamic_better(engine, heap->ids[right], heap->ids[best])) {
            best = right;
        }
        if (best == position) {
            break;
        }
        pb_index_heap_swap(heap, position, best);
        position = best;
    }
}

static int pb_index_heap_init(pb_index_heap *heap, size_t capacity) {
    size_t index;
    heap->ids = (uint32_t *)pb_calloc(capacity == 0u ? 1u : capacity, sizeof(*heap->ids));
    heap->positions = (uint32_t *)pb_calloc(capacity == 0u ? 1u : capacity, sizeof(*heap->positions));
    if (heap->ids == NULL || heap->positions == NULL) {
        free(heap->ids);
        free(heap->positions);
        memset(heap, 0, sizeof(*heap));
        return 0;
    }
    heap->cap = capacity;
    for (index = 0u; index < capacity; index++) {
        heap->positions[index] = PB_NONE;
    }
    return 1;
}

static int pb_index_heap_contains(const pb_index_heap *heap, uint32_t id) {
    return (size_t)id < heap->cap && heap->positions[id] != PB_NONE;
}

static void pb_index_heap_insert(pb_engine *engine, uint32_t id) {
    pb_index_heap *heap = &engine->current_heap;
    if (pb_index_heap_contains(heap, id) || heap->len >= heap->cap) {
        return;
    }
    heap->ids[heap->len] = id;
    heap->positions[id] = (uint32_t)heap->len;
    heap->len++;
    pb_index_heap_up(engine, heap->len - 1u);
}

static uint32_t pb_index_heap_remove_at(pb_engine *engine, size_t position) {
    pb_index_heap *heap = &engine->current_heap;
    uint32_t removed = heap->ids[position];
    uint32_t replacement = heap->ids[heap->len - 1u];
    heap->len--;
    heap->positions[removed] = PB_NONE;
    if (position < heap->len) {
        heap->ids[position] = replacement;
        heap->positions[replacement] = (uint32_t)position;
        if (position > 0u && pb_dynamic_better(engine, replacement, heap->ids[(position - 1u) / 2u])) {
            pb_index_heap_up(engine, position);
        } else {
            pb_index_heap_down(engine, position);
        }
    }
    return removed;
}

static void pb_index_heap_remove(pb_engine *engine, uint32_t id) {
    pb_index_heap *heap = &engine->current_heap;
    if (pb_index_heap_contains(heap, id)) {
        (void)pb_index_heap_remove_at(engine, (size_t)heap->positions[id]);
    }
}

static void pb_index_heap_update(pb_engine *engine, uint32_t id) {
    pb_index_heap *heap = &engine->current_heap;
    size_t position;
    if (!pb_index_heap_contains(heap, id)) {
        return;
    }
    position = (size_t)heap->positions[id];
    if (position > 0u && pb_dynamic_better(engine, id, heap->ids[(position - 1u) / 2u])) {
        pb_index_heap_up(engine, position);
    } else {
        pb_index_heap_down(engine, position);
    }
}

static uint32_t pb_index_heap_pop(pb_engine *engine) {
    return pb_index_heap_remove_at(engine, 0u);
}

static void pb_index_heap_destroy(pb_index_heap *heap) {
    free(heap->ids);
    free(heap->positions);
    memset(heap, 0, sizeof(*heap));
}

static int pb_original_better(const pb_engine *engine, uint32_t left, uint32_t right) {
    double lw = engine->edges[left].weight;
    double rw = engine->edges[right].weight;
    if (lw > rw + PB_EPSILON) {
        return 1;
    }
    if (rw > lw + PB_EPSILON) {
        return 0;
    }
    return left < right;
}

static void pb_idheap_swap(pb_idheap *heap, size_t left, size_t right) {
    uint32_t value = heap->ids[left];
    heap->ids[left] = heap->ids[right];
    heap->ids[right] = value;
}

static void pb_idheap_down(const pb_engine *engine, pb_idheap *heap, size_t position) {
    for (;;) {
        size_t left = position * 2u + 1u;
        size_t right = left + 1u;
        size_t best = position;
        if (left < heap->len && pb_original_better(engine, heap->ids[left], heap->ids[best])) {
            best = left;
        }
        if (right < heap->len && pb_original_better(engine, heap->ids[right], heap->ids[best])) {
            best = right;
        }
        if (best == position) {
            break;
        }
        pb_idheap_swap(heap, position, best);
        position = best;
    }
}

static void pb_idheap_up(const pb_engine *engine, pb_idheap *heap, size_t position) {
    while (position > 0u) {
        size_t parent = (position - 1u) / 2u;
        if (!pb_original_better(engine, heap->ids[position], heap->ids[parent])) {
            break;
        }
        pb_idheap_swap(heap, position, parent);
        position = parent;
    }
}

static int pb_idheap_push(const pb_engine *engine, pb_idheap *heap, uint32_t id) {
    uint32_t *replacement;
    size_t capacity;
    if (heap->len == heap->cap) {
        capacity = heap->cap == 0u ? 16u : heap->cap * 2u;
        replacement = (uint32_t *)realloc(heap->ids, capacity * sizeof(*replacement));
        if (replacement == NULL) {
            return 0;
        }
        heap->ids = replacement;
        heap->cap = capacity;
    }
    heap->ids[heap->len++] = id;
    pb_idheap_up(engine, heap, heap->len - 1u);
    return 1;
}

static uint32_t pb_idheap_pop(const pb_engine *engine, pb_idheap *heap) {
    uint32_t result = heap->ids[0];
    heap->len--;
    if (heap->len > 0u) {
        heap->ids[0] = heap->ids[heap->len];
        pb_idheap_down(engine, heap, 0u);
    }
    return result;
}

static uint64_t pb_fnv1a_update(uint64_t hash, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    size_t index;
    for (index = 0u; index < size; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int pb_edge_compare(const void *left_ptr, const void *right_ptr) {
    const pb_edge *left = (const pb_edge *)left_ptr;
    const pb_edge *right = (const pb_edge *)right_ptr;
    if (left->u < right->u) {
        return -1;
    }
    if (left->u > right->u) {
        return 1;
    }
    if (left->v < right->v) {
        return -1;
    }
    if (left->v > right->v) {
        return 1;
    }
    return 0;
}

static int pb_temp_edge_compare_desc(const void *left_ptr, const void *right_ptr) {
    const pb_temp_edge *left = (const pb_temp_edge *)left_ptr;
    const pb_temp_edge *right = (const pb_temp_edge *)right_ptr;
    if (left->weight > right->weight + PB_EPSILON) {
        return -1;
    }
    if (right->weight > left->weight + PB_EPSILON) {
        return 1;
    }
    if (left->u != right->u) {
        return left->u < right->u ? -1 : 1;
    }
    if (left->v != right->v) {
        return left->v < right->v ? -1 : 1;
    }
    return 0;
}

void pb_config_init(pb_config *config) {
    if (config == NULL) {
        return;
    }
    memset(config, 0, sizeof(*config));
    config->struct_size = (uint32_t)sizeof(*config);
    config->abi_version = PB_ABI_VERSION;
    config->method = PB_METHOD_PERBACCO;
    config->cda = PB_CDA_LOUVAIN;
    config->batch_size = 10u;
    config->top_k = PB_DEFAULT_TOP_K;
    config->seed = 42u;
    config->lambda_w = 0.05;
    /* Zero selects the paper defaults: 0.75 up to 20k records, 0.50 above. */
    config->louvain_resolution = 0.0;
    config->louvain_threshold = 1e-4;
}

const char *pb_strerror(pb_status status) {
    switch (status) {
        case PB_OK: return "success";
        case PB_DONE: return "no inferable candidate edges remain";
        case PB_BUDGET_EXHAUSTED: return "query budget exhausted";
        case PB_EINVAL: return "invalid argument";
        case PB_ENOMEM: return "out of memory";
        case PB_ESTATE: return "invalid engine state";
        case PB_ECONFLICT: return "oracle partition conflicts with known facts";
        case PB_EOVERFLOW: return "numeric or capacity overflow";
        case PB_EVERSION: return "ABI or snapshot version mismatch";
        case PB_EINTERNAL: return "internal invariant failure";
        default: return "unknown pERbacco status";
    }
}

const char *pb_engine_last_error(const pb_engine *engine) {
    if (engine == NULL || engine->last_error[0] == '\0') {
        return "";
    }
    return engine->last_error;
}

typedef struct pb_nodegroup {
    uint32_t *nodes;
    size_t len;
    unsigned depth;
} pb_nodegroup;

static int pb_choose_tie(pb_engine *engine, size_t tie_count) {
    return pb_rng_bounded(engine, tie_count) == 0u;
}

static pb_status pb_greedy_hs(
    pb_engine *engine,
    const uint32_t *candidates,
    size_t candidate_count,
    const pb_temp_edge *edges,
    size_t edge_count,
    uint32_t *out,
    size_t *out_count) {
    double *incident_sum = NULL;
    double *gain = NULL;
    uint8_t *is_candidate = NULL;
    uint8_t *selected = NULL;
    size_t target;
    size_t index;
    size_t chosen = 0u;
    uint32_t first = PB_NONE;
    uint32_t second = PB_NONE;
    double best;
    size_t ties;

    if (candidate_count == 0u) {
        *out_count = 0u;
        return PB_OK;
    }
    target = candidate_count < (size_t)engine->config.batch_size
                 ? candidate_count
                 : (size_t)engine->config.batch_size;
    if (candidate_count <= target) {
        memcpy(out, candidates, candidate_count * sizeof(*out));
        *out_count = candidate_count;
        return PB_OK;
    }
    if (target == 2u && edge_count > 0u) {
        size_t best_edge = 0u;
        size_t tie_count = 1u;
        for (index = 1u; index < edge_count; index++) {
            if (edges[index].weight > edges[best_edge].weight + PB_EPSILON) {
                best_edge = index;
                tie_count = 1u;
            } else if (fabs(edges[index].weight - edges[best_edge].weight) <= PB_EPSILON) {
                tie_count++;
                if (pb_choose_tie(engine, tie_count)) {
                    best_edge = index;
                }
            }
        }
        out[0] = edges[best_edge].u;
        out[1] = edges[best_edge].v;
        *out_count = 2u;
        return PB_OK;
    }

    incident_sum = (double *)pb_calloc(engine->node_count, sizeof(*incident_sum));
    gain = (double *)pb_calloc(engine->node_count, sizeof(*gain));
    is_candidate = (uint8_t *)pb_calloc(engine->node_count, sizeof(*is_candidate));
    selected = (uint8_t *)pb_calloc(engine->node_count, sizeof(*selected));
    if (incident_sum == NULL || gain == NULL || is_candidate == NULL || selected == NULL) {
        free(incident_sum);
        free(gain);
        free(is_candidate);
        free(selected);
        return PB_ENOMEM;
    }
    for (index = 0u; index < candidate_count; index++) {
        is_candidate[candidates[index]] = 1u;
    }
    for (index = 0u; index < edge_count; index++) {
        if (is_candidate[edges[index].u] && is_candidate[edges[index].v]) {
            incident_sum[edges[index].u] += edges[index].weight;
            incident_sum[edges[index].v] += edges[index].weight;
        }
    }

    best = -DBL_MAX;
    ties = 0u;
    for (index = 0u; index < candidate_count; index++) {
        uint32_t node = candidates[index];
        if (incident_sum[node] > best + PB_EPSILON) {
            best = incident_sum[node];
            first = node;
            ties = 1u;
        } else if (fabs(incident_sum[node] - best) <= PB_EPSILON) {
            ties++;
            if (pb_choose_tie(engine, ties)) {
                first = node;
            }
        }
    }
    out[chosen++] = first;
    selected[first] = 1u;

    best = -DBL_MAX;
    ties = 0u;
    for (index = 0u; index < edge_count; index++) {
        uint32_t other = PB_NONE;
        if (edges[index].u == first && is_candidate[edges[index].v]) {
            other = edges[index].v;
        } else if (edges[index].v == first && is_candidate[edges[index].u]) {
            other = edges[index].u;
        }
        if (other != PB_NONE && !selected[other]) {
            if (edges[index].weight > best + PB_EPSILON) {
                best = edges[index].weight;
                second = other;
                ties = 1u;
            } else if (fabs(edges[index].weight - best) <= PB_EPSILON) {
                ties++;
                if (pb_choose_tie(engine, ties)) {
                    second = other;
                }
            }
        }
    }
    if (second == PB_NONE) {
        ties = 0u;
        for (index = 0u; index < candidate_count; index++) {
            uint32_t node = candidates[index];
            if (!selected[node]) {
                ties++;
                if (second == PB_NONE || pb_choose_tie(engine, ties)) {
                    second = node;
                }
            }
        }
    }
    if (second != PB_NONE && chosen < target) {
        out[chosen++] = second;
        selected[second] = 1u;
    }

    while (chosen < target) {
        uint32_t best_vertex = PB_NONE;
        uint32_t edge_u = PB_NONE;
        uint32_t edge_v = PB_NONE;
        double vertex_gain = -DBL_MAX;
        double edge_weight = -DBL_MAX;
        size_t vertex_ties = 0u;
        size_t edge_ties = 0u;

        memset(gain, 0, engine->node_count * sizeof(*gain));
        for (index = 0u; index < edge_count; index++) {
            uint32_t u = edges[index].u;
            uint32_t v = edges[index].v;
            if (selected[u] && is_candidate[v] && !selected[v]) {
                gain[v] += edges[index].weight;
            } else if (selected[v] && is_candidate[u] && !selected[u]) {
                gain[u] += edges[index].weight;
            }
            if (is_candidate[u] && is_candidate[v] && !selected[u] && !selected[v]) {
                if (edges[index].weight > edge_weight + PB_EPSILON) {
                    edge_weight = edges[index].weight;
                    edge_u = u;
                    edge_v = v;
                    edge_ties = 1u;
                } else if (fabs(edges[index].weight - edge_weight) <= PB_EPSILON) {
                    edge_ties++;
                    if (pb_choose_tie(engine, edge_ties)) {
                        edge_u = u;
                        edge_v = v;
                    }
                }
            }
        }
        for (index = 0u; index < candidate_count; index++) {
            uint32_t node = candidates[index];
            if (selected[node]) {
                continue;
            }
            if (gain[node] > vertex_gain + PB_EPSILON) {
                vertex_gain = gain[node];
                best_vertex = node;
                vertex_ties = 1u;
            } else if (fabs(gain[node] - vertex_gain) <= PB_EPSILON) {
                vertex_ties++;
                if (pb_choose_tie(engine, vertex_ties)) {
                    best_vertex = node;
                }
            }
        }
        if (edge_u != PB_NONE && chosen + 2u <= target && edge_weight > vertex_gain + PB_EPSILON) {
            out[chosen++] = edge_u;
            out[chosen++] = edge_v;
            selected[edge_u] = 1u;
            selected[edge_v] = 1u;
        } else if (best_vertex != PB_NONE) {
            out[chosen++] = best_vertex;
            selected[best_vertex] = 1u;
        } else {
            break;
        }
    }

    free(incident_sum);
    free(gain);
    free(is_candidate);
    free(selected);
    *out_count = chosen;
    return PB_OK;
}

typedef struct pb_louvain_graph {
    size_t node_count;
    size_t edge_count;
    pb_temp_edge *edges;
    pb_u32vec *adjacency;
    double *degree;
    double total_degree;
} pb_louvain_graph;

static void pb_louvain_graph_destroy(pb_louvain_graph *graph) {
    size_t index;
    if (graph->adjacency != NULL) {
        for (index = 0u; index < graph->node_count; index++) {
            free(graph->adjacency[index].items);
        }
    }
    free(graph->adjacency);
    free(graph->degree);
    free(graph->edges);
    memset(graph, 0, sizeof(*graph));
}

static pb_status pb_louvain_graph_index(pb_louvain_graph *graph) {
    size_t index;
    graph->adjacency = (pb_u32vec *)pb_calloc(
        graph->node_count == 0u ? 1u : graph->node_count, sizeof(*graph->adjacency));
    graph->degree = (double *)pb_calloc(
        graph->node_count == 0u ? 1u : graph->node_count, sizeof(*graph->degree));
    if (graph->adjacency == NULL || graph->degree == NULL) {
        return PB_ENOMEM;
    }
    for (index = 0u; index < graph->edge_count; index++) {
        const pb_temp_edge *edge = &graph->edges[index];
        graph->total_degree += 2.0 * edge->weight;
        if (edge->u == edge->v) {
            graph->degree[edge->u] += 2.0 * edge->weight;
        } else {
            graph->degree[edge->u] += edge->weight;
            graph->degree[edge->v] += edge->weight;
            if (!pb_vec_push(&graph->adjacency[edge->u], (uint32_t)index) ||
                !pb_vec_push(&graph->adjacency[edge->v], (uint32_t)index)) {
                return PB_ENOMEM;
            }
        }
    }
    return PB_OK;
}

static pb_status pb_louvain_initial_graph(
    const pb_engine *engine,
    const uint32_t *global_nodes,
    size_t node_count,
    pb_louvain_graph *out_graph) {
    uint32_t *local_of = NULL;
    size_t edge_count = 0u;
    size_t index;
    pb_status status = PB_OK;

    local_of = (uint32_t *)pb_calloc(engine->node_count, sizeof(*local_of));
    if (local_of == NULL) {
        return PB_ENOMEM;
    }
    for (index = 0u; index < (size_t)engine->node_count; index++) {
        local_of[index] = PB_NONE;
    }
    for (index = 0u; index < node_count; index++) {
        local_of[global_nodes[index]] = (uint32_t)index;
    }
    for (index = 0u; index < node_count; index++) {
        uint32_t global = global_nodes[index];
        size_t adjacency_index;
        for (adjacency_index = 0u; adjacency_index < engine->incident[global].len;
             adjacency_index++) {
            const pb_edge *edge = &engine->edges[engine->incident[global].items[adjacency_index]];
            if (edge->u == global && local_of[edge->v] != PB_NONE) {
                edge_count++;
            }
        }
    }
    if (edge_count > (size_t)UINT32_MAX) {
        status = PB_EOVERFLOW;
        goto cleanup;
    }
    out_graph->node_count = node_count;
    out_graph->edge_count = edge_count;
    out_graph->edges = (pb_temp_edge *)pb_calloc(
        edge_count == 0u ? 1u : edge_count, sizeof(*out_graph->edges));
    if (out_graph->edges == NULL) {
        status = PB_ENOMEM;
        goto cleanup;
    }
    edge_count = 0u;
    for (index = 0u; index < node_count; index++) {
        uint32_t global = global_nodes[index];
        size_t adjacency_index;
        for (adjacency_index = 0u; adjacency_index < engine->incident[global].len;
             adjacency_index++) {
            const pb_edge *edge = &engine->edges[engine->incident[global].items[adjacency_index]];
            uint32_t local_other;
            if (edge->u != global) {
                continue;
            }
            local_other = local_of[edge->v];
            if (local_other == PB_NONE) {
                continue;
            }
            out_graph->edges[edge_count].u = (uint32_t)index;
            out_graph->edges[edge_count].v = local_other;
            out_graph->edges[edge_count].weight = edge->weight;
            edge_count++;
        }
    }
    status = pb_louvain_graph_index(out_graph);

cleanup:
    free(local_of);
    if (status != PB_OK) {
        pb_louvain_graph_destroy(out_graph);
    }
    return status;
}

static pb_status pb_louvain_aggregate(
    const pb_louvain_graph *graph,
    const uint32_t *labels,
    size_t community_count,
    pb_louvain_graph *out_graph) {
    pb_pairmap edge_map = {0};
    size_t result_count = 0u;
    size_t index;
    pb_status status = PB_OK;

    if (!pb_pairmap_init(&edge_map, graph->edge_count)) {
        return PB_ENOMEM;
    }
    out_graph->node_count = community_count;
    out_graph->edges = (pb_temp_edge *)pb_calloc(
        graph->edge_count == 0u ? 1u : graph->edge_count, sizeof(*out_graph->edges));
    if (out_graph->edges == NULL) {
        status = PB_ENOMEM;
        goto cleanup;
    }
    for (index = 0u; index < graph->edge_count; index++) {
        const pb_temp_edge *edge = &graph->edges[index];
        uint32_t left = labels[edge->u];
        uint32_t right = labels[edge->v];
        uint64_t key = pb_pair_key(left, right);
        uint32_t existing;
        if (pb_pairmap_get(&edge_map, key, &existing)) {
            out_graph->edges[existing].weight += edge->weight;
        } else {
            if (result_count > (size_t)UINT32_MAX) {
                status = PB_EOVERFLOW;
                goto cleanup;
            }
            out_graph->edges[result_count].u = left < right ? left : right;
            out_graph->edges[result_count].v = left < right ? right : left;
            out_graph->edges[result_count].weight = edge->weight;
            if (!pb_pairmap_put(&edge_map, key, (uint32_t)result_count)) {
                status = PB_ENOMEM;
                goto cleanup;
            }
            result_count++;
        }
    }
    out_graph->edge_count = result_count;
    status = pb_louvain_graph_index(out_graph);

cleanup:
    pb_pairmap_destroy(&edge_map);
    if (status != PB_OK) {
        pb_louvain_graph_destroy(out_graph);
    }
    return status;
}

static double pb_louvain_modularity(
    const pb_louvain_graph *graph,
    const uint32_t *labels,
    size_t community_count,
    double resolution) {
    double *internal = NULL;
    double *total = NULL;
    double result = 0.0;
    double total_weight = graph->total_degree / 2.0;
    size_t index;

    if (total_weight <= 0.0) {
        return 0.0;
    }
    internal = (double *)pb_calloc(community_count == 0u ? 1u : community_count,
                                   sizeof(*internal));
    total = (double *)pb_calloc(community_count == 0u ? 1u : community_count,
                                sizeof(*total));
    if (internal == NULL || total == NULL) {
        free(internal);
        free(total);
        return -DBL_MAX;
    }
    for (index = 0u; index < graph->node_count; index++) {
        total[labels[index]] += graph->degree[index];
    }
    for (index = 0u; index < graph->edge_count; index++) {
        const pb_temp_edge *edge = &graph->edges[index];
        if (labels[edge->u] == labels[edge->v]) {
            internal[labels[edge->u]] += edge->weight;
        }
    }
    for (index = 0u; index < community_count; index++) {
        double fraction = total[index] / graph->total_degree;
        result += internal[index] / total_weight - resolution * fraction * fraction;
    }
    free(internal);
    free(total);
    return result;
}

static pb_status pb_louvain_one_level(
    pb_engine *engine,
    const pb_louvain_graph *graph,
    double resolution,
    uint32_t *out_labels,
    size_t *out_community_count,
    int *out_moved) {
    uint32_t *labels = NULL;
    uint32_t *order = NULL;
    uint32_t *label_map = NULL;
    uint32_t *touched = NULL;
    uint8_t *touched_flag = NULL;
    double *community_total = NULL;
    double *weights_to = NULL;
    size_t index;
    unsigned pass;
    int any_moved = 0;
    pb_status status = PB_OK;

    labels = (uint32_t *)pb_calloc(graph->node_count, sizeof(*labels));
    order = (uint32_t *)pb_calloc(graph->node_count, sizeof(*order));
    label_map = (uint32_t *)pb_calloc(graph->node_count, sizeof(*label_map));
    touched = (uint32_t *)pb_calloc(graph->node_count, sizeof(*touched));
    touched_flag = (uint8_t *)pb_calloc(graph->node_count, sizeof(*touched_flag));
    community_total = (double *)pb_calloc(graph->node_count, sizeof(*community_total));
    weights_to = (double *)pb_calloc(graph->node_count, sizeof(*weights_to));
    if (labels == NULL || order == NULL || label_map == NULL || touched == NULL ||
        touched_flag == NULL || community_total == NULL || weights_to == NULL) {
        status = PB_ENOMEM;
        goto cleanup;
    }
    for (index = 0u; index < graph->node_count; index++) {
        labels[index] = (uint32_t)index;
        order[index] = (uint32_t)index;
        label_map[index] = PB_NONE;
        community_total[index] = graph->degree[index];
    }
    /* NetworkX shuffles once per level and reuses that order for all sweeps. */
    for (index = graph->node_count; index > 1u; index--) {
        size_t other = pb_rng_bounded(engine, index);
        uint32_t value = order[index - 1u];
        order[index - 1u] = order[other];
        order[other] = value;
    }
    for (pass = 0u; pass < 100u; pass++) {
        size_t order_index;
        size_t moves = 0u;
        for (order_index = 0u; order_index < graph->node_count; order_index++) {
            uint32_t node = order[order_index];
            uint32_t old_label = labels[node];
            uint32_t best_label = old_label;
            double remove_cost;
            double best_gain = 0.0;
            size_t touched_count = 0u;
            size_t adjacency_index;

            for (adjacency_index = 0u; adjacency_index < graph->adjacency[node].len;
                 adjacency_index++) {
                const pb_temp_edge *edge =
                    &graph->edges[graph->adjacency[node].items[adjacency_index]];
                uint32_t neighbor = edge->u == node ? edge->v : edge->u;
                uint32_t neighbor_label = labels[neighbor];
                if (!touched_flag[neighbor_label]) {
                    touched_flag[neighbor_label] = 1u;
                    touched[touched_count++] = neighbor_label;
                }
                weights_to[neighbor_label] += edge->weight;
            }
            if (!touched_flag[old_label]) {
                touched_flag[old_label] = 1u;
                touched[touched_count++] = old_label;
            }
            community_total[old_label] -= graph->degree[node];
            remove_cost = -weights_to[old_label] +
                          resolution * community_total[old_label] * graph->degree[node] /
                              graph->total_degree;
            for (index = 0u; index < touched_count; index++) {
                uint32_t candidate = touched[index];
                double gain = remove_cost + weights_to[candidate] -
                              resolution * community_total[candidate] * graph->degree[node] /
                                  graph->total_degree;
                if (gain > best_gain + PB_EPSILON) {
                    best_gain = gain;
                    best_label = candidate;
                }
            }
            labels[node] = best_label;
            community_total[best_label] += graph->degree[node];
            if (best_label != old_label) {
                moves++;
                any_moved = 1;
            }
            for (index = 0u; index < touched_count; index++) {
                uint32_t label = touched[index];
                weights_to[label] = 0.0;
                touched_flag[label] = 0u;
            }
        }
        if (moves == 0u) {
            break;
        }
    }
    {
        size_t count = 0u;
        for (index = 0u; index < graph->node_count; index++) {
            uint32_t label = labels[index];
            if (label_map[label] == PB_NONE) {
                label_map[label] = (uint32_t)count++;
            }
            out_labels[index] = label_map[label];
        }
        *out_community_count = count;
    }
    *out_moved = any_moved;

cleanup:
    free(labels);
    free(order);
    free(label_map);
    free(touched);
    free(touched_flag);
    free(community_total);
    free(weights_to);
    return status;
}

static pb_status pb_louvain_partition(
    pb_engine *engine,
    const uint32_t *global_nodes,
    size_t node_count,
    uint32_t *out_labels,
    size_t *out_community_count) {
    pb_louvain_graph graph = {0};
    uint32_t *original_labels = NULL;
    uint32_t *level_labels = NULL;
    double resolution = engine->config.louvain_resolution;
    double previous_modularity;
    size_t community_count;
    size_t index;
    unsigned level;
    pb_status status;

    if (node_count == 0u) {
        *out_community_count = 0u;
        return PB_OK;
    }
    if (node_count == 1u) {
        out_labels[0] = 0u;
        *out_community_count = 1u;
        return PB_OK;
    }
    if (resolution <= 0.0) {
        resolution = engine->node_count > 20000u ? 0.50 : 0.75;
    }
    status = pb_louvain_initial_graph(engine, global_nodes, node_count, &graph);
    if (status != PB_OK) {
        return status;
    }
    original_labels = (uint32_t *)pb_calloc(node_count, sizeof(*original_labels));
    level_labels = (uint32_t *)pb_calloc(node_count, sizeof(*level_labels));
    if (original_labels == NULL || level_labels == NULL) {
        status = PB_ENOMEM;
        goto cleanup;
    }
    for (index = 0u; index < node_count; index++) {
        original_labels[index] = (uint32_t)index;
        level_labels[index] = (uint32_t)index;
    }
    community_count = node_count;
    if (graph.edge_count == 0u || graph.total_degree <= 0.0) {
        memcpy(out_labels, original_labels, node_count * sizeof(*out_labels));
        *out_community_count = community_count;
        status = PB_OK;
        goto cleanup;
    }
    previous_modularity = pb_louvain_modularity(
        &graph, level_labels, graph.node_count, resolution);
    if (previous_modularity == -DBL_MAX) {
        status = PB_ENOMEM;
        goto cleanup;
    }
    for (level = 0u; level < 100u; level++) {
        pb_louvain_graph aggregate = {0};
        double modularity;
        int moved = 0;

        status = pb_louvain_one_level(
            engine, &graph, resolution, level_labels, &community_count, &moved);
        if (status != PB_OK) {
            goto cleanup;
        }
        modularity = pb_louvain_modularity(
            &graph, level_labels, community_count, resolution);
        if (modularity == -DBL_MAX) {
            status = PB_ENOMEM;
            goto cleanup;
        }
        for (index = 0u; index < node_count; index++) {
            original_labels[index] = level_labels[original_labels[index]];
        }
        if (!moved || community_count == graph.node_count ||
            modularity - previous_modularity <= engine->config.louvain_threshold) {
            break;
        }
        previous_modularity = modularity;
        status = pb_louvain_aggregate(&graph, level_labels, community_count, &aggregate);
        if (status != PB_OK) {
            goto cleanup;
        }
        pb_louvain_graph_destroy(&graph);
        graph = aggregate;
    }
    memcpy(out_labels, original_labels, node_count * sizeof(*out_labels));
    *out_community_count = community_count;
    status = PB_OK;

cleanup:
    free(original_labels);
    free(level_labels);
    pb_louvain_graph_destroy(&graph);
    return status;
}

static double pb_group_weight(
    const pb_engine *engine,
    const uint32_t *nodes,
    size_t node_count,
    uint8_t *marks) {
    size_t index;
    double weight = 0.0;
    for (index = 0u; index < node_count; index++) {
        marks[nodes[index]] = 1u;
    }
    for (index = 0u; index < node_count; index++) {
        uint32_t node = nodes[index];
        size_t adjacency_index;
        for (adjacency_index = 0u; adjacency_index < engine->incident[node].len; adjacency_index++) {
            uint32_t edge_id = engine->incident[node].items[adjacency_index];
            const pb_edge *edge = &engine->edges[edge_id];
            if (edge->u == node && marks[edge->v]) {
                weight += edge->weight;
            }
        }
    }
    for (index = 0u; index < node_count; index++) {
        marks[nodes[index]] = 0u;
    }
    return weight;
}

static void pb_free_nodegroups(pb_nodegroup *groups, size_t count) {
    size_t index;
    for (index = 0u; index < count; index++) {
        free(groups[index].nodes);
    }
    free(groups);
}

static pb_status pb_build_louvain_communities(
    pb_engine *engine,
    pb_nodegroup **out_groups,
    size_t *out_count) {
    pb_nodegroup *queue = NULL;
    size_t queue_len = 0u;
    size_t queue_cap = 0u;
    size_t queue_pos = 0u;
    pb_nodegroup *heavy = NULL;
    size_t heavy_len = 0u;
    size_t heavy_cap = 0u;
    uint8_t *marks = NULL;
    size_t index;
    pb_status status = PB_OK;
    size_t min_size = engine->config.batch_size > 10u ? engine->config.batch_size : 10u;

    marks = (uint8_t *)pb_calloc(engine->node_count, sizeof(*marks));
    queue = (pb_nodegroup *)pb_calloc(4u, sizeof(*queue));
    if (marks == NULL || queue == NULL) {
        free(marks);
        free(queue);
        return PB_ENOMEM;
    }
    queue_cap = 4u;
    queue[0].nodes = (uint32_t *)pb_calloc(engine->node_count, sizeof(*queue[0].nodes));
    if (queue[0].nodes == NULL) {
        free(marks);
        free(queue);
        return PB_ENOMEM;
    }
    queue[0].len = engine->node_count;
    for (index = 0u; index < (size_t)engine->node_count; index++) {
        queue[0].nodes[index] = (uint32_t)index;
    }
    queue_len = 1u;

    while (queue_pos < queue_len) {
        pb_nodegroup group = queue[queue_pos++];
        uint32_t *labels = NULL;
        size_t community_count = 0u;
        pb_nodegroup *parts = NULL;
        size_t *part_sizes = NULL;
        size_t part_index;

        if (group.len < 2u || group.depth > 64u) {
            free(group.nodes);
            queue[queue_pos - 1u].nodes = NULL;
            continue;
        }
        labels = (uint32_t *)pb_calloc(group.len, sizeof(*labels));
        if (labels == NULL) {
            status = PB_ENOMEM;
            free(group.nodes);
            queue[queue_pos - 1u].nodes = NULL;
            goto cleanup;
        }
        status = pb_louvain_partition(engine, group.nodes, group.len, labels, &community_count);
        if (status != PB_OK) {
            free(labels);
            free(group.nodes);
            queue[queue_pos - 1u].nodes = NULL;
            goto cleanup;
        }
        if (community_count <= 1u) {
            free(labels);
            free(group.nodes);
            queue[queue_pos - 1u].nodes = NULL;
            continue;
        }
        parts = (pb_nodegroup *)pb_calloc(community_count, sizeof(*parts));
        part_sizes = (size_t *)pb_calloc(community_count, sizeof(*part_sizes));
        if (parts == NULL || part_sizes == NULL) {
            free(parts);
            free(part_sizes);
            free(labels);
            free(group.nodes);
            queue[queue_pos - 1u].nodes = NULL;
            status = PB_ENOMEM;
            goto cleanup;
        }
        for (index = 0u; index < group.len; index++) {
            part_sizes[labels[index]]++;
        }
        for (part_index = 0u; part_index < community_count; part_index++) {
            parts[part_index].nodes = (uint32_t *)pb_calloc(
                part_sizes[part_index] == 0u ? 1u : part_sizes[part_index], sizeof(uint32_t));
            if (parts[part_index].nodes == NULL) {
                status = PB_ENOMEM;
                goto part_cleanup;
            }
            parts[part_index].depth = group.depth + 1u;
        }
        memset(part_sizes, 0, community_count * sizeof(*part_sizes));
        for (index = 0u; index < group.len; index++) {
            uint32_t label = labels[index];
            parts[label].nodes[part_sizes[label]++] = group.nodes[index];
            parts[label].len++;
        }
        for (part_index = 0u; part_index < community_count; part_index++) {
            pb_nodegroup part = parts[part_index];
            double weight = pb_group_weight(engine, part.nodes, part.len, marks);
            double denominator = part.len < 2u ? 1.0 : (double)part.len * (double)(part.len - 1u) / 2.0;
            double density = weight / denominator;
            if (part.len >= min_size && density + PB_EPSILON >= engine->config.lambda_w) {
                if (heavy_len == heavy_cap) {
                    size_t replacement_cap = heavy_cap == 0u ? 8u : heavy_cap * 2u;
                    pb_nodegroup *replacement = (pb_nodegroup *)realloc(
                        heavy, replacement_cap * sizeof(*replacement));
                    if (replacement == NULL) {
                        status = PB_ENOMEM;
                        goto part_cleanup;
                    }
                    heavy = replacement;
                    heavy_cap = replacement_cap;
                }
                heavy[heavy_len++] = part;
                parts[part_index].nodes = NULL;
            } else if (density + PB_EPSILON < engine->config.lambda_w && part.len >= min_size) {
                if (queue_len == queue_cap) {
                    size_t replacement_cap = queue_cap * 2u;
                    pb_nodegroup *replacement = (pb_nodegroup *)realloc(
                        queue, replacement_cap * sizeof(*replacement));
                    if (replacement == NULL) {
                        status = PB_ENOMEM;
                        goto part_cleanup;
                    }
                    queue = replacement;
                    memset(queue + queue_cap, 0, (replacement_cap - queue_cap) * sizeof(*queue));
                    queue_cap = replacement_cap;
                }
                queue[queue_len++] = part;
                parts[part_index].nodes = NULL;
            }
        }

part_cleanup:
        for (part_index = 0u; part_index < community_count; part_index++) {
            free(parts[part_index].nodes);
        }
        free(parts);
        free(part_sizes);
        free(labels);
        free(group.nodes);
        queue[queue_pos - 1u].nodes = NULL;
        if (status != PB_OK) {
            goto cleanup;
        }
    }

    /* Stable insertion sort by descending internal weight. */
    for (index = 1u; index < heavy_len; index++) {
        pb_nodegroup item = heavy[index];
        double item_weight = pb_group_weight(engine, item.nodes, item.len, marks);
        size_t position = index;
        while (position > 0u) {
            double previous_weight = pb_group_weight(
                engine, heavy[position - 1u].nodes, heavy[position - 1u].len, marks);
            if (previous_weight >= item_weight - PB_EPSILON) {
                break;
            }
            heavy[position] = heavy[position - 1u];
            position--;
        }
        heavy[position] = item;
    }
    *out_groups = heavy;
    *out_count = heavy_len;
    heavy = NULL;
    heavy_len = 0u;

cleanup:
    pb_free_nodegroups(queue, queue_len);
    pb_free_nodegroups(heavy, heavy_len);
    free(marks);
    return status;
}

typedef struct pb_truth_root {
    uint32_t label;
    uint32_t root;
    uint32_t size;
} pb_truth_root;

typedef struct pb_truth_group {
    size_t start;
    size_t len;
    uint32_t label;
    uint32_t total_size;
    double factor;
} pb_truth_group;

typedef struct pb_truth_rank {
    size_t group;
    uint32_t label;
    uint32_t total_size;
} pb_truth_rank;

typedef struct pb_truth_cursor {
    size_t left;
    size_t right;
    size_t end;
    double factor;
} pb_truth_cursor;

static int pb_truth_root_compare(const void *left_ptr, const void *right_ptr) {
    const pb_truth_root *left = (const pb_truth_root *)left_ptr;
    const pb_truth_root *right = (const pb_truth_root *)right_ptr;
    if (left->label != right->label) {
        return left->label < right->label ? -1 : 1;
    }
    if (left->size != right->size) {
        return left->size > right->size ? -1 : 1;
    }
    if (left->root != right->root) {
        return left->root < right->root ? -1 : 1;
    }
    return 0;
}

static int pb_truth_rank_compare(const void *left_ptr, const void *right_ptr) {
    const pb_truth_rank *left = (const pb_truth_rank *)left_ptr;
    const pb_truth_rank *right = (const pb_truth_rank *)right_ptr;
    if (left->total_size != right->total_size) {
        return left->total_size > right->total_size ? -1 : 1;
    }
    if (left->label != right->label) {
        return left->label < right->label ? -1 : 1;
    }
    return 0;
}

static double pb_truth_cursor_weight(
    const pb_truth_root *roots,
    const pb_truth_cursor *cursor) {
    return cursor->factor * (double)roots[cursor->left].size *
           (double)roots[cursor->right].size;
}

static int pb_truth_cursor_better(
    const pb_truth_root *roots,
    const pb_truth_cursor *cursors,
    size_t left_id,
    size_t right_id) {
    const pb_truth_cursor *left = &cursors[left_id];
    const pb_truth_cursor *right = &cursors[right_id];
    double left_weight = pb_truth_cursor_weight(roots, left);
    double right_weight = pb_truth_cursor_weight(roots, right);
    uint32_t left_u;
    uint32_t left_v;
    uint32_t right_u;
    uint32_t right_v;
    if (left_weight > right_weight + PB_EPSILON) {
        return 1;
    }
    if (right_weight > left_weight + PB_EPSILON) {
        return 0;
    }
    left_u = roots[left->left].root;
    left_v = roots[left->right].root;
    right_u = roots[right->left].root;
    right_v = roots[right->right].root;
    if (left_u != right_u) {
        return left_u < right_u;
    }
    return left_v < right_v;
}

static void pb_truth_heap_push(
    const pb_truth_root *roots,
    const pb_truth_cursor *cursors,
    size_t *heap,
    size_t *heap_len,
    size_t cursor_id) {
    size_t position = *heap_len;
    heap[position] = cursor_id;
    (*heap_len)++;
    while (position > 0u) {
        size_t parent = (position - 1u) / 2u;
        size_t temporary;
        if (!pb_truth_cursor_better(roots, cursors, heap[position], heap[parent])) {
            break;
        }
        temporary = heap[position];
        heap[position] = heap[parent];
        heap[parent] = temporary;
        position = parent;
    }
}

static size_t pb_truth_heap_pop(
    const pb_truth_root *roots,
    const pb_truth_cursor *cursors,
    size_t *heap,
    size_t *heap_len) {
    size_t result = heap[0];
    size_t position = 0u;
    (*heap_len)--;
    if (*heap_len == 0u) {
        return result;
    }
    heap[0] = heap[*heap_len];
    for (;;) {
        size_t left = position * 2u + 1u;
        size_t right = left + 1u;
        size_t best = position;
        size_t temporary;
        if (left < *heap_len &&
            pb_truth_cursor_better(roots, cursors, heap[left], heap[best])) {
            best = left;
        }
        if (right < *heap_len &&
            pb_truth_cursor_better(roots, cursors, heap[right], heap[best])) {
            best = right;
        }
        if (best == position) {
            break;
        }
        temporary = heap[position];
        heap[position] = heap[best];
        heap[best] = temporary;
        position = best;
    }
    return result;
}

/*
 * Paper SubOpt: build the hidden ground-truth benefit graph and apply
 * GreedyHS1000.  Each truth component is a clique whose tiny rank factor
 * reproduces the prototype's preference for larger entities.  The row cursors
 * lazily enumerate the globally heaviest edges, avoiding the quadratic graph.
 */
static pb_status pb_select_subopt(pb_engine *engine, uint32_t *records, size_t *out_len) {
    pb_truth_root *roots = NULL;
    pb_truth_group *groups = NULL;
    pb_truth_rank *ranks = NULL;
    pb_truth_cursor *cursors = NULL;
    size_t *heap = NULL;
    pb_temp_edge *edges = NULL;
    uint32_t *candidates = NULL;
    uint8_t *seen = NULL;
    size_t root_count = 0u;
    size_t group_count = 0u;
    size_t rank_count = 0u;
    size_t cursor_count = 0u;
    size_t heap_len = 0u;
    size_t edge_count = 0u;
    size_t candidate_count = 0u;
    size_t index;
    size_t edge_limit = (size_t)engine->config.top_k;
    uint32_t largest_entity = 0u;
    pb_status status = PB_OK;

    roots = (pb_truth_root *)pb_calloc(engine->node_count, sizeof(*roots));
    groups = (pb_truth_group *)pb_calloc(engine->node_count, sizeof(*groups));
    ranks = (pb_truth_rank *)pb_calloc(engine->node_count, sizeof(*ranks));
    cursors = (pb_truth_cursor *)pb_calloc(engine->node_count, sizeof(*cursors));
    heap = (size_t *)pb_calloc(engine->node_count, sizeof(*heap));
    edges = (pb_temp_edge *)pb_calloc(edge_limit, sizeof(*edges));
    candidates = (uint32_t *)pb_calloc(edge_limit * 2u, sizeof(*candidates));
    seen = (uint8_t *)pb_calloc(engine->node_count, sizeof(*seen));
    if (roots == NULL || groups == NULL || ranks == NULL || cursors == NULL || heap == NULL ||
        edges == NULL || candidates == NULL || seen == NULL) {
        status = PB_ENOMEM;
        goto cleanup;
    }
    for (index = 0u; index < (size_t)engine->node_count; index++) {
        if (engine->parent[index] == index) {
            roots[root_count].label = engine->truth_labels[index];
            roots[root_count].root = (uint32_t)index;
            roots[root_count].size = engine->component_size[index];
            root_count++;
        }
    }
    qsort(roots, root_count, sizeof(*roots), pb_truth_root_compare);
    index = 0u;
    while (index < root_count) {
        size_t end = index + 1u;
        uint64_t total_size = (uint64_t)roots[index].size;
        while (end < root_count && roots[end].label == roots[index].label) {
            total_size += (uint64_t)roots[end].size;
            end++;
        }
        if (total_size > UINT32_MAX) {
            status = PB_EOVERFLOW;
            goto cleanup;
        }
        groups[group_count].start = index;
        groups[group_count].len = end - index;
        groups[group_count].label = roots[index].label;
        groups[group_count].total_size = (uint32_t)total_size;
        if (groups[group_count].total_size >= 2u) {
            ranks[rank_count].group = group_count;
            ranks[rank_count].label = groups[group_count].label;
            ranks[rank_count].total_size = groups[group_count].total_size;
            rank_count++;
            if (groups[group_count].total_size > largest_entity) {
                largest_entity = groups[group_count].total_size;
            }
        }
        group_count++;
        index = end;
    }
    qsort(ranks, rank_count, sizeof(*ranks), pb_truth_rank_compare);
    if (rank_count > 0u) {
        double constant = ((double)engine->config.batch_size *
                           (double)(engine->config.batch_size - 1u) / 2.0) *
                          (double)largest_entity;
        for (index = 0u; index < rank_count; index++) {
            groups[ranks[index].group].factor =
                1.0 + 1.0 / (((double)index + 1.0) * constant);
        }
    }
    for (index = 0u; index < group_count; index++) {
        const pb_truth_group *group = &groups[index];
        size_t left;
        if (group->len < 2u) {
            continue;
        }
        for (left = group->start; left + 1u < group->start + group->len; left++) {
            cursors[cursor_count].left = left;
            cursors[cursor_count].right = left + 1u;
            cursors[cursor_count].end = group->start + group->len;
            cursors[cursor_count].factor = group->factor;
            pb_truth_heap_push(roots, cursors, heap, &heap_len, cursor_count);
            cursor_count++;
        }
    }
    while (edge_count < edge_limit && heap_len > 0u) {
        size_t cursor_id = pb_truth_heap_pop(roots, cursors, heap, &heap_len);
        pb_truth_cursor *cursor = &cursors[cursor_id];
        uint32_t u = roots[cursor->left].root;
        uint32_t v = roots[cursor->right].root;
        edges[edge_count].u = u;
        edges[edge_count].v = v;
        edges[edge_count].weight = pb_truth_cursor_weight(roots, cursor);
        edge_count++;
        if (!seen[u]) {
            seen[u] = 1u;
            candidates[candidate_count++] = u;
        }
        if (!seen[v]) {
            seen[v] = 1u;
            candidates[candidate_count++] = v;
        }
        cursor->right++;
        if (cursor->right < cursor->end) {
            pb_truth_heap_push(roots, cursors, heap, &heap_len, cursor_id);
        }
    }
    status = pb_greedy_hs(
        engine, candidates, candidate_count, edges, edge_count, records, out_len);

cleanup:
    free(roots);
    free(groups);
    free(ranks);
    free(cursors);
    free(heap);
    free(edges);
    free(candidates);
    free(seen);
    return status;
}

static pb_status pb_copy_and_validate_edges(
    pb_engine *engine,
    pb_edge *input_edges,
    size_t edge_count,
    int borrow_edges) {
    size_t index;
    double maximum = 0.0;
    size_t *degrees = NULL;
    size_t *cursor = NULL;
    uint64_t hash = UINT64_C(1469598103934665603);

    if (edge_count > (size_t)UINT32_MAX) {
        return PB_EOVERFLOW;
    }
    if (edge_count > 0u && input_edges == NULL) {
        return PB_EINVAL;
    }
    if (borrow_edges) {
        engine->edges = input_edges;
        engine->owns_edges = 0u;
    } else {
        engine->edges = (pb_edge *)pb_calloc(
            edge_count == 0u ? 1u : edge_count, sizeof(*engine->edges));
        if (engine->edges == NULL) {
            return PB_ENOMEM;
        }
        engine->owns_edges = 1u;
    }
    engine->edge_count = edge_count;
    for (index = 0u; index < edge_count; index++) {
        pb_edge edge = input_edges[index];
        if (edge.u >= engine->node_count || edge.v >= engine->node_count || edge.u == edge.v ||
            !isfinite(edge.weight) || edge.weight < 0.0) {
            return PB_EINVAL;
        }
        if (edge.v < edge.u) {
            uint32_t value = edge.u;
            edge.u = edge.v;
            edge.v = value;
        }
        engine->edges[index] = edge;
        if (edge.weight > maximum) {
            maximum = edge.weight;
        }
    }
    qsort(engine->edges, edge_count, sizeof(*engine->edges), pb_edge_compare);
    for (index = 1u; index < edge_count; index++) {
        if (engine->edges[index - 1u].u == engine->edges[index].u &&
            engine->edges[index - 1u].v == engine->edges[index].v) {
            return PB_EINVAL;
        }
    }
    if (maximum > 0.0) {
        for (index = 0u; index < edge_count; index++) {
            engine->edges[index].weight /= maximum;
        }
    }
    hash = pb_fnv1a_update(hash, &engine->node_count, sizeof(engine->node_count));
    hash = pb_fnv1a_update(hash, &edge_count, sizeof(edge_count));
    hash = pb_fnv1a_update(hash, engine->edges, edge_count * sizeof(*engine->edges));
    engine->graph_hash = hash;

    engine->incident = (pb_u32vec *)pb_calloc(engine->node_count, sizeof(*engine->incident));
    degrees = (size_t *)pb_calloc(engine->node_count, sizeof(*degrees));
    cursor = (size_t *)pb_calloc(engine->node_count, sizeof(*cursor));
    if (engine->incident == NULL || degrees == NULL || cursor == NULL) {
        free(degrees);
        free(cursor);
        return PB_ENOMEM;
    }
    for (index = 0u; index < edge_count; index++) {
        degrees[engine->edges[index].u]++;
        degrees[engine->edges[index].v]++;
    }
    for (index = 0u; index < (size_t)engine->node_count; index++) {
        if (degrees[index] > 0u) {
            engine->incident[index].items = (uint32_t *)pb_calloc(degrees[index], sizeof(uint32_t));
            if (engine->incident[index].items == NULL) {
                free(degrees);
                free(cursor);
                return PB_ENOMEM;
            }
            engine->incident[index].len = degrees[index];
            engine->incident[index].cap = degrees[index];
        }
    }
    for (index = 0u; index < edge_count; index++) {
        uint32_t u = engine->edges[index].u;
        uint32_t v = engine->edges[index].v;
        engine->incident[u].items[cursor[u]++] = (uint32_t)index;
        engine->incident[v].items[cursor[v]++] = (uint32_t)index;
    }
    free(degrees);
    free(cursor);
    return PB_OK;
}

static void pb_update_dynamic_edge(pb_engine *engine, uint32_t edge_id) {
    pb_dynedge *edge = &engine->dynamic_edges[edge_id];
    double benefit;
    int eligible;
    if (!edge->active) {
        pb_index_heap_remove(engine, edge_id);
        return;
    }
    /*
     * The non-match map is the source of truth.  In particular, component
     * contraction can migrate a non-match from (loser, other) to an already
     * existing (winner, other) edge.  Keeping eligibility dependent only on
     * the edge-local cache would leave that edge schedulable indefinitely.
     */
    if (pb_pairmap_get(&engine->nonmatch_map, pb_pair_key(edge->u, edge->v), NULL)) {
        edge->blocked = 1u;
    }
    benefit = pb_edge_benefit(engine, edge);
    eligible = engine->current[edge->u] && engine->current[edge->v] && !edge->blocked &&
               benefit > 0.0;
    if (eligible) {
        if (pb_index_heap_contains(&engine->current_heap, edge_id)) {
            pb_index_heap_update(engine, edge_id);
        } else {
            pb_index_heap_insert(engine, edge_id);
        }
    } else {
        pb_index_heap_remove(engine, edge_id);
    }
}

static pb_status pb_initialize_dynamic_graph(pb_engine *engine) {
    size_t index;
    engine->dynamic_edges = (pb_dynedge *)pb_calloc(
        engine->edge_count == 0u ? 1u : engine->edge_count, sizeof(*engine->dynamic_edges));
    engine->nonmatch_incident = (pb_u32vec *)pb_calloc(
        engine->node_count, sizeof(*engine->nonmatch_incident));
    if (engine->dynamic_edges == NULL || engine->nonmatch_incident == NULL ||
        !pb_pairmap_init(&engine->edge_map, engine->edge_count) ||
        !pb_pairmap_init(&engine->nonmatch_map, engine->node_count) ||
        !pb_index_heap_init(&engine->current_heap, engine->edge_count)) {
        return PB_ENOMEM;
    }
    for (index = 0u; index < engine->edge_count; index++) {
        pb_dynedge *dynamic = &engine->dynamic_edges[index];
        dynamic->u = engine->edges[index].u;
        dynamic->v = engine->edges[index].v;
        dynamic->sum = engine->edges[index].weight;
        dynamic->maximum = engine->edges[index].weight;
        dynamic->active = 1u;
        if (!pb_pairmap_put(
                &engine->edge_map,
                pb_pair_key(dynamic->u, dynamic->v),
                (uint32_t)index)) {
            return PB_ENOMEM;
        }
    }
    return PB_OK;
}

static double pb_nodes_weight(pb_engine *engine, const uint32_t *nodes, size_t len) {
    uint8_t *marks;
    double result;
    marks = (uint8_t *)pb_calloc(engine->node_count, sizeof(*marks));
    if (marks == NULL) {
        return -1.0;
    }
    result = pb_group_weight(engine, nodes, len, marks);
    free(marks);
    return result;
}

static pb_status pb_copy_external_groups(
    pb_engine *engine,
    const pb_communities *communities,
    pb_nodegroup **out_groups,
    size_t *out_count) {
    pb_nodegroup *groups;
    uint8_t *seen;
    size_t index;
    if (communities == NULL || communities->community_count == 0u) {
        *out_groups = NULL;
        *out_count = 0u;
        return PB_OK;
    }
    if (communities->offsets == NULL || communities->nodes == NULL || communities->offsets[0] != 0u) {
        return PB_EINVAL;
    }
    groups = (pb_nodegroup *)pb_calloc(communities->community_count, sizeof(*groups));
    seen = (uint8_t *)pb_calloc(engine->node_count, sizeof(*seen));
    if (groups == NULL || seen == NULL) {
        free(groups);
        free(seen);
        return PB_ENOMEM;
    }
    for (index = 0u; index < communities->community_count; index++) {
        size_t start = communities->offsets[index];
        size_t end = communities->offsets[index + 1u];
        size_t node_index;
        if (end < start) {
            pb_free_nodegroups(groups, communities->community_count);
            free(seen);
            return PB_EINVAL;
        }
        groups[index].len = end - start;
        groups[index].nodes = (uint32_t *)pb_calloc(
            groups[index].len == 0u ? 1u : groups[index].len, sizeof(uint32_t));
        if (groups[index].nodes == NULL) {
            pb_free_nodegroups(groups, communities->community_count);
            free(seen);
            return PB_ENOMEM;
        }
        for (node_index = 0u; node_index < groups[index].len; node_index++) {
            uint32_t node = communities->nodes[start + node_index];
            if (node >= engine->node_count || seen[node]) {
                pb_free_nodegroups(groups, communities->community_count);
                free(seen);
                return PB_EINVAL;
            }
            seen[node] = 1u;
            groups[index].nodes[node_index] = node;
        }
    }
    free(seen);
    /* Sort final groups by weight, as required by Algorithm 1. */
    for (index = 1u; index < communities->community_count; index++) {
        pb_nodegroup item = groups[index];
        double item_weight = pb_nodes_weight(engine, item.nodes, item.len);
        size_t position = index;
        if (item_weight < 0.0) {
            pb_free_nodegroups(groups, communities->community_count);
            return PB_ENOMEM;
        }
        while (position > 0u) {
            double previous_weight = pb_nodes_weight(
                engine, groups[position - 1u].nodes, groups[position - 1u].len);
            if (previous_weight < 0.0) {
                pb_free_nodegroups(groups, communities->community_count);
                return PB_ENOMEM;
            }
            if (previous_weight >= item_weight - PB_EPSILON) {
                break;
            }
            groups[position] = groups[position - 1u];
            position--;
        }
        groups[position] = item;
    }
    *out_groups = groups;
    *out_count = communities->community_count;
    return PB_OK;
}

static pb_status pb_materialize_communities(
    pb_engine *engine,
    pb_nodegroup *groups,
    size_t group_count) {
    uint32_t *label = NULL;
    size_t *edge_counts = NULL;
    size_t *edge_cursor = NULL;
    size_t index;
    engine->communities = (pb_community *)pb_calloc(group_count == 0u ? 1u : group_count,
                                                     sizeof(*engine->communities));
    if (engine->communities == NULL) {
        return PB_ENOMEM;
    }
    engine->community_count = group_count;
    if (group_count == 0u) {
        return PB_OK;
    }
    label = (uint32_t *)pb_calloc(engine->node_count, sizeof(*label));
    edge_counts = (size_t *)pb_calloc(group_count, sizeof(*edge_counts));
    edge_cursor = (size_t *)pb_calloc(group_count, sizeof(*edge_cursor));
    if (label == NULL || edge_counts == NULL || edge_cursor == NULL) {
        free(label);
        free(edge_counts);
        free(edge_cursor);
        return PB_ENOMEM;
    }
    for (index = 0u; index < (size_t)engine->node_count; index++) {
        label[index] = PB_NONE;
    }
    for (index = 0u; index < group_count; index++) {
        size_t node_index;
        pb_community *community = &engine->communities[index];
        community->nodes = groups[index].nodes;
        groups[index].nodes = NULL;
        community->node_count = groups[index].len;
        community->unqueried_count = groups[index].len;
        community->weight = pb_nodes_weight(engine, community->nodes, community->node_count);
        if (community->weight < 0.0) {
            free(label);
            free(edge_counts);
            free(edge_cursor);
            return PB_ENOMEM;
        }
        for (node_index = 0u; node_index < community->node_count; node_index++) {
            label[community->nodes[node_index]] = (uint32_t)index;
        }
    }
    for (index = 0u; index < engine->edge_count; index++) {
        uint32_t group = label[engine->edges[index].u];
        if (group != PB_NONE && group == label[engine->edges[index].v]) {
            edge_counts[group]++;
        }
    }
    for (index = 0u; index < group_count; index++) {
        pb_idheap *heap = &engine->communities[index].edges;
        heap->ids = (uint32_t *)pb_calloc(edge_counts[index] == 0u ? 1u : edge_counts[index], sizeof(uint32_t));
        if (heap->ids == NULL) {
            free(label);
            free(edge_counts);
            free(edge_cursor);
            return PB_ENOMEM;
        }
        heap->cap = edge_counts[index];
        heap->len = edge_counts[index];
    }
    for (index = 0u; index < engine->edge_count; index++) {
        uint32_t group = label[engine->edges[index].u];
        if (group != PB_NONE && group == label[engine->edges[index].v]) {
            engine->communities[group].edges.ids[edge_cursor[group]++] = (uint32_t)index;
        }
    }
    for (index = 0u; index < group_count; index++) {
        pb_idheap *heap = &engine->communities[index].edges;
        size_t position = heap->len / 2u;
        while (position > 0u) {
            position--;
            pb_idheap_down(engine, heap, position);
        }
    }
    free(label);
    free(edge_counts);
    free(edge_cursor);
    return PB_OK;
}

static void pb_mark_all_current(pb_engine *engine) {
    size_t index;
    for (index = 0u; index < (size_t)engine->node_count; index++) {
        if (engine->parent[index] == index) {
            engine->current[index] = 1u;
        }
    }
    for (index = 0u; index < engine->edge_count; index++) {
        pb_update_dynamic_edge(engine, (uint32_t)index);
    }
}

static pb_status pb_activate_community(pb_engine *engine, size_t community_index) {
    pb_community *community = &engine->communities[community_index];
    size_t index;
    for (index = 0u; index < community->node_count; index++) {
        uint32_t root = pb_find(engine, community->nodes[index]);
        engine->current[root] = 1u;
    }
    for (index = 0u; index < community->node_count; index++) {
        uint32_t node = community->nodes[index];
        size_t adjacency_index;
        for (adjacency_index = 0u; adjacency_index < engine->incident[node].len; adjacency_index++) {
            uint32_t edge_id = engine->incident[node].items[adjacency_index];
            if ((size_t)edge_id < engine->edge_count) {
                pb_update_dynamic_edge(engine, edge_id);
            }
        }
    }
    engine->community_active = 1u;
    return PB_OK;
}

static pb_status pb_engine_create_impl(
    uint32_t node_count,
    pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine,
    int borrow_edges) {
    pb_engine *engine = NULL;
    pb_config resolved;
    pb_edge *working_edges = edges;
    size_t working_edge_count = edge_count;
    pb_nodegroup *groups = NULL;
    size_t group_count = 0u;
    size_t index;
    pb_status status;

    if (out_engine == NULL || config == NULL || node_count == 0u) {
        return PB_EINVAL;
    }
    *out_engine = NULL;
    if (config->struct_size != sizeof(*config) || config->abi_version != PB_ABI_VERSION) {
        return PB_EVERSION;
    }
    resolved = *config;
    if (resolved.batch_size < 2u || resolved.batch_size > node_count ||
        resolved.top_k == 0u || resolved.lambda_w < 0.0 || resolved.lambda_w > 1.0 ||
        resolved.louvain_threshold < 0.0 || resolved.method < PB_METHOD_PERBACCO ||
        resolved.method > PB_METHOD_SUBOPT || resolved.cda < PB_CDA_NONE ||
        resolved.cda > PB_CDA_EXTERNAL) {
        return PB_EINVAL;
    }
    if (resolved.method == PB_METHOD_SUBOPT) {
        if (truth_labels == NULL) {
            return PB_EINVAL;
        }
        working_edges = NULL;
        working_edge_count = 0u;
        resolved.cda = PB_CDA_NONE;
        borrow_edges = 0;
    }
    engine = (pb_engine *)pb_calloc(1u, sizeof(*engine));
    if (engine == NULL) {
        return PB_ENOMEM;
    }
    engine->config = resolved;
    engine->node_count = node_count;
    engine->rng_state = resolved.seed;
    engine->temperature = resolved.method == PB_METHOD_PERBACCO ? (double)resolved.batch_size : 0.0;
    engine->pending_records = (uint32_t *)pb_calloc(resolved.batch_size, sizeof(*engine->pending_records));
    engine->parent = (uint32_t *)pb_calloc(node_count, sizeof(*engine->parent));
    engine->component_size = (uint32_t *)pb_calloc(node_count, sizeof(*engine->component_size));
    engine->member_head = (uint32_t *)pb_calloc(node_count, sizeof(*engine->member_head));
    engine->member_tail = (uint32_t *)pb_calloc(node_count, sizeof(*engine->member_tail));
    engine->member_next = (uint32_t *)pb_calloc(node_count, sizeof(*engine->member_next));
    engine->current = (uint8_t *)pb_calloc(node_count, sizeof(*engine->current));
    engine->queried = (uint8_t *)pb_calloc(node_count, sizeof(*engine->queried));
    if (engine->pending_records == NULL || engine->parent == NULL || engine->component_size == NULL ||
        engine->member_head == NULL || engine->member_tail == NULL || engine->member_next == NULL ||
        engine->current == NULL || engine->queried == NULL) {
        status = PB_ENOMEM;
        goto fail;
    }
    for (index = 0u; index < (size_t)node_count; index++) {
        engine->parent[index] = (uint32_t)index;
        engine->component_size[index] = 1u;
        engine->member_head[index] = (uint32_t)index;
        engine->member_tail[index] = (uint32_t)index;
        engine->member_next[index] = PB_NONE;
    }
    if (truth_labels != NULL) {
        engine->truth_labels = (uint32_t *)pb_calloc(node_count, sizeof(*engine->truth_labels));
        if (engine->truth_labels == NULL) {
            status = PB_ENOMEM;
            goto fail;
        }
        memcpy(engine->truth_labels, truth_labels, node_count * sizeof(*truth_labels));
    }
    status = pb_copy_and_validate_edges(
        engine, working_edges, working_edge_count, borrow_edges);
    if (status != PB_OK) {
        goto fail;
    }

    if (resolved.method == PB_METHOD_PERBACCO && resolved.cda == PB_CDA_LOUVAIN) {
        status = pb_build_louvain_communities(engine, &groups, &group_count);
        if (status != PB_OK) {
            goto fail;
        }
    } else if (resolved.method == PB_METHOD_PERBACCO && resolved.cda == PB_CDA_EXTERNAL) {
        status = pb_copy_external_groups(engine, communities, &groups, &group_count);
        if (status != PB_OK) {
            goto fail;
        }
    }
    status = pb_materialize_communities(engine, groups, group_count);
    pb_free_nodegroups(groups, group_count);
    groups = NULL;
    if (status != PB_OK) {
        goto fail;
    }
    {
        uint32_t has_truth = engine->truth_labels != NULL ? 1u : 0u;
        uint64_t stored_community_count = (uint64_t)engine->community_count;
        engine->graph_hash = pb_fnv1a_update(
            engine->graph_hash, &has_truth, sizeof(has_truth));
        if (engine->truth_labels != NULL) {
            engine->graph_hash = pb_fnv1a_update(
                engine->graph_hash,
                engine->truth_labels,
                (size_t)engine->node_count * sizeof(*engine->truth_labels));
        }
        engine->graph_hash = pb_fnv1a_update(
            engine->graph_hash,
            &stored_community_count,
            sizeof(stored_community_count));
        for (index = 0u; index < engine->community_count; index++) {
            uint64_t stored_node_count = (uint64_t)engine->communities[index].node_count;
            engine->graph_hash = pb_fnv1a_update(
                engine->graph_hash, &stored_node_count, sizeof(stored_node_count));
            engine->graph_hash = pb_fnv1a_update(
                engine->graph_hash,
                engine->communities[index].nodes,
                engine->communities[index].node_count *
                    sizeof(*engine->communities[index].nodes));
        }
    }
    status = pb_initialize_dynamic_graph(engine);
    if (status != PB_OK) {
        goto fail;
    }
    if (resolved.method != PB_METHOD_PERBACCO || group_count == 0u) {
        engine->phase_final = 1u;
        engine->temperature = 0.0;
        pb_mark_all_current(engine);
    }
    *out_engine = engine;
    return PB_OK;

fail:
    pb_free_nodegroups(groups, group_count);
    pb_engine_free(engine);
    return status;
}

pb_status pb_engine_create(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine) {
    return pb_engine_create_impl(
        node_count, (pb_edge *)(uintptr_t)edges, edge_count, config,
        communities, truth_labels, out_engine, 0);
}

pb_status pb_engine_create_borrowed(
    uint32_t node_count,
    pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    pb_engine **out_engine) {
    return pb_engine_create_impl(
        node_count, edges, edge_count, config, communities, truth_labels,
        out_engine, 1);
}

pb_status pb_engine_create_subopt(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    uint32_t batch_size,
    uint64_t seed,
    uint64_t max_queries,
    const uint32_t *truth_labels,
    pb_engine **out_engine) {
    pb_config config;
    pb_config_init(&config);
    config.method = PB_METHOD_SUBOPT;
    config.cda = PB_CDA_NONE;
    config.batch_size = batch_size;
    config.seed = seed;
    config.max_queries = max_queries;
    return pb_engine_create(node_count, edges, edge_count, &config, NULL, truth_labels, out_engine);
}

static pb_status pb_mark_nonmatch(pb_engine *engine, uint32_t left, uint32_t right) {
    uint64_t key;
    uint32_t edge_id;
    int known;
    left = pb_find(engine, left);
    right = pb_find(engine, right);
    if (left == right) {
        pb_set_error(engine, "cannot mark entity %u as a non-match with itself", left);
        return PB_ECONFLICT;
    }
    key = pb_pair_key(left, right);
    known = pb_pairmap_get(&engine->nonmatch_map, key, NULL);
    if (!known) {
        if (!pb_vec_push(&engine->nonmatch_incident[left], right) ||
            !pb_vec_push(&engine->nonmatch_incident[right], left) ||
            !pb_pairmap_put(&engine->nonmatch_map, key, 1u)) {
            return PB_ENOMEM;
        }
    }
    /* Also repair an edge whose non-match key was created by contraction. */
    if (pb_pairmap_get(&engine->edge_map, key, &edge_id)) {
        engine->dynamic_edges[edge_id].blocked = 1u;
        pb_update_dynamic_edge(engine, edge_id);
    }
    return PB_OK;
}

static pb_status pb_union_components(
    pb_engine *engine,
    uint32_t left,
    uint32_t right,
    uint64_t *out_new_matches) {
    uint32_t winner;
    uint32_t loser;
    uint64_t product;
    size_t index;

    left = pb_find(engine, left);
    right = pb_find(engine, right);
    if (left == right) {
        *out_new_matches = 0u;
        return PB_OK;
    }
    if (pb_pairmap_get(&engine->nonmatch_map, pb_pair_key(left, right), NULL)) {
        pb_set_error(engine, "oracle attempted to merge known non-matching entities %u and %u", left, right);
        return PB_ECONFLICT;
    }
    if (engine->incident[left].len > engine->incident[right].len ||
        (engine->incident[left].len == engine->incident[right].len &&
         (engine->component_size[left] > engine->component_size[right] ||
          (engine->component_size[left] == engine->component_size[right] && left < right)))) {
        winner = left;
        loser = right;
    } else {
        winner = right;
        loser = left;
    }
    product = (uint64_t)engine->component_size[winner] *
              (uint64_t)engine->component_size[loser];

    for (index = 0u; index < engine->nonmatch_incident[loser].len; index++) {
        uint32_t other = pb_find(engine, engine->nonmatch_incident[loser].items[index]);
        uint64_t old_key;
        uint64_t new_key;
        if (other == loser || other == winner) {
            continue;
        }
        old_key = pb_pair_key(loser, other);
        if (!pb_pairmap_get(&engine->nonmatch_map, old_key, NULL)) {
            continue;
        }
        pb_pairmap_remove(&engine->nonmatch_map, old_key);
        new_key = pb_pair_key(winner, other);
        if (!pb_pairmap_get(&engine->nonmatch_map, new_key, NULL)) {
            if (!pb_vec_push(&engine->nonmatch_incident[winner], other) ||
                !pb_vec_push(&engine->nonmatch_incident[other], winner) ||
                !pb_pairmap_put(&engine->nonmatch_map, new_key, 1u)) {
                return PB_ENOMEM;
            }
        }
    }

    engine->parent[loser] = winner;
    engine->component_size[winner] += engine->component_size[loser];
    engine->component_size[loser] = 0u;
    engine->member_next[engine->member_tail[winner]] = engine->member_head[loser];
    engine->member_tail[winner] = engine->member_tail[loser];
    engine->current[winner] = (uint8_t)(engine->current[winner] || engine->current[loser]);
    engine->current[loser] = 0u;

    for (index = 0u; index < engine->incident[loser].len; index++) {
        uint32_t edge_id = engine->incident[loser].items[index];
        pb_dynedge *edge;
        uint32_t other;
        uint64_t old_key;
        uint64_t new_key;
        uint32_t existing_id;
        if ((size_t)edge_id >= engine->edge_count) {
            continue;
        }
        edge = &engine->dynamic_edges[edge_id];
        if (!edge->active || (edge->u != loser && edge->v != loser)) {
            continue;
        }
        other = edge->u == loser ? edge->v : edge->u;
        other = pb_find(engine, other);
        old_key = pb_pair_key(edge->u, edge->v);
        pb_index_heap_remove(engine, edge_id);
        pb_pairmap_remove(&engine->edge_map, old_key);
        if (other == winner) {
            edge->active = 0u;
            continue;
        }
        new_key = pb_pair_key(winner, other);
        if (pb_pairmap_get(&engine->edge_map, new_key, &existing_id)) {
            pb_dynedge *existing = &engine->dynamic_edges[existing_id];
            existing->sum += edge->sum;
            if (edge->maximum > existing->maximum) {
                existing->maximum = edge->maximum;
            }
            existing->blocked = (uint8_t)(existing->blocked || edge->blocked ||
                pb_pairmap_get(&engine->nonmatch_map, new_key, NULL));
            edge->active = 0u;
            pb_update_dynamic_edge(engine, existing_id);
        } else {
            edge->u = winner < other ? winner : other;
            edge->v = winner < other ? other : winner;
            edge->blocked = (uint8_t)pb_pairmap_get(&engine->nonmatch_map, new_key, NULL);
            if (!pb_pairmap_put(&engine->edge_map, new_key, edge_id) ||
                !pb_vec_push(&engine->incident[winner], edge_id)) {
                return PB_ENOMEM;
            }
            pb_update_dynamic_edge(engine, edge_id);
        }
    }
    for (index = 0u; index < engine->incident[winner].len; index++) {
        uint32_t edge_id = engine->incident[winner].items[index];
        if ((size_t)edge_id < engine->edge_count) {
            pb_dynedge *edge = &engine->dynamic_edges[edge_id];
            if (edge->active && (edge->u == winner || edge->v == winner)) {
                pb_update_dynamic_edge(engine, edge_id);
            }
        }
    }
    *out_new_matches = product;
    return PB_OK;
}

static void pb_mark_component_queried(pb_engine *engine, uint32_t representative) {
    uint32_t root = pb_find(engine, representative);
    uint32_t member = engine->member_head[root];
    while (member != PB_NONE) {
        engine->queried[member] = 1u;
        member = engine->member_next[member];
    }
}

static pb_status pb_set_pending_batch(
    pb_engine *engine,
    pb_batch_kind kind,
    const uint32_t *records,
    size_t len,
    pb_record_id *caller_records,
    pb_batch *out_batch) {
    if (len < 2u || len > (size_t)engine->config.batch_size) {
        return PB_EINTERNAL;
    }
    memcpy(engine->pending_records, records, len * sizeof(*records));
    memcpy(caller_records, records, len * sizeof(*records));
    engine->pending_kind = kind;
    engine->pending_len = len;
    engine->pending_matches_before = engine->discovered_matches;
    engine->awaiting_submission = 1u;
    out_batch->kind = kind;
    out_batch->len = len;
    out_batch->records = caller_records;
    return PB_OK;
}

static pb_status pb_select_current(pb_engine *engine, uint32_t *records, size_t *out_len) {
    uint32_t *popped = NULL;
    uint32_t *candidates = NULL;
    uint8_t *seen = NULL;
    pb_temp_edge *temporary = NULL;
    size_t popped_count = 0u;
    size_t candidate_count = 0u;
    size_t temporary_count = 0u;
    size_t limit = (size_t)engine->config.top_k;
    pb_status status;
    size_t index;

    if (limit > engine->current_heap.len) {
        limit = engine->current_heap.len;
    }
    popped = (uint32_t *)pb_calloc(limit == 0u ? 1u : limit, sizeof(*popped));
    candidates = (uint32_t *)pb_calloc(limit == 0u ? 1u : limit * 2u, sizeof(*candidates));
    temporary = (pb_temp_edge *)pb_calloc(limit == 0u ? 1u : limit, sizeof(*temporary));
    seen = (uint8_t *)pb_calloc(engine->node_count, sizeof(*seen));
    if (popped == NULL || candidates == NULL || temporary == NULL || seen == NULL) {
        free(popped);
        free(candidates);
        free(temporary);
        free(seen);
        return PB_ENOMEM;
    }
    while (popped_count < limit && engine->current_heap.len > 0u) {
        uint32_t edge_id = pb_index_heap_pop(engine);
        pb_dynedge *edge = &engine->dynamic_edges[edge_id];
        double benefit = pb_edge_benefit(engine, edge);
        popped[popped_count++] = edge_id;
        if (!edge->active || edge->blocked || benefit <= engine->temperature + PB_EPSILON) {
            break;
        }
        temporary[temporary_count].u = edge->u;
        temporary[temporary_count].v = edge->v;
        temporary[temporary_count].weight = benefit;
        temporary_count++;
        if (!seen[edge->u]) {
            seen[edge->u] = 1u;
            candidates[candidate_count++] = edge->u;
        }
        if (!seen[edge->v]) {
            seen[edge->v] = 1u;
            candidates[candidate_count++] = edge->v;
        }
    }
    for (index = 0u; index < popped_count; index++) {
        pb_update_dynamic_edge(engine, popped[index]);
    }
    if ((!engine->phase_final && candidate_count < (size_t)engine->config.batch_size) ||
        candidate_count < 2u) {
        *out_len = 0u;
        free(popped);
        free(candidates);
        free(temporary);
        free(seen);
        return PB_OK;
    }
    qsort(temporary, temporary_count, sizeof(*temporary), pb_temp_edge_compare_desc);
    status = pb_greedy_hs(
        engine, candidates, candidate_count, temporary, temporary_count, records, out_len);
    free(popped);
    free(candidates);
    free(temporary);
    free(seen);
    return status;
}

static pb_status pb_select_community(pb_engine *engine, uint32_t *records, size_t *out_len) {
    pb_community *community = &engine->communities[engine->community_index];
    uint32_t *candidates = NULL;
    uint32_t *popped = NULL;
    pb_temp_edge *temporary = NULL;
    size_t candidate_count = 0u;
    size_t popped_count = 0u;
    size_t temporary_count = 0u;
    size_t limit = (size_t)engine->config.top_k;
    size_t index;
    pb_status status;

    candidates = (uint32_t *)pb_calloc(
        community->node_count == 0u ? 1u : community->node_count, sizeof(*candidates));
    if (candidates == NULL) {
        return PB_ENOMEM;
    }
    for (index = 0u; index < community->node_count; index++) {
        uint32_t node = community->nodes[index];
        if (!engine->queried[node]) {
            candidates[candidate_count++] = node;
        }
    }
    community->unqueried_count = candidate_count;
    if (candidate_count < (size_t)engine->config.batch_size) {
        free(candidates);
        *out_len = 0u;
        return PB_OK;
    }
    if (limit > community->edges.len) {
        limit = community->edges.len;
    }
    popped = (uint32_t *)pb_calloc(limit == 0u ? 1u : limit, sizeof(*popped));
    temporary = (pb_temp_edge *)pb_calloc(limit == 0u ? 1u : limit, sizeof(*temporary));
    if (popped == NULL || temporary == NULL) {
        free(candidates);
        free(popped);
        free(temporary);
        return PB_ENOMEM;
    }
    while (popped_count < limit && community->edges.len > 0u) {
        uint32_t edge_id = pb_idheap_pop(engine, &community->edges);
        const pb_edge *edge = &engine->edges[edge_id];
        if (engine->queried[edge->u] || engine->queried[edge->v]) {
            continue;
        }
        popped[popped_count++] = edge_id;
        temporary[temporary_count].u = edge->u;
        temporary[temporary_count].v = edge->v;
        temporary[temporary_count].weight = edge->weight;
        temporary_count++;
    }
    for (index = 0u; index < popped_count; index++) {
        if (!pb_idheap_push(engine, &community->edges, popped[index])) {
            free(candidates);
            free(popped);
            free(temporary);
            return PB_ENOMEM;
        }
    }
    qsort(temporary, temporary_count, sizeof(*temporary), pb_temp_edge_compare_desc);
    status = pb_greedy_hs(
        engine, candidates, candidate_count, temporary, temporary_count, records, out_len);
    free(candidates);
    free(popped);
    free(temporary);
    return status;
}

static pb_status pb_journal_append(
    pb_engine *engine,
    pb_batch_kind kind,
    const uint32_t *records,
    const uint32_t *labels,
    size_t len) {
    pb_journal_entry *replacement;
    pb_journal_entry *entry;
    size_t capacity;
    if (engine->journal_len == engine->journal_cap) {
        capacity = engine->journal_cap == 0u ? 16u : engine->journal_cap * 2u;
        replacement = (pb_journal_entry *)realloc(
            engine->journal, capacity * sizeof(*replacement));
        if (replacement == NULL) {
            return PB_ENOMEM;
        }
        memset(replacement + engine->journal_cap, 0,
               (capacity - engine->journal_cap) * sizeof(*replacement));
        engine->journal = replacement;
        engine->journal_cap = capacity;
    }
    entry = &engine->journal[engine->journal_len];
    entry->records = (uint32_t *)pb_calloc(len, sizeof(*entry->records));
    entry->labels = (uint32_t *)pb_calloc(len, sizeof(*entry->labels));
    if (entry->records == NULL || entry->labels == NULL) {
        free(entry->records);
        free(entry->labels);
        memset(entry, 0, sizeof(*entry));
        return PB_ENOMEM;
    }
    entry->kind = kind;
    entry->len = (uint32_t)len;
    memcpy(entry->records, records, len * sizeof(*records));
    memcpy(entry->labels, labels, len * sizeof(*labels));
    engine->journal_len++;
    return PB_OK;
}

pb_status pb_engine_next_batch(
    pb_engine *engine,
    pb_record_id *records,
    size_t capacity,
    pb_batch *out_batch) {
    uint32_t *selected;
    size_t selected_count;
    pb_status status;
    if (engine == NULL || records == NULL || out_batch == NULL) {
        return PB_EINVAL;
    }
    if (engine->awaiting_submission) {
        pb_set_error(engine, "submit the outstanding oracle partition before requesting another batch");
        return PB_ESTATE;
    }
    if (capacity < (size_t)engine->config.batch_size) {
        pb_set_error(engine, "record buffer capacity is smaller than batch_size");
        return PB_EINVAL;
    }
    if (engine->finished) {
        return PB_DONE;
    }
    if (engine->config.max_queries > 0u && engine->query_count >= engine->config.max_queries) {
        return PB_BUDGET_EXHAUSTED;
    }
    selected = engine->pending_records;
    if (engine->config.method == PB_METHOD_SUBOPT) {
        selected_count = 0u;
        status = pb_select_subopt(engine, selected, &selected_count);
        if (status != PB_OK) {
            return status;
        }
        if (selected_count == 0u) {
            engine->finished = 1u;
            return PB_DONE;
        }
        return pb_set_pending_batch(
            engine, PB_BATCH_CURRENT, selected, selected_count, records, out_batch);
    }
    for (;;) {
        selected_count = 0u;
        if (engine->phase_final) {
            status = pb_select_current(engine, selected, &selected_count);
            if (status != PB_OK) {
                return status;
            }
            if (selected_count == 0u) {
                engine->finished = 1u;
                return PB_DONE;
            }
            return pb_set_pending_batch(
                engine, PB_BATCH_CURRENT, selected, selected_count, records, out_batch);
        }
        if (engine->community_index >= engine->community_count) {
            engine->phase_final = 1u;
            engine->temperature = 0.0;
            pb_mark_all_current(engine);
            continue;
        }
        if (!engine->community_active) {
            status = pb_activate_community(engine, engine->community_index);
            if (status != PB_OK) {
                return status;
            }
            engine->prefer_current = 0u;
        }
        if (!engine->prefer_current) {
            status = pb_select_community(engine, selected, &selected_count);
            if (status != PB_OK) {
                return status;
            }
            if (selected_count == (size_t)engine->config.batch_size) {
                return pb_set_pending_batch(
                    engine, PB_BATCH_COMMUNITY, selected, selected_count, records, out_batch);
            }
            engine->community_index++;
            engine->community_active = 0u;
            engine->prefer_current = 0u;
        } else {
            status = pb_select_current(engine, selected, &selected_count);
            if (status != PB_OK) {
                return status;
            }
            if (selected_count == (size_t)engine->config.batch_size) {
                return pb_set_pending_batch(
                    engine, PB_BATCH_CURRENT, selected, selected_count, records, out_batch);
            }
            engine->temperature *= 1.0 - 1.0 / (double)engine->config.batch_size;
            engine->prefer_current = 0u;
        }
    }
}

pb_status pb_engine_submit_partition(
    pb_engine *engine,
    const uint32_t *cluster_labels,
    size_t label_count) {
    uint32_t *records_copy;
    uint32_t *labels_copy;
    uint64_t gained = 0u;
    size_t left;
    size_t right;
    pb_status status;
    if (engine == NULL || cluster_labels == NULL) {
        return PB_EINVAL;
    }
    if (!engine->awaiting_submission || label_count != engine->pending_len) {
        pb_set_error(engine, "partition length does not match an outstanding batch");
        return PB_ESTATE;
    }
    for (left = 0u; left < label_count; left++) {
        uint32_t left_root = pb_find(engine, engine->pending_records[left]);
        for (right = left + 1u; right < label_count; right++) {
            uint32_t right_root = pb_find(engine, engine->pending_records[right]);
            if (left_root == right_root && cluster_labels[left] != cluster_labels[right]) {
                pb_set_error(engine, "partition splits an already known entity");
                return PB_ECONFLICT;
            }
            if (left_root != right_root && cluster_labels[left] == cluster_labels[right] &&
                pb_pairmap_get(&engine->nonmatch_map, pb_pair_key(left_root, right_root), NULL)) {
                pb_set_error(engine, "partition merges a pair already known to be distinct");
                return PB_ECONFLICT;
            }
        }
    }
    records_copy = (uint32_t *)pb_calloc(label_count, sizeof(*records_copy));
    labels_copy = (uint32_t *)pb_calloc(label_count, sizeof(*labels_copy));
    if (records_copy == NULL || labels_copy == NULL) {
        free(records_copy);
        free(labels_copy);
        return PB_ENOMEM;
    }
    memcpy(records_copy, engine->pending_records, label_count * sizeof(*records_copy));
    memcpy(labels_copy, cluster_labels, label_count * sizeof(*labels_copy));

    for (left = 0u; left < label_count; left++) {
        for (right = left + 1u; right < label_count; right++) {
            if (cluster_labels[left] == cluster_labels[right]) {
                uint64_t new_matches;
                status = pb_union_components(
                    engine, engine->pending_records[left], engine->pending_records[right], &new_matches);
                if (status != PB_OK) {
                    free(records_copy);
                    free(labels_copy);
                    return status;
                }
                gained += new_matches;
            }
        }
    }
    for (left = 0u; left < label_count; left++) {
        for (right = left + 1u; right < label_count; right++) {
            if (cluster_labels[left] != cluster_labels[right]) {
                uint32_t left_root = pb_find(engine, engine->pending_records[left]);
                uint32_t right_root = pb_find(engine, engine->pending_records[right]);
                if (left_root != right_root) {
                    status = pb_mark_nonmatch(engine, left_root, right_root);
                    if (status != PB_OK) {
                        free(records_copy);
                        free(labels_copy);
                        return status;
                    }
                }
            }
        }
    }
    for (left = 0u; left < label_count; left++) {
        pb_mark_component_queried(engine, engine->pending_records[left]);
    }
    status = pb_journal_append(
        engine, engine->pending_kind, records_copy, labels_copy, label_count);
    free(records_copy);
    free(labels_copy);
    if (status != PB_OK) {
        return status;
    }
    engine->discovered_matches += gained;
    engine->query_count++;
    if (engine->pending_kind == PB_BATCH_COMMUNITY) {
        engine->community_batches++;
        engine->community_gain_sum += gained;
        engine->community_gain_count++;
    } else {
        engine->current_batches++;
        if (!engine->phase_final && engine->community_gain_count > 0u &&
            (double)gained < (double)engine->community_gain_sum /
                                 (double)engine->community_gain_count) {
            if (engine->temperature <= DBL_MAX / 2.0) {
                engine->temperature *= 2.0;
            } else {
                engine->temperature = DBL_MAX;
            }
        }
    }
    engine->awaiting_submission = 0u;
    engine->pending_len = 0u;
    engine->prefer_current = 1u;
    return PB_OK;
}

pb_status pb_engine_group_members(
    const pb_engine *engine,
    pb_record_id representative,
    pb_record_id *members,
    size_t capacity,
    size_t *out_count) {
    uint32_t root;
    uint32_t member;
    size_t count = 0u;
    if (engine == NULL || out_count == NULL || representative >= engine->node_count ||
        (capacity > 0u && members == NULL)) {
        return PB_EINVAL;
    }
    root = pb_find_const(engine, representative);
    member = engine->member_head[root];
    while (member != PB_NONE) {
        if (count < capacity) {
            members[count] = member;
        }
        count++;
        member = engine->member_next[member];
    }
    *out_count = count;
    return capacity < count ? PB_EOVERFLOW : PB_OK;
}

pb_status pb_engine_get_stats(const pb_engine *engine, pb_stats *out_stats) {
    if (engine == NULL || out_stats == NULL) {
        return PB_EINVAL;
    }
    out_stats->query_count = engine->query_count;
    out_stats->discovered_matches = engine->discovered_matches;
    out_stats->candidate_edges = (uint64_t)engine->current_heap.len;
    out_stats->community_batches = engine->community_batches;
    out_stats->current_batches = engine->current_batches;
    out_stats->community_records = 0u;
    for (size_t index = 0u; index < engine->community_count; index++) {
        out_stats->community_records += (uint64_t)engine->communities[index].node_count;
    }
    out_stats->temperature = engine->temperature;
    out_stats->finished = engine->finished ? 1 : 0;
    return PB_OK;
}

static uint64_t pb_config_hash(const pb_config *config) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = pb_fnv1a_update(hash, &config->method, sizeof(config->method));
    hash = pb_fnv1a_update(hash, &config->cda, sizeof(config->cda));
    hash = pb_fnv1a_update(hash, &config->batch_size, sizeof(config->batch_size));
    hash = pb_fnv1a_update(hash, &config->top_k, sizeof(config->top_k));
    hash = pb_fnv1a_update(hash, &config->seed, sizeof(config->seed));
    hash = pb_fnv1a_update(hash, &config->max_queries, sizeof(config->max_queries));
    hash = pb_fnv1a_update(hash, &config->lambda_w, sizeof(config->lambda_w));
    hash = pb_fnv1a_update(hash, &config->louvain_resolution, sizeof(config->louvain_resolution));
    hash = pb_fnv1a_update(hash, &config->louvain_threshold, sizeof(config->louvain_threshold));
    return hash;
}

static void pb_write_u32(uint8_t **cursor, uint32_t value) {
    uint8_t *output = *cursor;
    output[0] = (uint8_t)(value & UINT32_C(0xff));
    output[1] = (uint8_t)((value >> 8u) & UINT32_C(0xff));
    output[2] = (uint8_t)((value >> 16u) & UINT32_C(0xff));
    output[3] = (uint8_t)((value >> 24u) & UINT32_C(0xff));
    *cursor += 4u;
}

static void pb_write_u64(uint8_t **cursor, uint64_t value) {
    unsigned shift;
    for (shift = 0u; shift < 64u; shift += 8u) {
        **cursor = (uint8_t)((value >> shift) & UINT64_C(0xff));
        (*cursor)++;
    }
}

static int pb_read_u32(const uint8_t **cursor, const uint8_t *end, uint32_t *value) {
    const uint8_t *input = *cursor;
    if ((size_t)(end - input) < 4u) {
        return 0;
    }
    *value = (uint32_t)input[0] |
             ((uint32_t)input[1] << 8u) |
             ((uint32_t)input[2] << 16u) |
             ((uint32_t)input[3] << 24u);
    *cursor += 4u;
    return 1;
}

static int pb_read_u64(const uint8_t **cursor, const uint8_t *end, uint64_t *value) {
    const uint8_t *input = *cursor;
    unsigned shift;
    uint64_t result = 0u;
    if ((size_t)(end - input) < 8u) {
        return 0;
    }
    for (shift = 0u; shift < 64u; shift += 8u) {
        result |= (uint64_t)(*input++) << shift;
    }
    *cursor = input;
    *value = result;
    return 1;
}

pb_status pb_engine_snapshot_size(const pb_engine *engine, size_t *out_size) {
    size_t size = 64u;
    size_t index;
    if (engine == NULL || out_size == NULL) {
        return PB_EINVAL;
    }
    if (engine->awaiting_submission) {
        return PB_ESTATE;
    }
    for (index = 0u; index < engine->journal_len; index++) {
        size_t records_size = (size_t)engine->journal[index].len * 8u;
        if (size > SIZE_MAX - 8u || records_size > SIZE_MAX - size - 8u) {
            return PB_EOVERFLOW;
        }
        size += 8u + records_size;
    }
    *out_size = size;
    return PB_OK;
}

pb_status pb_engine_snapshot(
    const pb_engine *engine,
    void *buffer,
    size_t capacity,
    size_t *out_size) {
    static const uint8_t magic[8] = {'P', 'B', 'J', 'R', 'N', 'L', '1', '\0'};
    uint8_t *output = (uint8_t *)buffer;
    uint8_t *cursor;
    size_t needed;
    size_t index;
    uint64_t checksum;
    pb_status status = pb_engine_snapshot_size(engine, &needed);
    if (status != PB_OK) {
        return status;
    }
    if (out_size == NULL || buffer == NULL) {
        return PB_EINVAL;
    }
    *out_size = needed;
    if (capacity < needed) {
        return PB_EOVERFLOW;
    }
    cursor = output;
    memcpy(cursor, magic, sizeof(magic));
    cursor += sizeof(magic);
    pb_write_u32(&cursor, PB_SNAPSHOT_VERSION);
    pb_write_u32(&cursor, PB_ABI_VERSION);
    pb_write_u32(&cursor, engine->node_count);
    pb_write_u32(&cursor, engine->finished ? 1u : 0u);
    pb_write_u64(&cursor, (uint64_t)engine->edge_count);
    pb_write_u64(&cursor, engine->graph_hash);
    pb_write_u64(&cursor, pb_config_hash(&engine->config));
    pb_write_u64(&cursor, (uint64_t)engine->journal_len);
    for (index = 0u; index < engine->journal_len; index++) {
        const pb_journal_entry *entry = &engine->journal[index];
        size_t item;
        pb_write_u32(&cursor, (uint32_t)entry->kind);
        pb_write_u32(&cursor, entry->len);
        for (item = 0u; item < (size_t)entry->len; item++) {
            pb_write_u32(&cursor, entry->records[item]);
        }
        for (item = 0u; item < (size_t)entry->len; item++) {
            pb_write_u32(&cursor, entry->labels[item]);
        }
    }
    checksum = pb_fnv1a_update(UINT64_C(1469598103934665603), output, needed - 8u);
    pb_write_u64(&cursor, checksum);
    return (size_t)(cursor - output) == needed ? PB_OK : PB_EINTERNAL;
}

pb_status pb_engine_restore(
    uint32_t node_count,
    const pb_edge *edges,
    size_t edge_count,
    const pb_config *config,
    const pb_communities *communities,
    const uint32_t *truth_labels,
    const void *snapshot,
    size_t snapshot_size,
    pb_engine **out_engine) {
    static const uint8_t magic[8] = {'P', 'B', 'J', 'R', 'N', 'L', '1', '\0'};
    const uint8_t *input = (const uint8_t *)snapshot;
    const uint8_t *cursor;
    const uint8_t *payload_end;
    uint32_t version;
    uint32_t abi;
    uint32_t stored_nodes;
    uint32_t flags;
    uint64_t stored_edges;
    uint64_t stored_graph_hash;
    uint64_t stored_config_hash;
    uint64_t entry_count;
    uint64_t stored_checksum;
    uint64_t actual_checksum;
    pb_engine *engine = NULL;
    uint32_t *expected_records = NULL;
    uint32_t *labels = NULL;
    uint32_t *actual_records = NULL;
    uint64_t entry_index;
    pb_status status;

    if (snapshot == NULL || out_engine == NULL || config == NULL || snapshot_size < 64u) {
        return PB_EINVAL;
    }
    *out_engine = NULL;
    payload_end = input + snapshot_size - 8u;
    cursor = payload_end;
    if (!pb_read_u64(&cursor, input + snapshot_size, &stored_checksum)) {
        return PB_EINVAL;
    }
    actual_checksum = pb_fnv1a_update(
        UINT64_C(1469598103934665603), input, snapshot_size - 8u);
    if (stored_checksum != actual_checksum || memcmp(input, magic, sizeof(magic)) != 0) {
        return PB_EVERSION;
    }
    cursor = input + sizeof(magic);
    if (!pb_read_u32(&cursor, payload_end, &version) ||
        !pb_read_u32(&cursor, payload_end, &abi) ||
        !pb_read_u32(&cursor, payload_end, &stored_nodes) ||
        !pb_read_u32(&cursor, payload_end, &flags) ||
        !pb_read_u64(&cursor, payload_end, &stored_edges) ||
        !pb_read_u64(&cursor, payload_end, &stored_graph_hash) ||
        !pb_read_u64(&cursor, payload_end, &stored_config_hash) ||
        !pb_read_u64(&cursor, payload_end, &entry_count)) {
        return PB_EINVAL;
    }
    if (version != PB_SNAPSHOT_VERSION || abi != PB_ABI_VERSION) {
        return PB_EVERSION;
    }
    if (stored_nodes != node_count || stored_edges != (uint64_t)edge_count || flags > 1u) {
        return PB_EINVAL;
    }
    status = pb_engine_create(
        node_count, edges, edge_count, config, communities, truth_labels, &engine);
    if (status != PB_OK) {
        return status;
    }
    if (engine->graph_hash != stored_graph_hash ||
        pb_config_hash(&engine->config) != stored_config_hash) {
        pb_engine_free(engine);
        return PB_EVERSION;
    }
    expected_records = (uint32_t *)pb_calloc(config->batch_size, sizeof(*expected_records));
    labels = (uint32_t *)pb_calloc(config->batch_size, sizeof(*labels));
    actual_records = (uint32_t *)pb_calloc(config->batch_size, sizeof(*actual_records));
    if (expected_records == NULL || labels == NULL || actual_records == NULL) {
        status = PB_ENOMEM;
        goto restore_fail;
    }
    for (entry_index = 0u; entry_index < entry_count; entry_index++) {
        uint32_t kind;
        uint32_t len;
        size_t item;
        pb_batch batch;
        if (!pb_read_u32(&cursor, payload_end, &kind) ||
            !pb_read_u32(&cursor, payload_end, &len) || len < 2u ||
            len > config->batch_size || kind > (uint32_t)PB_BATCH_CURRENT) {
            status = PB_EVERSION;
            goto restore_fail;
        }
        for (item = 0u; item < (size_t)len; item++) {
            if (!pb_read_u32(&cursor, payload_end, &expected_records[item])) {
                status = PB_EVERSION;
                goto restore_fail;
            }
        }
        for (item = 0u; item < (size_t)len; item++) {
            if (!pb_read_u32(&cursor, payload_end, &labels[item])) {
                status = PB_EVERSION;
                goto restore_fail;
            }
        }
        status = pb_engine_next_batch(
            engine, actual_records, config->batch_size, &batch);
        if (status != PB_OK || batch.kind != (pb_batch_kind)kind || batch.len != (size_t)len ||
            memcmp(expected_records, actual_records, (size_t)len * sizeof(*actual_records)) != 0) {
            status = PB_EVERSION;
            goto restore_fail;
        }
        status = pb_engine_submit_partition(engine, labels, len);
        if (status != PB_OK) {
            goto restore_fail;
        }
    }
    if (cursor != payload_end) {
        status = PB_EVERSION;
        goto restore_fail;
    }
    if (flags != 0u) {
        pb_batch batch;
        status = pb_engine_next_batch(engine, actual_records, config->batch_size, &batch);
        if (status != PB_DONE) {
            status = PB_EVERSION;
            goto restore_fail;
        }
    }
    free(expected_records);
    free(labels);
    free(actual_records);
    *out_engine = engine;
    return PB_OK;

restore_fail:
    free(expected_records);
    free(labels);
    free(actual_records);
    pb_engine_free(engine);
    return status;
}

void pb_engine_free(pb_engine *engine) {
    size_t index;
    if (engine == NULL) {
        return;
    }
    if (engine->incident != NULL) {
        for (index = 0u; index < (size_t)engine->node_count; index++) {
            free(engine->incident[index].items);
        }
    }
    if (engine->nonmatch_incident != NULL) {
        for (index = 0u; index < (size_t)engine->node_count; index++) {
            free(engine->nonmatch_incident[index].items);
        }
    }
    if (engine->communities != NULL) {
        for (index = 0u; index < engine->community_count; index++) {
            free(engine->communities[index].nodes);
            free(engine->communities[index].edges.ids);
        }
    }
    if (engine->journal != NULL) {
        for (index = 0u; index < engine->journal_len; index++) {
            free(engine->journal[index].records);
            free(engine->journal[index].labels);
        }
    }
    if (engine->owns_edges) {
        free(engine->edges);
    }
    free(engine->parent);
    free(engine->component_size);
    free(engine->member_head);
    free(engine->member_tail);
    free(engine->member_next);
    free(engine->current);
    free(engine->queried);
    free(engine->truth_labels);
    free(engine->incident);
    free(engine->nonmatch_incident);
    free(engine->dynamic_edges);
    free(engine->communities);
    free(engine->pending_records);
    free(engine->journal);
    pb_pairmap_destroy(&engine->edge_map);
    pb_pairmap_destroy(&engine->nonmatch_map);
    pb_index_heap_destroy(&engine->current_heap);
    free(engine);
}
