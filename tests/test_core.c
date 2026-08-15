#include "perbacco.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NODE_COUNT = 6, EDGE_COUNT = 15, BATCH_SIZE = 3 };

static const uint32_t truth[NODE_COUNT] = {0u, 0u, 0u, 1u, 1u, 2u};

static void build_complete_graph(pb_edge edges[EDGE_COUNT]) {
    size_t cursor = 0u;
    uint32_t left;
    for (left = 0u; left < NODE_COUNT; left++) {
        uint32_t right;
        for (right = left + 1u; right < NODE_COUNT; right++) {
            edges[cursor].u = left;
            edges[cursor].v = right;
            edges[cursor].weight = truth[left] == truth[right]
                                               ? 1.0
                                               : 0.2 + (double)(left + right) / 100.0;
            cursor++;
        }
    }
    assert(cursor == EDGE_COUNT);
}

static void oracle_labels(const pb_batch *batch, uint32_t labels[BATCH_SIZE]) {
    size_t index;
    for (index = 0u; index < batch->len; index++) {
        labels[index] = truth[batch->records[index]];
    }
}

static pb_engine *create_engine(pb_method method, pb_cda cda, uint64_t max_queries) {
    pb_edge edges[EDGE_COUNT];
    pb_config config;
    pb_engine *engine = NULL;
    pb_status status;
    const size_t offsets[3] = {0u, 3u, 6u};
    const uint32_t nodes[NODE_COUNT] = {0u, 1u, 2u, 3u, 4u, 5u};
    const pb_communities communities = {2u, offsets, nodes};
    build_complete_graph(edges);
    pb_config_init(&config);
    config.method = method;
    config.cda = cda;
    config.batch_size = BATCH_SIZE;
    config.top_k = 32u;
    config.max_queries = max_queries;
    status = pb_engine_create(
        NODE_COUNT, edges, EDGE_COUNT, &config,
        cda == PB_CDA_EXTERNAL ? &communities : NULL,
        method == PB_METHOD_SUBOPT ? truth : NULL, &engine);
    assert(status == PB_OK);
    assert(engine != NULL);
    return engine;
}

static void run_to_completion(pb_method method, pb_cda cda) {
    pb_engine *engine = create_engine(method, cda, 0u);
    uint32_t records[BATCH_SIZE];
    uint32_t labels[BATCH_SIZE];
    pb_batch batch;
    pb_stats stats;
    pb_status status;
    size_t guard = 0u;
    do {
        status = pb_engine_next_batch(engine, records, BATCH_SIZE, &batch);
        if (status == PB_OK) {
            assert(batch.records == records);
            assert(batch.len >= 2u && batch.len <= BATCH_SIZE);
            oracle_labels(&batch, labels);
            assert(pb_engine_submit_partition(engine, labels, batch.len) == PB_OK);
        }
        guard++;
        assert(guard < 100u);
    } while (status == PB_OK);
    assert(status == PB_DONE);
    assert(pb_engine_get_stats(engine, &stats) == PB_OK);
    assert(stats.finished == 1);
    assert(stats.discovered_matches == 4u);
    assert(stats.query_count > 0u);
    pb_engine_free(engine);
}

static void test_all_methods(void) {
    run_to_completion(PB_METHOD_PERBAC, PB_CDA_NONE);
    run_to_completion(PB_METHOD_ONLINE, PB_CDA_NONE);
    run_to_completion(PB_METHOD_SUBOPT, PB_CDA_NONE);
    run_to_completion(PB_METHOD_PERBACCO, PB_CDA_EXTERNAL);
    run_to_completion(PB_METHOD_PERBACCO, PB_CDA_LOUVAIN);
}

static void test_budget_and_state(void) {
    pb_engine *engine = create_engine(PB_METHOD_PERBAC, PB_CDA_NONE, 1u);
    uint32_t records[BATCH_SIZE];
    uint32_t labels[BATCH_SIZE];
    pb_batch batch;
    pb_stats stats;
    assert(pb_engine_next_batch(engine, records, BATCH_SIZE, &batch) == PB_OK);
    assert(pb_engine_next_batch(engine, records, BATCH_SIZE, &batch) == PB_ESTATE);
    oracle_labels(&batch, labels);
    assert(pb_engine_submit_partition(engine, labels, batch.len) == PB_OK);
    assert(pb_engine_next_batch(engine, records, BATCH_SIZE, &batch) == PB_BUDGET_EXHAUSTED);
    assert(pb_engine_get_stats(engine, &stats) == PB_OK);
    assert(stats.query_count == 1u);
    pb_engine_free(engine);
}

static void test_members(void) {
    pb_engine *engine = create_engine(PB_METHOD_SUBOPT, PB_CDA_NONE, 1u);
    uint32_t records[BATCH_SIZE];
    uint32_t labels[BATCH_SIZE];
    uint32_t members[NODE_COUNT];
    pb_batch batch;
    size_t member_count = 0u;
    assert(pb_engine_next_batch(engine, records, BATCH_SIZE, &batch) == PB_OK);
    oracle_labels(&batch, labels);
    assert(pb_engine_submit_partition(engine, labels, batch.len) == PB_OK);
    assert(pb_engine_group_members(engine, records[0], NULL, 0u, &member_count) == PB_EOVERFLOW);
    assert(member_count >= 1u && member_count <= BATCH_SIZE);
    assert(pb_engine_group_members(engine, records[0], members, NODE_COUNT, &member_count) == PB_OK);
    pb_engine_free(engine);
}

static void test_snapshot_replay(void) {
    pb_edge edges[EDGE_COUNT];
    pb_config config;
    pb_engine *engine = NULL;
    pb_engine *restored = NULL;
    uint32_t records[BATCH_SIZE];
    uint32_t restored_records[BATCH_SIZE];
    uint32_t labels[BATCH_SIZE];
    pb_batch batch;
    pb_batch restored_batch;
    pb_stats stats;
    pb_stats restored_stats;
    void *snapshot;
    size_t snapshot_size = 0u;
    size_t written = 0u;
    size_t index;
    build_complete_graph(edges);
    pb_config_init(&config);
    config.method = PB_METHOD_ONLINE;
    config.cda = PB_CDA_NONE;
    config.batch_size = BATCH_SIZE;
    config.top_k = 32u;
    for (index = 0u; index < 2u; index++) {
        if (engine == NULL) {
            assert(pb_engine_create(
                       NODE_COUNT, edges, EDGE_COUNT, &config, NULL, NULL, &engine) == PB_OK);
        }
        assert(pb_engine_next_batch(engine, records, BATCH_SIZE, &batch) == PB_OK);
        oracle_labels(&batch, labels);
        assert(pb_engine_submit_partition(engine, labels, batch.len) == PB_OK);
    }
    assert(pb_engine_snapshot_size(engine, &snapshot_size) == PB_OK);
    snapshot = malloc(snapshot_size);
    assert(snapshot != NULL);
    assert(pb_engine_snapshot(engine, snapshot, snapshot_size, &written) == PB_OK);
    assert(written == snapshot_size);
    assert(pb_engine_restore(
               NODE_COUNT, edges, EDGE_COUNT, &config, NULL, NULL,
               snapshot, snapshot_size, &restored) == PB_OK);
    assert(pb_engine_get_stats(engine, &stats) == PB_OK);
    assert(pb_engine_get_stats(restored, &restored_stats) == PB_OK);
    assert(stats.query_count == restored_stats.query_count);
    assert(stats.discovered_matches == restored_stats.discovered_matches);
    assert(pb_engine_next_batch(engine, records, BATCH_SIZE, &batch) == PB_OK);
    assert(pb_engine_next_batch(restored, restored_records, BATCH_SIZE, &restored_batch) == PB_OK);
    assert(batch.kind == restored_batch.kind && batch.len == restored_batch.len);
    assert(memcmp(records, restored_records, batch.len * sizeof(*records)) == 0);
    free(snapshot);
    pb_engine_free(restored);
    pb_engine_free(engine);
}

static void test_validation(void) {
    const pb_edge duplicate[2] = {{0u, 1u, 1.0}, {1u, 0u, 0.5}};
    pb_config config;
    pb_engine *engine = NULL;
    pb_config_init(&config);
    config.batch_size = 2u;
    assert(pb_engine_create(2u, duplicate, 2u, &config, NULL, NULL, &engine) == PB_EINVAL);
    assert(engine == NULL);
    config.struct_size = 0u;
    assert(pb_engine_create(2u, duplicate, 1u, &config, NULL, NULL, &engine) == PB_EVERSION);
}

static void test_borrowed_edges_and_direct_subopt(void) {
    pb_edge borrowed[1] = {{1u, 0u, 4.0}};
    pb_config config;
    pb_engine *engine = NULL;
    uint32_t subopt_truth[15];
    uint32_t records[10];
    uint32_t labels[10];
    pb_batch batch;
    pb_stats stats;
    pb_status status;
    size_t index;

    pb_config_init(&config);
    config.method = PB_METHOD_ONLINE;
    config.cda = PB_CDA_NONE;
    config.batch_size = 2u;
    assert(pb_engine_create_borrowed(
               2u, borrowed, 1u, &config, NULL, NULL, &engine) == PB_OK);
    assert(borrowed[0].u == 0u && borrowed[0].v == 1u && borrowed[0].weight == 1.0);
    pb_engine_free(engine);

    for (index = 0u; index < 15u; index++) {
        subopt_truth[index] = 0u;
    }
    engine = NULL;
    assert(pb_engine_create_subopt(
               15u, NULL, 0u, 10u, 42u, 0u, subopt_truth, &engine) == PB_OK);
    while ((status = pb_engine_next_batch(engine, records, 10u, &batch)) == PB_OK) {
        for (index = 0u; index < batch.len; index++) {
            labels[index] = 0u;
        }
        assert(pb_engine_submit_partition(engine, labels, batch.len) == PB_OK);
    }
    assert(status == PB_DONE);
    assert(pb_engine_get_stats(engine, &stats) == PB_OK);
    assert(stats.query_count == 2u);
    assert(stats.discovered_matches == 105u);
    pb_engine_free(engine);
}

int main(void) {
    test_validation();
    test_borrowed_edges_and_direct_subopt();
    test_budget_and_state();
    test_members();
    test_snapshot_replay();
    test_all_methods();
    puts("C core tests passed");
    return 0;
}
