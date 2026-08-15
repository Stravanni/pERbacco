#include "perbacco.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int fail(pb_engine *engine, pb_status status) {
    const char *detail = pb_engine_last_error(engine);
    fprintf(stderr, "pERbacco: %s\n", detail != NULL ? detail : pb_strerror(status));
    pb_engine_free(engine);
    return EXIT_FAILURE;
}

int main(void) {
    pb_edge edges[] = {
        {0u, 1u, 0.96},
        {2u, 3u, 0.93},
        {0u, 2u, 0.08},
        {1u, 3u, 0.06},
    };
    const uint32_t truth[] = {0u, 0u, 1u, 1u};
    pb_record_id records[3];
    uint32_t labels[3];
    pb_config config;
    pb_engine *engine = NULL;
    pb_stats stats;
    pb_batch batch;

    pb_config_init(&config);
    config.method = PB_METHOD_PERBAC;
    config.cda = PB_CDA_NONE;
    config.batch_size = 3u;
    config.max_queries = 10u;

    pb_status status = pb_engine_create(
        4u,
        edges,
        sizeof(edges) / sizeof(edges[0]),
        &config,
        NULL,
        truth,
        &engine);
    if (status != PB_OK) {
        return fail(engine, status);
    }

    for (;;) {
        status = pb_engine_next_batch(engine, records, 3u, &batch);
        if (status == PB_DONE || status == PB_BUDGET_EXHAUSTED) {
            break;
        }
        if (status != PB_OK) {
            return fail(engine, status);
        }
        for (size_t index = 0u; index < batch.len; ++index) {
            labels[index] = truth[batch.records[index]];
        }
        status = pb_engine_submit_partition(engine, labels, batch.len);
        if (status != PB_OK) {
            return fail(engine, status);
        }
    }

    status = pb_engine_get_stats(engine, &stats);
    if (status != PB_OK) {
        return fail(engine, status);
    }
    printf("queries=%" PRIu64 ", matches=%" PRIu64 "\n",
           stats.query_count,
           stats.discovered_matches);
    pb_engine_free(engine);
    return EXIT_SUCCESS;
}
