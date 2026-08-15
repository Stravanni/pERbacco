# C guide

The native library is a single C99 translation unit with no dependency beyond
the C standard library and `libm`. Its ABI is declared in
[`include/perbacco.h`](https://github.com/Stravanni/pERbacco/blob/main/include/perbacco.h).

## Build

```bash
make
```

Artifacts:

- `build/libperbacco.a` — static library
- `build/libperbacco.so` — Linux shared library
- `build/libperbacco.dylib` — macOS shared library

Compile the canonical static example:

```bash
cc -std=c99 -Iinclude examples/c/quickstart.c build/libperbacco.a -lm -o quickstart
./quickstart
```

For dynamic linking, replace the archive with `-Lbuild -lperbacco` and configure
the platform loader path when installing the shared library outside a standard
location.

## Complete example

The acceptance suite compiles and runs this exact source against the public
header and static library:

```c
--8<-- "examples/c/quickstart.c"
```

## Ownership model

`pb_engine_create()` validates and copies edges, truth labels, and external
communities. The caller may release those inputs after successful construction.
The returned opaque `pb_engine *` owns all mutable scheduling state and must be
released exactly once with `pb_engine_free()`; passing `NULL` to free is safe.

`pb_engine_create_borrowed()` is the large-graph path. It normalizes and sorts
the caller's edge array **in place**, then borrows it. The array must be writable,
must remain alive, and must not change until the engine is freed. Truth labels
and external communities are still copied. The Python wrapper retains its
borrowed array automatically.

## Configuration

Always call `pb_config_init()` before modifying fields. The `struct_size` and
`abi_version` fields make layout mismatches fail with `PB_EVERSION` instead of
being interpreted silently.

```c
pb_config config;
pb_config_init(&config);
config.method = PB_METHOD_PERBACCO;
config.cda = PB_CDA_LOUVAIN;
config.batch_size = 10u;
config.max_queries = 300u;
```

Zero `max_queries` means unlimited. `batch_size` must be between 2 and
`node_count`; `top_k` must be positive; `lambda_w` must be in `[0, 1]`.

## Query lifecycle

1. Call `pb_engine_next_batch()` with capacity at least `config.batch_size`.
2. On `PB_OK`, inspect `pb_batch.kind`, `pb_batch.len`, and the caller-owned
   records buffer referenced by `pb_batch.records`.
3. Ask the oracle for a complete partition and align one `uint32_t` label with
   each record. Only equality of label values matters.
4. Call `pb_engine_submit_partition()` once.
5. Repeat until `PB_DONE` or `PB_BUDGET_EXHAUSTED`.

Only one batch may be outstanding. Calling `next_batch` twice or snapshotting
before submission returns `PB_ESTATE`. A partition is validated before state is
mutated.

## Status handling

Treat `PB_OK`, `PB_DONE`, and `PB_BUDGET_EXHAUSTED` as normal control flow. All
negative values are errors. `pb_strerror()` returns a static generic message;
`pb_engine_last_error()` may provide a more specific engine diagnostic. Copy a
diagnostic before freeing the engine if it must outlive the engine.

```c
status = pb_engine_next_batch(engine, records, config.batch_size, &batch);
if (status < 0) {
    fprintf(stderr, "%s\n", pb_engine_last_error(engine));
}
```

## Snapshots

Snapshots are versioned, checksummed query journals. They exclude the immutable
graph and configuration. First ask for the required size, allocate the buffer,
then serialize:

```c
size_t size = 0u;
void *snapshot = NULL;

if (pb_engine_snapshot_size(engine, &size) == PB_OK) {
    snapshot = malloc(size);
    if (snapshot != NULL) {
        size_t written = 0u;
        status = pb_engine_snapshot(engine, snapshot, size, &written);
    }
}
```

`pb_engine_restore()` copies the supplied immutable inputs, checks ABI/snapshot
versions plus graph, truth, community, and configuration hashes, and replays the
journal. It returns a new engine; it never takes ownership of the snapshot
buffer.

See the [handwritten C API reference](reference/c.md) for every structure and
function.
