# Contributing and verification

## Source layout

```text
include/perbacco.h       public C ABI
src/perbacco.c           zero-dependency C99 engine
python/perbacco/         ctypes wrapper, oracles, CLI, experiments
examples/                canonical executable documentation sources
tests/                   C, Python, ABI, oracle, and docs acceptance tests
docs/                    MkDocs source
```

The top-level historical Python scripts are compatibility/research references.
New scheduler work belongs in the C engine with wrapper coverage.

## Build and test

```bash
make clean
make
make test
```

The strict C flags include C99, warnings as errors, shadowing, conversions, and
strict prototypes. `make test` runs the native unit binary and Python discovery
suite. Documentation acceptance tests execute the local quickstart, custom fake
oracle, snapshot/restore flow, and compiled C quickstart. Network entry points
are patched to fail during those examples.

## Sanitizers

```bash
make sanitize
```

This rebuilds the native unit suite with AddressSanitizer and
UndefinedBehaviorSanitizer, runs it, and removes the instrumented build. Run a
normal `make` afterward if you need the shared library.

## Lint and formatting

```bash
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Ruff configuration lives in `pyproject.toml`. Do not mechanically rewrite the
inactive historical research class unless its exclusion is intentionally
removed.

## Documentation

Documentation dependencies are pinned separately from package dependencies:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

The strict build treats missing pages, unresolved links/anchors, omitted
Markdown pages, snippet drift, import errors, and Markdown warnings as failures.
Python reference pages are generated from docstrings; the C manual is reviewed
against the stable header.

The project-level `site_url` includes the `/pERbacco/` path as recommended by
the [Material for MkDocs configuration
guide](https://squidfunk.github.io/mkdocs-material/creating-your-site/), keeping
canonical and relative asset URLs correct below a domain root.

Canonical examples live under `examples/` and are included with the snippets
extension. Edit the executable file rather than copying a code block into a
page. Every documentation test must remain unable to reach a live API.

## Packaging

```bash
python -m pip install build
python -m build
python -m pip install --force-reinstall dist/*.whl
```

The build hook invokes `make all` and places the platform shared library inside
the wheel. The wheel is Python-ABI-independent (`py3-none-<platform>`) but not
platform-independent. Verify imports from outside the checkout so a local
`PYTHONPATH` cannot mask a missing packaged library.

## CI and Pages

The CI workflow builds and tests Linux/macOS with supported Python versions,
runs sanitizers, and installs the produced wheel. The documentation workflow:

- runs examples and `mkdocs build --strict` on pull requests;
- uploads no deployable site on pull requests;
- on `main`, configures Pages, uploads the generated site artifact, and deploys
  through the protected `github-pages` environment.

Pages uses GitHub's artifact deployment actions rather than committing a
generated `gh-pages` tree. The repository setting must select **GitHub Actions**
as the Pages source. The workflow follows GitHub's [custom Pages workflow
guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Contribution workflow

1. Create a focused branch.
2. Add a deterministic test for behavior changes.
3. Update Python docstrings, the handwritten C reference, and a canonical
   example when the public contract changes.
4. Run the complete verification commands above.
5. Open a pull request describing behavioral and research-result impact.

Do not commit API keys, JSONL oracle journals, local paper PDFs, generated
experiment artifacts, native build products, wheel/sdist output, MkDocs `site/`,
or temporary renders.

## License

New library and documentation contributions are accepted under the repository's
[BSD-3-Clause license](https://github.com/Stravanni/pERbacco/blob/main/LICENSE).
Related-work repositories without a compatible license remain behavioral
references only; do not copy their source.
