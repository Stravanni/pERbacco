# Verification record

This page records the release-branch checks run on 2026-08-16. Post-merge Pages
deployment is intentionally listed separately because the release plan forbids
publishing before the pull request is reviewed and merged.

## Local environment

- branch: `codex/library-docs-pages`
- macOS Darwin 25.5.0, arm64
- Apple Clang 21.0.0
- Python 3.14.3

## Native and Python suites

```bash
make test
uv run python -m unittest discover -s tests -p 'test_*.py' -v
```

- strict C99 core suite: pass
- standard-library environment: 24 Python tests discovered; 22 pass and 2
  optional-backend tests skip
- project environment with igraph and Leiden: 24 Python tests pass
- ABI export checks: pass
- deterministic randomized scheduler checks: pass
- exact-budget and outstanding-batch checks: pass
- snapshot replay, immutable-input identity rejection, and SubOpt ranking checks:
  pass
- fake Responses/Chat structured-output and retry accounting checks: pass
- implementation-fingerprinted experiment-cache rejection: pass
- local Python, custom fake-oracle, snapshot, and compiled C documentation
  examples: pass with live network entry points blocked

## Sanitizers

```bash
make sanitize
```

The C unit suite passes with AddressSanitizer and UndefinedBehaviorSanitizer,
warnings as errors, and frame pointers enabled.

## Static and documentation checks

```bash
uv run ruff check .
uv run ruff format --check .
git diff --check
uv run --with-requirements requirements-docs.txt mkdocs build --strict
```

All commands pass. Generated-site inspection confirms:

- canonical URLs retain the `/pERbacco/` repository path;
- home, quickstart, Python reference, C reference, and research-status pages
  exist;
- the executable Python/C snippets are present and highlighted;
- navigation, heading anchors, search, and code-copy controls are generated;
- social metadata and its validated image are present;
- responsive single-column rules and visible `:focus-visible` styling exist;
- no key-shaped `sk-…` or bearer-token value is present.

## Packaging

```bash
uv build
```

The source distribution and native macOS wheel build successfully. The source
distribution contains MkDocs configuration, pinned docs requirements, CSS,
template override, social asset, canonical examples, public header, source, and
tests. The wheel installs into a fresh temporary Python 3.14 environment and
runs an engine from `/tmp`, proving that the packaged native library—not the
checkout—is loaded.

## Experiment-result status

Table III remains the only artifact labeled exact. The release documentation
now labels current Figure 4 output “maintained C implementation results” and
records NetworkX-versus-C Louvain, deterministic ties, exact budgets, benefit
invalidation, live representatives, SubOpt behavior, and unresolved large curve
gaps. Experiment cache manifests now fingerprint result-affecting code, the
Python runtime, and optional graph/community backends so old curves cannot be
silently reused after an engine or dependency change.

## Pending after review and merge

1. Confirm the repository Pages source is **GitHub Actions**.
2. Merge the single release pull request into `main`.
3. Wait for the `docs` workflow's `github-pages` deployment.
4. Verify the workflow-reported public URL and these routes:
   `/`, `/getting-started/`, `/reference/python/`, `/reference/c/`, and
   `/research-status/`.

Until those steps complete, the public deployment is not claimed complete.
