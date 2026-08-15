---
hide:
  - toc
---

<div class="perbacco-hero" markdown>

<p class="perbacco-kicker">C99 engine · Python interface · four schedulers</p>

# Ask the oracle a better batch.

<p class="perbacco-lede">
pERbacco selects the next bounded set of current entities to send to an
entity-resolution oracle, using a compact dependency-free C engine and a small
Python API.
</p>

<div class="perbacco-actions">
  <a class="perbacco-button" href="getting-started/">Get started</a>
  <a class="perbacco-button perbacco-button--quiet" href="reference/python/">View API</a>
</div>

</div>

<div class="perbacco-section" markdown>

## A complete local run

This executable example uses known labels as the oracle. It makes no API call
and exercises the same `Oracle` protocol used by remote models.

```python
--8<-- "examples/python/quickstart.py"
```

</div>

<div class="perbacco-section" markdown>

## Choose the policy, keep the loop

<div class="perbacco-grid">
  <div class="perbacco-card" markdown>
  ### pERbacco
  Mean-benefit scheduling preceded by recursive heavy-community batches. The
  main method from the paper.
  </div>
  <div class="perbacco-card" markdown>
  ### pERbac
  Mean-benefit scheduling without community preprocessing. Useful when the
  similarity graph has no meaningful large communities.
  </div>
  <div class="perbacco-card" markdown>
  ### Online
  Selects using the current maximum edge probability. A direct online baseline
  from the paper.
  </div>
  <div class="perbacco-card" markdown>
  ### SubOpt
  A ground-truth-only comparison baseline. It cannot be used with a real API
  oracle because scheduling itself requires hidden labels.
  </div>
</div>

</div>

<div class="perbacco-section" markdown>

## One state transition at a time

<div class="perbacco-flow" role="img" aria-label="Similarity graph flows to scheduler, oracle, and state update">
  <div class="perbacco-flow__step">Similarity graph</div>
  <div class="perbacco-flow__arrow" aria-hidden="true">→</div>
  <div class="perbacco-flow__step">Scheduler</div>
  <div class="perbacco-flow__arrow" aria-hidden="true">→</div>
  <div class="perbacco-flow__step">Oracle partition</div>
  <div class="perbacco-flow__arrow" aria-hidden="true">→</div>
  <div class="perbacco-flow__step">Entity-state update</div>
</div>

The oracle answers with a **complete partition of the current entities**, not a
list of pairwise decisions. The engine validates the answer before mutating
state, contracts matches, records non-matches, updates affected benefits, and
then selects again. See [core concepts](concepts.md) for the data model.

</div>

!!! note "Research implementation status"

    Table III is reproduced exactly. Current Figure 4 artifacts are maintained
    C implementation results—not an exact reconstruction of the published
    Python/NetworkX curves. The [research-status page](research-status.md)
    explains what changed and what remains unresolved.
