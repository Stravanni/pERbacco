# Python API reference

The objects below are the supported exports from `perbacco`. Signatures and
descriptions are rendered directly from the maintained package, so a strict
documentation build fails when imports drift.

## Graph

::: perbacco.Graph
    options:
      members:
        - from_edges
        - from_native_edges
        - id_to_dense

## Configuration and enums

::: perbacco.EngineConfig

::: perbacco.Method

::: perbacco.CDA

::: perbacco.Status

::: perbacco.BatchKind

## Engine lifecycle

::: perbacco.Engine
    options:
      members:
        - restore
        - next_batch
        - submit_partition
        - group_members
        - stats
        - snapshot
        - close

::: perbacco.Batch

::: perbacco.Statistics

::: perbacco.PerbaccoError

## Oracle types

::: perbacco.EntityView

::: perbacco.OracleResult

::: perbacco.Oracle

::: perbacco.GroundTruthOracle
    options:
      members:
        - partition

::: perbacco.OpenAICompatibleOracle
    options:
      members:
        - openai
        - openrouter
        - partition

::: perbacco.OracleProtocolError

## Runner

::: perbacco.run

::: perbacco.RunEvent

::: perbacco.RunResult
