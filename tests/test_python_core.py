from __future__ import annotations

import random
import unittest

from perbacco.oracle import GroundTruthOracle
from perbacco.runner import run

from perbacco import CDA, Engine, EngineConfig, Graph, Method, PerbaccoError, Status


def complete_graph(size: int, truth: dict[int, int]) -> Graph:
    edges = []
    for left in range(size):
        for right in range(left + 1, size):
            weight = 1.0 if truth[left] == truth[right] else 0.05 + (left + right) / 1000
            edges.append((left, right, weight))
    return Graph.from_edges(edges)


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
        self.graph = complete_graph(6, self.truth)

    def test_all_methods_reach_complete_recall(self) -> None:
        for method in Method:
            with self.subTest(method=method):
                config = EngineConfig(
                    method=method,
                    cda=CDA.NONE,
                    batch_size=3,
                    top_k=100,
                    seed=42,
                )
                with Engine(
                    self.graph,
                    config,
                    truth=self.truth if method == Method.SUBOPT else None,
                ) as engine:
                    result = run(engine, GroundTruthOracle(self.truth), total_truth_matches=4)
                    self.assertEqual(result.stats.discovered_matches, 4)
                    self.assertEqual(result.events[-1].recall, 1.0)

    def test_snapshot_replays_the_next_batch(self) -> None:
        config = EngineConfig(method=Method.ONLINE, cda=CDA.NONE, batch_size=3, top_k=100)
        with Engine(self.graph, config) as engine:
            batch = engine.next_batch()
            assert batch is not None
            engine.submit_partition([self.truth[item] for item in batch.records])
            snapshot = engine.snapshot()
            expected = engine.next_batch()
            with Engine.restore(self.graph, snapshot, config) as restored:
                actual = restored.next_batch()
                self.assertEqual(expected, actual)

    def test_snapshot_rejects_different_external_communities(self) -> None:
        config = EngineConfig(
            method=Method.PERBACCO,
            cda=CDA.EXTERNAL,
            batch_size=3,
            top_k=100,
        )
        first = [[0, 1, 2], [3, 4, 5]]
        second = [[0, 1, 3], [2, 4, 5]]
        with Engine(self.graph, config, communities=first) as engine:
            snapshot = engine.snapshot()
        with self.assertRaises(PerbaccoError) as raised:
            Engine.restore(self.graph, snapshot, config, communities=second)
        self.assertEqual(raised.exception.status, Status.VERSION)

    def test_budget_is_exact(self) -> None:
        config = EngineConfig(
            method=Method.PERBAC, cda=CDA.NONE, batch_size=3, top_k=100, max_queries=1
        )
        with Engine(self.graph, config) as engine:
            batch = engine.next_batch()
            assert batch is not None
            engine.submit_partition([self.truth[item] for item in batch.records])
            self.assertIsNone(engine.next_batch())
            self.assertEqual(engine.stats.query_count, 1)
            self.assertFalse(engine.stats.finished)

    def test_component_contraction_never_reschedules_a_known_nonmatch(self) -> None:
        # Query 1 learns 1 !~ 2 without a similarity edge between them.
        # Query 2 merges 0 ~ 1, migrating that non-match onto the existing
        # similarity edge (0, 2).  The migrated edge must leave the heap.
        graph = Graph.from_edges(
            ((1, 3, 100.0), (2, 3, 99.0), (0, 1, 90.0), (0, 2, 80.0)),
            nodes=(0, 1, 2, 3),
        )
        truth = {0: 0, 1: 0, 2: 1, 3: 2}
        config = EngineConfig(
            method=Method.PERBAC,
            cda=CDA.NONE,
            batch_size=3,
            top_k=100,
        )
        with Engine(graph, config) as engine:
            first = engine.next_batch()
            assert first is not None
            self.assertEqual(set(first.records), {1, 2, 3})
            engine.submit_partition([truth[item] for item in first.records])

            second = engine.next_batch()
            assert second is not None
            self.assertEqual(set(second.records), {0, 1, 2})
            engine.submit_partition([truth[item] for item in second.records])

            self.assertIsNone(engine.next_batch())
            self.assertTrue(engine.stats.finished)
            self.assertEqual(engine.stats.candidate_edges, 0)

    def test_outstanding_batch_is_enforced(self) -> None:
        config = EngineConfig(method=Method.PERBAC, cda=CDA.NONE, batch_size=3)
        with Engine(self.graph, config) as engine:
            self.assertIsNotNone(engine.next_batch())
            with self.assertRaises(PerbaccoError) as raised:
                engine.next_batch()
            self.assertEqual(raised.exception.status, Status.STATE)

    def test_deterministic_randomized_runs(self) -> None:
        for seed in range(10):
            randomizer = random.Random(seed)
            labels = {index: randomizer.randrange(4) for index in range(12)}
            graph = complete_graph(12, labels)
            for method in Method:
                curves = []
                for _repeat in range(2):
                    config = EngineConfig(
                        method=method,
                        cda=CDA.NONE,
                        batch_size=4,
                        top_k=25,
                        seed=seed,
                    )
                    with Engine(
                        graph,
                        config,
                        truth=labels if method == Method.SUBOPT else None,
                    ) as engine:
                        result = run(engine, GroundTruthOracle(labels))
                        counts = [
                            list(labels.values()).count(label) for label in set(labels.values())
                        ]
                        expected = sum(count * (count - 1) // 2 for count in counts)
                        self.assertEqual(result.stats.discovered_matches, expected)
                        curves.append(
                            [(event.query, event.discovered_matches) for event in result.events]
                        )
                self.assertEqual(curves[0], curves[1])

    def test_subopt_needs_no_similarity_edges(self) -> None:
        truth = {index: 0 for index in range(15)}
        graph = Graph.from_edges((), nodes=truth)
        config = EngineConfig(method=Method.SUBOPT, cda=CDA.NONE, batch_size=10)
        with Engine(graph, config, truth=truth) as engine:
            result = run(engine, GroundTruthOracle(truth), total_truth_matches=105)
        self.assertEqual(result.stats.query_count, 2)
        self.assertEqual(result.stats.discovered_matches, 105)
        self.assertEqual(result.events[-1].recall, 1.0)

    def test_subopt_paper_ranking_prefers_the_larger_truth_entity(self) -> None:
        truth = {index: 0 if index < 10 else 1 for index in range(22)}
        graph = Graph.from_edges((), nodes=truth)
        config = EngineConfig(
            method=Method.SUBOPT,
            cda=CDA.NONE,
            batch_size=10,
            top_k=1000,
            seed=42,
        )
        with Engine(graph, config, truth=truth) as engine:
            batch = engine.next_batch()
            assert batch is not None
            self.assertEqual(len(batch.records), 10)
            self.assertEqual({truth[record] for record in batch.records}, {1})


if __name__ == "__main__":
    unittest.main()
