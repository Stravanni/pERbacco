from __future__ import annotations

import importlib.util
import unittest

from perbacco.communities import external_heavy_communities

from perbacco import CDA, Engine, EngineConfig, Graph, Method

HAS_IGRAPH = importlib.util.find_spec("igraph") is not None
HAS_LEIDEN = importlib.util.find_spec("leidenalg") is not None


class CommunityTests(unittest.TestCase):
    def test_builtin_louvain_runs_coarse_levels(self) -> None:
        # The first Louvain level consists of four five-node communities.
        # Only the contraction phase can merge them into two heavy ten-node
        # communities accepted by the batch-size threshold.
        edges = []
        for community in range(4):
            nodes = range(community * 5, (community + 1) * 5)
            edges.extend((left, right, 1.0) for left in nodes for right in nodes if left < right)
        for community in range(3):
            left_nodes = range(community * 5, (community + 1) * 5)
            right_nodes = range((community + 1) * 5, (community + 2) * 5)
            edges.extend((left, right, 0.4) for left in left_nodes for right in right_nodes)
        graph = Graph.from_edges(edges, nodes=range(20))
        config = EngineConfig(
            method=Method.PERBACCO,
            cda=CDA.LOUVAIN,
            batch_size=10,
            seed=42,
            max_queries=1,
            lambda_w=0.01,
            louvain_resolution=0.75,
        )
        with Engine(graph, config) as engine:
            self.assertEqual(engine.stats.community_records, 20)

    @unittest.skipUnless(HAS_IGRAPH, "optional igraph extra is not installed")
    def test_louvain_is_seeded(self) -> None:
        graph = self._graph()
        first = external_heavy_communities(
            graph, algorithm="louvain", lambda_w=0.1, batch_size=3, seed=42
        )
        second = external_heavy_communities(
            graph, algorithm="louvain", lambda_w=0.1, batch_size=3, seed=42
        )
        self.assertEqual(first, second)

    @unittest.skipUnless(HAS_IGRAPH and HAS_LEIDEN, "optional Leiden extra is not installed")
    def test_leiden_is_seeded(self) -> None:
        graph = self._graph()
        first = external_heavy_communities(
            graph, algorithm="leiden", lambda_w=0.1, batch_size=3, seed=42
        )
        second = external_heavy_communities(
            graph, algorithm="leiden", lambda_w=0.1, batch_size=3, seed=42
        )
        self.assertEqual(first, second)

    @staticmethod
    def _graph() -> Graph:
        edges = []
        for left in range(30):
            for right in range(left + 1, 30):
                same = left // 10 == right // 10
                edges.append((left, right, 1.0 if same else 0.01))
        return Graph.from_edges(edges)


if __name__ == "__main__":
    unittest.main()
