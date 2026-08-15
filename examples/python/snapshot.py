"""Checkpoint an engine between oracle calls and restore it later."""

from perbacco import CDA, Engine, EngineConfig, Graph, Method


def main() -> None:
    truth = {0: "x", 1: "x", 2: "y", 3: "y"}
    graph = Graph.from_edges([(0, 1, 0.95), (2, 3, 0.91), (0, 2, 0.10), (1, 3, 0.08)])
    config = EngineConfig(method=Method.ONLINE, cda=CDA.NONE, batch_size=3)

    with Engine(graph, config) as engine:
        batch = engine.next_batch()
        assert batch is not None
        engine.submit_partition([truth[record] for record in batch.records])
        checkpoint = engine.snapshot()
        expected = engine.next_batch()

    with Engine.restore(graph, checkpoint, config) as restored:
        actual = restored.next_batch()
        assert actual == expected

    print(f"restored_next_batch={actual.records if actual else None}")


if __name__ == "__main__":
    main()
