"""Five-minute pERbacco run with a local ground-truth oracle."""

from perbacco import CDA, Engine, EngineConfig, Graph, GroundTruthOracle, Method, run


def main() -> None:
    truth = {"a1": "alice", "a2": "alice", "b1": "bob", "b2": "bob"}
    records = {
        "a1": {"name": "Alice Rossi", "city": "Rome"},
        "a2": {"name": "A. Rossi", "city": "Roma"},
        "b1": {"name": "Bob Smith", "city": "London"},
        "b2": {"name": "Robert Smith", "city": "London"},
    }
    graph = Graph.from_edges(
        [
            ("a1", "a2", 0.96),
            ("b1", "b2", 0.93),
            ("a1", "b1", 0.08),
            ("a2", "b2", 0.06),
        ],
        records=records,
    )
    config = EngineConfig(
        method=Method.PERBACCO,
        cda=CDA.LOUVAIN,
        batch_size=3,
        max_queries=10,
    )

    with Engine(graph, config) as engine:
        result = run(
            engine,
            GroundTruthOracle(truth),
            total_truth_matches=2,
        )

    print(f"queries={result.stats.query_count}")
    print(f"matches={result.stats.discovered_matches}")
    print(f"recall={result.events[-1].recall:.1f}")


if __name__ == "__main__":
    main()
