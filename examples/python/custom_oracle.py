"""Implement the provider-neutral oracle protocol without network access."""

from collections.abc import Hashable, Sequence

from perbacco import CDA, Engine, EngineConfig, EntityView, Graph, Method, OracleResult, run


class CanonicalFieldOracle:
    """Test oracle that clusters records by a canonical_id attribute."""

    def partition(self, entities: Sequence[EntityView]) -> OracleResult:
        labels: list[Hashable] = []
        for entity in entities:
            values = {record["canonical_id"] for record in entity.records}
            if len(values) != 1:
                raise ValueError("a current entity contains conflicting canonical IDs")
            labels.append(next(iter(values)))
        return OracleResult(tuple(labels), {"input_tokens": 0, "output_tokens": 0})


def main() -> None:
    records = {
        "r1": {"name": "Ada Lovelace", "canonical_id": "ada"},
        "r2": {"name": "A. Lovelace", "canonical_id": "ada"},
        "r3": {"name": "Grace Hopper", "canonical_id": "grace"},
    }
    graph = Graph.from_edges(
        [("r1", "r2", 0.98), ("r1", "r3", 0.12), ("r2", "r3", 0.10)],
        records=records,
    )
    config = EngineConfig(method=Method.PERBAC, cda=CDA.NONE, batch_size=3)

    with Engine(graph, config) as engine:
        result = run(engine, CanonicalFieldOracle())

    print(f"queries={result.stats.query_count}, finished={result.stats.finished}")


if __name__ == "__main__":
    main()
