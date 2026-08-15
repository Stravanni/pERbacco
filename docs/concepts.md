# Core concepts

## Similarity graph

The library begins after blocking and similarity computation. Its immutable
input is an undirected weighted graph:

- one node per record;
- one candidate edge for each pair worth considering;
- a finite non-negative similarity weight per edge.

The engine normalizes edge weights by the graph maximum. A missing edge means
“not a candidate,” not necessarily a confirmed non-match. Feature extraction,
blocking, similarity model training, and noisy-label repair are outside the
library boundary.

## Entity components

At startup, every record is a singleton entity component. When the oracle says
two current entities match, the engine contracts their components. Each later
query contains one **representative** per current component, while the oracle
receives every member record through an `EntityView`.

This distinction matters: batch size counts current entities, not raw records.
A representative may stand for many records after several merges.

## Batches and partitions

A `Batch` is an ordered set of current entity representatives chosen for one
oracle query. The oracle must return one label per batch position. Equal labels
mean merge; different labels mean confirmed separation for that query.

For a batch `[A, B, C, D]`, the labels `[0, 0, 1, 2]` express the complete
partition `{A, B}`, `{C}`, `{D}`. Label names have no meaning beyond equality.
The engine validates the whole partition before it changes state.

## Candidate benefits

The schedulers rank live cross-component edges by an expected benefit derived
from normalized similarity and current component size. GreedyHS considers a
bounded high-benefit prefix (`top_k`, 1000 in the paper configuration) when
constructing a record batch. Benefits incident to a changed component are
recomputed after every accepted partition.

## Community phase

pERbacco can first identify recursively dense, sufficiently large communities
and spend community batches within them. `lambda_w` controls the normalized
internal-density threshold. After that phase, scheduling continues on the
current residual graph. The built-in backend is deterministic weighted Louvain;
callers may instead supply disjoint external communities.

## Query budgets

`max_queries` is an exact cap on **submitted** oracle partitions. Zero means no
cap. Selection stops before issuing query `max_queries + 1`; there is no
inclusive-loop extra query. When the budget ends a nonterminal run,
`stats.finished` remains false.

Snapshots are valid between queries, after a partition has been submitted and
before the next batch is selected. They contain a versioned query journal, not
the immutable graph, so restore requires the same graph and configuration.

## Scheduler comparison

| Scheduler | Ranking signal | Community preprocessing | Intended role |
| --- | --- | --- | --- |
| pERbacco | mean benefit + temperature | yes | primary progressive method |
| pERbac | mean benefit | no | ablation/baseline |
| Online | maximum edge probability | no | online baseline |
| SubOpt | hidden truth-graph benefit | no | experimental upper-bound comparison |

SubOpt is not deployable against an API oracle: it uses ground truth while
choosing the batch, not only while answering it.
