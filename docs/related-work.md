# Related-work audit

This audit explains why the selectable scheduler enum contains the four methods
from the pERbacco paper rather than attaching unrelated names to behaviorally
different workflows.

The supplied related-work papers were reviewed as behavioral references:

- BatchER, *Cost-Effective In-Context Learning for Entity Resolution: A Design
  Space Exploration* (ICDE 2024).
- LLM-CER, *In-context Clustering-based Entity Resolution with Large Language
  Models: A Design Space Exploration* (arXiv:2506.02509v1).

The public BatchER repository was inspected at commit
`b55ba123834fd15163f6f2286d6d03299893ec4d`. It has no license file, so no
source was copied or redistributed. Its random, similar, and diverse batching
strategies batch already-selected pairwise ER questions into one prompt. They
do not select a bounded record set for a complete-partition oracle and therefore
are transport/prompt policies, not alternatives to the next-best-record-batch
schedulers exposed by this library. Its covering method similarly selects
in-context demonstrations for those pair questions.

The LLM-CER paper defines a compatible-looking Next Record Set Creation (NRS)
stage, but it belongs to a different end-to-end workflow: external blocking,
feature embeddings, elbow/K-means clustering, one-pass consumption of records,
misclustering guardrails, and hierarchical result merging. No public source
repository is linked by the supplied paper or was found during the audit. A
faithful NRS implementation would broaden this library beyond its declared
input—a precomputed similarity graph—and would not be a progressive scheduler
over current inferred entities.

Accordingly, the selectable V1 methods are the pERbacco paper's four schedulers:
pERbacco, pERbac, Online, and SubOpt. The provider-neutral oracle and complete
entity views deliberately leave room for BatchER-style prompt multiplexing or
LLM-CER-style guardrails without coupling either policy to the C scheduler.
Adding those end-to-end workflows requires a separate specification for
blocking, feature extraction, noisy-answer reconciliation, and evaluation; it
is not represented as a misleading scheduler alias.
