# Entity Resolution via Batched Oracle Queries

This repository contains the source code accompanying the paper:

**Entity Resolution via Batched Oracle Queries**

The code supports the experimental evaluation presented in the paper and allows reproducing the reported results.

---

## Contents

The repository includes implementations of the proposed and baseline algorithms, as well as scripts for computing theoretical bounds used in the analysis.

---

## Computing Bounds on Φ

To compute the upper and lower bounds of the function Φ defined in Equation (3) of the paper, for all datasets reported in Table 2, run:

```bash
python compute_bounds_phi.py
```

---

## Running Batched Entity Resolution Algorithms

To run the batched entity resolution algorithms **pERbacco**, **pERbac**, and **Online** on a given dataset with batch size equal to 10, execute:

```bash
python multiple_pERbacco.py --dataset datasetname
```

where `datasetname` specifies the target dataset.

**Note:** For `datasetname = "cora"` it should take a few minutes.

---

## Running Experiments on All Datasets

To apply **pERbacco**, **pERbac**, and **Online** to all datasets reported in Table 2 with batch size equal to 10, run:

```bash
python multiple_pERbacco.py --dataset all
```

**Note:** This experiment is computationally expensive and not parallelized. Running it may take several days.

---

## Running Individual Algorithms

The script `perbacco.py` allows running individual algorithms with fine-grained control over parameters.

### OpenAI-Backed Oracle

To run the pERbacco query strategy with a real OpenAI oracle instead of the simulated ground-truth oracle:

1. Set `OPENAI_API_KEY` in `llm_config_local.py` or export it in the shell.
2. Run:

```bash
python3 perbacco.py \
  --dataset cora \
  --batch_size 10 \
  --alg_community louvain \
  --lambda_w 0.05 \
  --mu_benefit brmean \
  --optimal False \
  --synth_precision False \
  --oracle_backend openai \
  --openai_model gpt-5-mini \
  --prompt_mode zero-shot \
  --max_llm_calls 50
```

The script prints a preflight token estimate before the first API call, then reports progress, recall, precision, and token usage after each query.

To warm up the state with the first 5 batches resolved from ground truth and only call the LLM afterwards, add:

```bash
  --skip-batches 5
```

### pERbac

To run **pERbac** on a given dataset with batch size `batch_size`, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "False" \
  --lambda_w "False" \
  --mu_benefit "brmean" \
  --optimal "False" \
  --synth_precision "False"
```

### Online

To run **Online** on a given dataset with batch size `batch_size`, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "False" \
  --lambda_w "False" \
  --mu_benefit "brmax" \
  --optimal "False" \
  --synth_precision "False"
```

### pERbacco (Proposed Method)

To run **pERbacco** with the parameter configuration described in the paper, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "louvain" \
  --lambda_w "0.05" \
  --mu_benefit "brmean" \
  --optimal "False" \
  --synth_precision "False"
```

### Suboptimal

To run **SubOptimal** with batch size `batch_size`, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "False" \
  --lambda_w "False" \
  --mu_benefit "brmax" \
  --optimal "True" \
  --synth_precision "False"
```

---

## Plot Generation

To generate the plots corresponding to Figure 3 and Figure 4 of the paper for a given dataset and batch size, run:

```bash
python make_plot.py --dataset datasetname --batch_size batch_size
```

To reproduce Figure 3(a) for Cora and add one LLM-backed line:

```bash
python3 test-fig3-cora.py \
  --include-llm \
  --openai-model gpt-5-mini \
  --prompt-mode few-shot \
  --max-llm-calls 50
```

---

## Reproducibility

All experiments reported in the paper can be reproduced using the scripts provided in this repository with the same parameter settings described in the experimental evaluation section.
