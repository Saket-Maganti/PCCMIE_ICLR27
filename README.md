# Prompt Conditions Can Change Measured Intervention Effects in LLM Evaluation

Anonymous code and reproduction package for the paper **“Prompt Conditions Can Change Measured Intervention Effects in LLM Evaluation.”**

We study whether the measured effect of a **silent self-check intervention** changes when the surrounding prompt condition changes. The main experiments use MMLU-Pro with three prompt conditions—**Standard**, **Masked**, and **Metadata**—and compare self-check against no-self-check within the same task and prompt condition.

For Gemma-4-12B-it, the Standard-minus-Masked difference in self-check gain is **+15.33 percentage points** in the three-model discovery study and **+20.00 pp [11.82, 28.18]** in a separate pre-specified 300-task replication. DeepSeek shows little observed interaction under the same discovery design, while Llama is inconclusive. The magnitude also depends on scoring: semantic rescoring of the replication outputs gives **+15.67 pp [7.82, 23.52]**, whereas a separate constrained-decoding evaluation gives **−1.00 pp [−6.87, 4.87]**.

> Implementation schemas retain the stable keys `BASE`, `MASK`, and `META`; in the paper these correspond to **Standard**, **Masked**, and **Metadata**, respectively.

## Setup

CPU reproduction requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Reproduce the reported CPU analyses and regenerate the included figures:

```bash
python3 scripts/reproduce.py
python3 scripts/make_figures.py
```

The default reproduction uses the anonymous scored projections included in `data/`. It checks the frozen contrast families, simultaneous max-t intervals, and headline values against the expected results in `configs/analysis.json`.

## Main results

### Supporting MATH paired comparison

| Quantity | Result |
|---|---:|
| Paired outcomes | 3,840 |
| Correctness flips | 814 (21.20%) |
| Gains | 408 |
| Losses | 406 |
| Accuracy before | 45.260% |
| Accuracy after | 45.313% |
| Aggregate change | +0.052 pp |
| Tasks with at least one flip | 80.42% |

This comparison motivates paired analysis: the aggregate mean is almost unchanged even though many individual outcomes change.

### MMLU-Pro discovery

The discovery study uses 300 MMLU-Pro tasks per model.

| Model | Standard gain | Masked gain | Metadata gain | Standard − Masked |
|---|---:|---:|---:|---:|
| DeepSeek R1 8B | +11.00 [2.12, 19.88] | +11.33 [4.00, 18.67] | +10.33 [2.12, 18.54] | −0.33 [−10.65, 9.98] |
| Gemma 12B | +29.33 [20.63, 38.04] | +14.00 [7.23, 20.77] | +20.00 [11.54, 28.46] | **+15.33 [6.70, 23.96]** |
| Llama 3.1 8B | +4.67 [−3.39, 12.73] | +0.33 [−6.92, 7.59] | +2.33 [−5.69, 10.36] | +4.33 [−5.95, 14.62] |

Values are percentage points with simultaneous 95% confidence intervals.

DeepSeek shows similar self-check gains across the three prompt conditions and little observed interaction. Llama is inconclusive. Gemma shows the large interaction carried forward to a separate replication.

### Pre-specified Gemma replication

The replication uses a new 300-task MMLU-Pro set and a separately fixed analysis plan.

| Quantity | Estimate | Simultaneous 95% CI |
|---|---:|---:|
| Standard gain | +30.33 | [22.45, 38.21] |
| Masked gain | +10.33 | [4.37, 16.30] |
| Metadata gain | +22.67 | [15.52, 29.81] |
| Standard − Masked | **+20.00** | **[11.82, 28.18]** |
| Standard − Metadata | +7.67 | [−0.30, 15.63] |
| Masked − Metadata | −12.33 | [−20.17, −4.50] |

The pre-specified threshold for the primary Standard-minus-Masked interaction was +5 pp.

### Scoring and output-format sensitivity

| Analysis | Standard − Masked interaction | 95% interval | Evaluation setting |
|---|---:|---:|---|
| Exact-format accuracy | +20.00 pp | [11.82, 28.18] | Same outputs and task set as the replication |
| Semantic answer scoring | +15.67 pp | [7.82, 23.52] | Same outputs and task set |
| Answer-format compliance | +26.33 pp | [18.65, 34.02] | Same outputs and task set |
| Constrained decoding | −1.00 pp | [−6.87, 4.87] | Separate 300-task set; output restricted to answer labels |

For constrained decoding, the self-check gains are:

- Standard: −13.00 pp [−18.93, −7.07]
- Masked: −12.00 pp [−18.83, −5.17]
- Metadata: −12.33 pp [−18.99, −5.67]
- Standard − Metadata: −0.67 pp [−6.72, 5.39]
- Metadata − Masked: −0.33 pp [−5.10, 4.44]

The constrained-decoding experiment changes both the task set and output format, so it does not isolate a causal effect of scoring and is not a direct replication of the exact-format result.

## Data preparation

Benchmark questions and answer options are **not redistributed** in this repository. MMLU-Pro inputs are acquired locally from the pinned official source and materialized only under ignored `data/local/`.

The release pins:

- dataset: `TIGER-Lab/MMLU-Pro`
- revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`
- file: `data/test-00000-of-00001.parquet`
- SHA-256: `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`

Install the input-preparation dependencies and build the frozen request packages:

```bash
python3 -m pip install -r requirements-inputs.txt

python3 scripts/prepare_inputs.py acquire --accept-source-terms
python3 scripts/prepare_inputs.py prepare --accept-source-terms
```

`--accept-source-terms` records that the operator reviewed the upstream source terms; it is not a permission grant. The preparation step verifies the pinned dataset hash, frozen panel commitments, tokenizer chat-template hashes, and prompt-tokenization constraints before writing local request packages.

If tokenizer repositories require authentication, provide `HF_TOKEN`. To avoid model-repository access, use `--local-tokenizers-only` with a local tokenizer cache.

## Experiments

### Three-model MMLU-Pro discovery

The discovery study evaluates 300 MMLU-Pro tasks for each of:

- `google/gemma-4-12B-it`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `meta-llama/Llama-3.1-8B-Instruct`

under Standard, Masked, and Metadata prompts, with and without self-check. Three deterministic repeats are used as reproducibility checks, for **16,200 generations** in total. Statistical inference is task-level; repeated generations are not treated as additional independent observations.

Exact model revisions and runtime parameters are frozen in `configs/experiment_locks.json`.

Example zero-request preflight:

```bash
python3 scripts/run_r2_gpu.py \
  --root data/local/prepared/r2 \
  --model google/gemma-4-12B-it \
  --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
  --dry-run
```

For live generation, supply the endpoint and execution identity arguments shown by:

```bash
python3 scripts/run_r2_gpu.py --help
```

The canonical discovery transport is deterministic and does not send the frozen repetition seed to the endpoint; the seed remains part of row identity and output metadata.

### Pre-specified Gemma replication

The replication uses a separate 300-task MMLU-Pro panel with Gemma-4-12B-it. It contains **1,800 primary generations** plus **360 duplicate generations** used only for reproducibility checks.

Qualification preflight:

```bash
python3 scripts/run_confirmation_gpu.py \
  --phase qualification \
  --input-package data/local/prepared/confirmation \
  --preflight-only
```

Production remains gated on a passing qualification validation and an immutable start lock bound to the runtime manifest and execution identity. The gate utilities are exposed through:

```bash
python3 scripts/validate_gates.py confirmation-qualification --help
python3 scripts/validate_gates.py confirmation-lock --help
```

No preflight sends model requests.

### Constrained-decoding evaluation

The constrained-decoding study uses a separate 300-task Gemma panel and restricts generation to the available answer labels. Scoring is direct label equality.

Preflight:

```bash
python3 scripts/run_s1_gpu.py \
  --phase production \
  --package-root data/local/prepared/s1 \
  --preflight-only
```

The frozen execution sequence requires smoke, sequential invariance, concurrent invariance, and qualification checks before production. The resulting gate receipt is validated with:

```bash
python3 scripts/validate_gates.py s1 --help
```

This experiment changes both the evaluation set and output format, so it is **not** a direct replication of the exact-format result and does not isolate a causal effect of scoring.

## Scoring fresh model outputs

The repository includes the canonical scoring bridge for discovery and replication outputs.

Discovery:

```bash
python3 scripts/score_outputs.py r2 \
  --outputs outputs/r2/gemma/official.jsonl \
            outputs/r2/llama/official.jsonl \
            outputs/r2/deepseek/official.jsonl \
  --out data/local/scored/r2.csv
```

Replication:

```bash
python3 scripts/score_outputs.py confirmation \
  --outputs data/local/prepared/confirmation/runtime_outputs/confirmation_responses.jsonl \
  --out data/local/scored/confirmation.csv
```

Recompute the paper statistics from fresh scored outputs:

```bash
python3 scripts/reproduce.py \
  --r2-outcomes data/local/scored/r2.csv \
  --confirmation-outcomes data/local/scored/confirmation.csv
```

## Reproducibility notes

All primary comparisons are paired at the task level. The discovery study uses max-t simultaneous 95% confidence intervals over its reported contrast family; the pre-specified replication uses simultaneous intervals over six fixed contrasts.

The public projections use randomized opaque `cluster_id` values and a frozen anonymous `analysis_order`. No benchmark task-ID crosswalk, benchmark question text, answer options, or raw model responses are included in the tracked repository. `data/local/` is intentionally ignored.

The large Gemma interaction is confirmed on MMLU-Pro, but it is not claimed to be universal across models or evaluation settings. DeepSeek shows little observed prompt-by-intervention interaction in discovery, Llama remains inconclusive, and there is no independent second-benchmark replication of the 20 pp Gemma result. Semantic rescoring preserves a positive interaction on the same outputs, while the separate constrained-decoding evaluation does not reproduce it.

## Repository structure

```text
.
├── configs/                  # frozen experiment and analysis settings
├── data/                     # anonymous scored projections and source commitments
├── figures/                  # regenerated paper-facing plots
├── results/                  # reproduced summary
├── scripts/
│   ├── prepare_inputs.py     # local MMLU-Pro acquisition and request preparation
│   ├── run_r2_gpu.py         # three-model discovery generation
│   ├── run_confirmation_gpu.py
│   ├── run_s1_gpu.py         # constrained-decoding evaluation
│   ├── score_outputs.py      # canonical output scoring
│   ├── validate_gates.py     # qualification/start-gate validation
│   ├── reproduce.py          # statistical reproduction
│   └── make_figures.py
├── requirements.txt
├── requirements-inputs.txt
└── SHA256SUMS.txt
```

## Integrity

`SHA256SUMS.txt` records the intended public release files. To verify the distributed payload:

```bash
sha256sum -c SHA256SUMS.txt
```

## License

Repository code is released under the included MIT `LICENSE`. MMLU-Pro and model assets remain subject to their own upstream terms. The benchmark is acquired locally and is not redistributed by this repository.
