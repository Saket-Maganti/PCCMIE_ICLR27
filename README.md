# BlackwellBench

BlackwellBench measures how evaluator reference choices affect paired model outcomes and treatment-effect estimates. The evidence separates behavioral turnover, R2 discovery, prospective Gemma magnitude confirmation, and later endpoint and execution diagnostics.

## CPU reproduction

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/reproduce.py
python3 scripts/make_figures.py
```

The outcome projections retain randomly generated opaque `cluster_id` values and randomized CSV row order. R2, confirmation, and S1 rows carry a zero-based `analysis_order` ordinal per cluster so the frozen max-t bootstrap reproduces the canonical draws. Opaque task fingerprints in `data/input_panel_manifest.json` preserve that anonymous analysis order separately from the original request order; no benchmark task IDs or task-ID crosswalk are included. `data/manifest.json` freezes the projection hashes.

## MMLU-Pro inputs and source terms

The source lock pins `TIGER-Lab/MMLU-Pro` at revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`, file `data/test-00000-of-00001.parquet`, SHA-256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`. The local project copy had no standalone dataset license file, so I checked the official metadata at that exact revision: it declares MIT. The MMLU-Pro paper also reports an MIT license for the dataset. The project MIT license applies to this repository's code; the upstream dataset declaration is separate. See `data/source_license_evidence.json` and the [official pinned dataset revision](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be).

No benchmark questions or options are included. Before preparing inputs, review the current source card and its MIT notice at the [official pinned source](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be). The acknowledgement flag records that review; it does not grant permission or change the source terms. The downloaded source, task IDs, prompts, request ledgers, and generated outputs stay under ignored `data/local/`.

```bash
python3 -m pip install -r requirements-inputs.txt
python3 scripts/prepare_inputs.py acquire --accept-source-terms
python3 scripts/prepare_inputs.py prepare --accept-source-terms
```

Preparation verifies the pinned source hash and frozen content commitments before writing local R2, prospective confirmation, and S1 request packages. It also verifies tokenizer chat-template hashes and the R2 `[MASK]`/`[META]` full-chat token-count lock. Provide `HF_TOKEN` when model repositories require authenticated tokenizer access. Use `--local-tokenizers-only` to prevent model-repository access, or pass `--tokenizer-cache-dir` pointing to directories named `llama31_8b_it`, `deepseek_r1_distill_llama8b`, and `gemma4_12b_it`.

## GPU runner preflights and runs

The preflight modes validate local request packages and send zero model requests:

```bash
python3 scripts/run_r2_gpu.py --root data/local/prepared/r2 --model google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --dry-run
python3 scripts/run_confirmation_gpu.py --phase qualification --input-package data/local/prepared/confirmation --preflight-only
python3 scripts/run_s1_gpu.py --phase production --package-root data/local/prepared/s1 --preflight-only
```

R2 runs against an OpenAI-compatible vLLM endpoint with the frozen model revision, 8-way concurrency, and 8,192-token limit. Supply `--url`, `--host-id`, `--instance-id`, `--gpu-uuid`, and `--out` for a live run. The confirmation runner supports qualification and confirmation; confirmation stays gated on a passing qualification-validation file and an immutable start lock bound to the runtime manifest and GPU. S1 requires its non-scientific endpoint smoke to pass before qualification or production. No preflight performs inference.

R2 keeps each frozen seed in its row key and output metadata; the canonical R2 request does not send a seed parameter. For fresh outcome projections, supply all three complete R2 output ledgers and the confirmation ledger to the canonical scorer, then analyze them:

```bash
python3 scripts/score_outputs.py r2 --outputs outputs/r2/gemma/official.jsonl outputs/r2/llama/official.jsonl outputs/r2/deepseek/official.jsonl --out data/local/scored/r2.csv
python3 scripts/score_outputs.py confirmation --outputs data/local/prepared/confirmation/runtime_outputs/confirmation_responses.jsonl --out data/local/scored/confirmation.csv
python3 scripts/reproduce.py --r2-outcomes data/local/scored/r2.csv --confirmation-outcomes data/local/scored/confirmation.csv
```

Before confirmation production, run `scripts/validate_gates.py confirmation-qualification` on the complete 180-row qualification output, then `confirmation-lock` with that PASS result, the runtime manifest, and the host, instance, and GPU identities. For S1, run smoke, sequential invariance, concurrent invariance, and qualification in that order; `python3 scripts/validate_gates.py s1` writes the receipt required before production. The receipt binds the exact prepared package and all four gate artifacts.

The GPU host must provide the exact runtime in `configs/experiment_locks.json`. `requirements-inputs.txt` installs local preparation and client libraries; vLLM and CUDA are host-specific and are not installed by the CPU reproduction requirements.

## Results

| Study | Result | Interpretation |
|---|---:|---|
| K4 → K4R | 814/3,840 paired outcomes switched; 408 gains, 406 losses; 99.75% cancellation; +0.052 pp aggregate drift | A rendering-correction transition illustrating turnover and cancellation, not a comparator-only causal intervention or universal treatment effect. |
| R2 Gemma discovery | BASE–MASK interaction +15.33 pp; simultaneous 95% CI [6.70, 23.96] pp | Bounded discovery. DeepSeek is a strict-scoring robust benefit but its machine-parser diagnostic is underdetermined; Llama is underdetermined. |
| Gemma prospective confirmation | BASE +30.33 pp, MASK +10.33 pp, token-matched metadata +22.67 pp; BASE–MASK +20.00 pp [11.82, 28.18] | Magnitude confirmation on 300 untouched tasks. All three gains are positive; this is not a benefit-to-harm reversal. |
| S1 constrained-choice stress | BASE −13.00 pp, MASK −12.00 pp, metadata −12.33 pp; BASE–MASK −1.00 pp [−6.87, +4.87] | Underdetermined. S1 changes both the panel and endpoint, so it does not isolate endpoint choice. |

The frozen `analysis_order` ordinals preserve the canonical 9,999-draw max-t sequence after cluster anonymization. `configs/analysis.json` records expected values extracted from the frozen canonical result artifacts, and `scripts/reproduce.py` checks every primary contrast family against them.

The confirmation panel has no overlap with the frozen admitted experiment registry or qualification panel; that does not establish absence from model pretraining. R2 retains a production-authorization chronology caveat, so its results are bounded discovery. Stage B portability qualification failed (8/27 comparisons exceeded the 7.5 pp tolerance, maximum 25 pp). Extensions E1, E2, E4, and E5 failed qualification and produced no production rows; E3 was not run. S1B Arm B V2 stopped at the sequential invariance strict-parser gate with zero production rows, so there is no crossed estimate. The evidence does not establish cross-domain confirmation or hardware-independent effects.

## License

The included `LICENSE` applies to repository code only. MMLU-Pro and model assets remain under their own terms. The release does not assert permission to redistribute MMLU-Pro or derived source-linked artifacts; source acquisition is local and conditional on the operator's review of the applicable terms.
