#!/usr/bin/env python3
"""Recompute CPU summaries and frozen task-clustered intervals."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFS = ["BASE", "MASK_SENTINEL", "METADATA_TOKEN_MATCHED_TO_MASK"]
TREATMENTS = ["T0_OFF", "T1_SILENT_CHECK"]
CELLS = [(ref, treatment) for ref in REFS for treatment in TREATMENTS]
NAMES = ["tau_BASE", "tau_MASK", "tau_METADATA", "Delta_BASE_MASK", "Delta_BASE_METADATA", "Delta_MASK_METADATA"]
LOADINGS = np.asarray([
    [-1, 1, 0, 0, 0, 0],
    [0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, -1, 1],
    [-1, 1, 1, -1, 0, 0],
    [-1, 1, 0, 0, 1, -1],
    [0, 0, -1, 1, 1, -1],
], dtype=float)
Z95 = 1.959963984540054


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_manifest(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("includes_question_text") is not False or manifest.get("includes_raw_model_responses") is not False:
        raise ValueError("data manifest must exclude prompt text and raw responses")
    if manifest.get("includes_benchmark_task_ids_in_projections") is not False or manifest.get("cluster_map_included") is not False:
        raise ValueError("score projections must not include benchmark task IDs or a cluster crosswalk")
    root = data_dir.resolve()
    for name, entry in manifest["files"].items():
        path = (data_dir / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or unsafe data file: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"data hash mismatch: {name}")
        if name.endswith(".csv") and len(read_csv(path)) != entry["records"]:
            raise ValueError(f"data row count mismatch: {name}")
        if name.endswith("_outcomes.csv"):
            header = next(csv.reader(path.open(newline="", encoding="utf-8")))
            needs_analysis_order = name in {"r2_scored_outcomes.csv", "confirmation_scored_outcomes.csv", "s1_scored_outcomes.csv"}
            if "task_id" in header or "task_order" in header or "cluster_id" not in header or (needs_analysis_order and "analysis_order" not in header):
                raise ValueError(f"score projection is not keyed by opaque clusters: {name}")
    return len(manifest["files"])


def truth(value):
    return str(value).strip().lower() in {"1", "true"}


def ordered_clusters(rows):
    order = {}
    for row in rows:
        cluster = str(row["cluster_id"])
        value = int(row["analysis_order"])
        if value < 0 or (cluster in order and order[cluster] != value):
            raise ValueError("analysis_order must be one consistent nonnegative ordinal per opaque cluster")
        order[cluster] = value
    values = list(order.values())
    if len(set(values)) != len(values) or sorted(values) != list(range(len(values))):
        raise ValueError("analysis_order must be unique and contiguous from zero within each panel")
    return [cluster for cluster, _ in sorted(order.items(), key=lambda item: item[1])]


def max_t(matrix, replicates, seed, alpha=0.05):
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("expected a task-by-contrast matrix with at least two tasks")
    n = matrix.shape[0]
    estimate = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / math.sqrt(n)
    rng = np.random.default_rng(seed)
    maxima = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = matrix[rng.integers(0, n, size=n)]
        boot_mean = sample.mean(axis=0)
        boot_se = sample.std(axis=0, ddof=1) / math.sqrt(n)
        centered = np.abs(boot_mean - estimate)
        studentized = np.divide(centered, boot_se, out=np.where(centered == 0, 0.0, np.inf), where=boot_se > 0)
        maxima[index] = float(np.max(studentized))
    critical = float(np.quantile(maxima, 1 - alpha, method="higher"))
    return estimate, se, critical


def intervals(names, estimate, se, critical, n):
    return [
        {
            "contrast": name,
            "estimate": float(estimate[i]),
            "estimate_pp": float(estimate[i] * 100),
            "se": float(se[i]),
            "simultaneous_ci_low": float(estimate[i] - critical * se[i]),
            "simultaneous_ci_high": float(estimate[i] + critical * se[i]),
            "n_tasks": int(n),
        }
        for i, name in enumerate(names)
    ]


def make_r2_matrix(rows, outcome):
    tasks = ordered_clusters(rows)
    models = sorted({row["model_id"] for row in rows})
    short = {row["model_id"]: row["model_short"] for row in rows}
    cells = defaultdict(list)
    for row in rows:
        key = (row["model_id"], str(row["cluster_id"]), row["reference_id"], row["treatment"])
        cells[key].append(float(row[outcome]))
    if len(cells) != len(models) * len(tasks) * len(REFS) * len(TREATMENTS):
        raise ValueError("R2 input does not contain the complete model/task/reference/treatment grid")
    means = {}
    for key, values in cells.items():
        if len(values) != 3:
            raise ValueError(f"R2 cell does not have three execution-integrity repetitions: {key}")
        if len(set(values)) != 1:
            raise ValueError(f"R2 execution-integrity repetitions disagree: {key}")
        means[key] = float(np.mean(values))
    columns = []
    metadata = []
    for model in models:
        effects = np.asarray([
            [means[(model, task, ref, TREATMENTS[1])] - means[(model, task, ref, TREATMENTS[0])] for ref in REFS]
            for task in tasks
        ], dtype=float)
        for index, ref in enumerate(REFS):
            columns.append(effects[:, index])
            metadata.append((model, "tau", ref))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            columns.append(effects[:, left] - effects[:, right])
            metadata.append((model, "direct_interaction", f"{REFS[left]}_minus_{REFS[right]}"))
    matrix = np.column_stack(columns)
    return tasks, models, short, means, matrix, metadata


def classify_r2(results, models, short, margin):
    decisions = {}
    for model in models:
        tau = [row for row in results if row["model_id"] == model and row["row_type"] == "tau"]
        delta = [row for row in results if row["model_id"] == model and row["row_type"] == "direct_interaction"]
        interaction = any(abs(row["estimate"]) >= margin and (row["simultaneous_ci_low"] > 0 or row["simultaneous_ci_high"] < 0) for row in delta)
        if interaction:
            decision = "REFERENCE_DEPENDENT"
        elif all(row["simultaneous_ci_low"] > 0 for row in tau):
            decision = "ROBUST_BENEFIT"
        elif all(row["simultaneous_ci_high"] < 0 for row in tau):
            decision = "ROBUST_HARM"
        elif all(row["simultaneous_ci_low"] >= -margin and row["simultaneous_ci_high"] <= margin for row in tau):
            decision = "PRACTICALLY_EQUIVALENT"
        else:
            decision = "UNDERDETERMINED"
        decisions[short[model]] = decision
    return decisions


def r2_analysis(rows, config):
    rep = config["bootstrap_replicates"]
    spec = config["r2"]
    strict = make_r2_matrix(rows, "strict_correct")
    semantic = make_r2_matrix(rows, "semantic_correct")
    for left, right in zip(strict[0:3], semantic[0:3]):
        if left != right:
            raise ValueError("strict and semantic R2 panels differ")
    tasks, models, short, means, matrix, metadata = strict
    strict_est, strict_se, strict_q = max_t(matrix, rep, spec["seed"], config["alpha"])
    strict_results = []
    for i, (model, kind, contrast) in enumerate(metadata):
        strict_results.append({
            "model_id": model, "model": short[model], "row_type": kind, "contrast": contrast,
            **intervals([contrast], strict_est[i:i + 1], strict_se[i:i + 1], strict_q, len(tasks))[0],
        })
    decisions = classify_r2(strict_results, models, short, spec["materiality_margin"])

    sem_est, sem_se, sem_q = max_t(semantic[4], rep, spec["seed"], config["alpha"])
    semantic_results = []
    for i, (model, kind, contrast) in enumerate(semantic[5]):
        semantic_results.append({
            "model_id": model, "model": short[model], "row_type": kind, "contrast": contrast,
            **intervals([contrast], sem_est[i:i + 1], sem_se[i:i + 1], sem_q, len(tasks))[0],
        })
    semantic_decisions = classify_r2(semantic_results, models, short, spec["materiality_margin"])

    baseline = {}
    cap_changes = []
    for model in models:
        baseline[short[model]] = {}
        for ref in REFS:
            vals = [means[(model, task, ref, TREATMENTS[0])] for task in tasks]
            baseline[short[model]][ref] = float(np.mean(vals))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            differences = np.asarray([
                means[(model, task, REFS[right], TREATMENTS[0])] - means[(model, task, REFS[left], TREATMENTS[0])]
                for task in tasks
            ])
            se = float(differences.std(ddof=1) / math.sqrt(len(tasks)))
            baseline[short[model]][f"{REFS[right]}_minus_{REFS[left]}"] = {
                "estimate": float(differences.mean()),
                "ci_low": float(differences.mean() - Z95 * se),
                "ci_high": float(differences.mean() + Z95 * se),
                "task_switches": int(np.count_nonzero(differences)),
            }
    cap_cells = defaultdict(list)
    for row in rows:
        cap_cells[(row["model_id"], str(row["cluster_id"]), row["reference_id"], row["treatment"])].append((float(row["strict_correct"]), truth(row["cap_proxy"])))
    for model in models:
        for ref in REFS:
            estimates = []
            uncapped = []
            for treatment in TREATMENTS:
                values = [item for task in tasks for item in cap_cells[(model, task, ref, treatment)]]
                kept = [value for value, capped in values if not capped]
                estimates.append(float(np.mean([value for value, _ in values])))
                uncapped.append(float(np.mean(kept)) if kept else math.nan)
            cap_changes.append(abs((uncapped[1] - uncapped[0]) - (estimates[1] - estimates[0])))
    max_cap_change = max(cap_changes)
    return {
        "tasks_per_model": len(tasks),
        "rows": len(rows),
        "integrity_cells": len(means),
        "max_t_critical": strict_q,
        "strict_results": strict_results,
        "strict_decisions": decisions,
        "semantic_diagnostic": {"max_t_critical": sem_q, "results": semantic_results, "decisions": semantic_decisions},
        "baseline_diagnostic": baseline,
        "cap_proxy_diagnostic": {"maximum_absolute_tau_change": max_cap_change, "headline_plausibly_cap_driven": bool(max_cap_change >= 0.05)},
        "parser_sensitivity": {"verdict": "YES_MAJOR_LIMITATION" if decisions != semantic_decisions else "NO_MATERIAL_CHANGE", "strict": decisions, "semantic": semantic_decisions},
    }


def confirmation_analysis(rows, config):
    spec = config["gemma_confirmation"]
    primary = [row for row in rows if row["execution_class"] == "primary"]
    duplicates = [row for row in rows if row["execution_class"] != "primary"]
    cluster_order = ordered_clusters(rows)
    by_cell = {}
    for row in rows:
        task = str(row["cluster_id"])
        if row["execution_class"] == "primary":
            key = (task, row["reference_id"], row["treatment_id"])
            if key in by_cell:
                raise ValueError(f"duplicate confirmation primary cell: {key}")
            by_cell[key] = row
    if len(primary) != 1800 or len(cluster_order) != 300 or len(duplicates) != 360:
        raise ValueError("confirmation row/task counts do not match the frozen design")
    duplicate_matches = 0
    for row in duplicates:
        key = (str(row["cluster_id"]), row["reference_id"], row["treatment_id"])
        ref = by_cell.get(key)
        if ref is None or truth(ref["strict_correct"]) != truth(row["strict_correct"]):
            raise ValueError(f"confirmation integrity duplicate mismatch: {key}")
        if truth(ref["semantic_correct"]) != truth(row["semantic_correct"]):
            raise ValueError(f"confirmation diagnostic duplicate mismatch: {key}")
        duplicate_matches += 1
    cluster_ids = cluster_order
    matrix = np.asarray([
        [float(truth(by_cell[(task, ref, treatment)]["strict_correct"])) for ref in REFS for treatment in TREATMENTS]
        for task in cluster_ids
    ]) @ LOADINGS.T
    semantic = np.asarray([
        [float(truth(by_cell[(task, ref, treatment)]["semantic_correct"])) for ref in REFS for treatment in TREATMENTS]
        for task in cluster_ids
    ]) @ LOADINGS.T
    names = ["tau_BASE", "tau_MASK_SENTINEL", "tau_METADATA_TOKEN_MATCHED_TO_MASK", "Delta_BASE_MASK", "Delta_BASE_METADATA", "Delta_MASK_METADATA"]
    strict_est, strict_se, strict_q = max_t(matrix, config["bootstrap_replicates"], spec["seed"], config["alpha"])
    sem_est, sem_se, sem_q = max_t(semantic, config["bootstrap_replicates"], spec["seed"], config["alpha"])
    strict_results = intervals(names, strict_est, strict_se, strict_q, len(cluster_ids))
    semantic_results = intervals(names, sem_est, sem_se, sem_q, len(cluster_ids))
    delta = strict_results[3]
    category = "MATERIAL_CONFIRMATION" if delta["simultaneous_ci_low"] > 0.05 else "UNDERDETERMINED"
    return {
        "unique_tasks": len(cluster_ids), "primary_rows": len(primary), "integrity_duplicates": len(duplicates),
        "integrity_duplicate_endpoint_matches": duplicate_matches,
        "strict_max_t_critical": strict_q, "strict_results": strict_results,
        "semantic_diagnostic_max_t_critical": sem_q, "semantic_diagnostic_results": semantic_results,
        "primary_category": category,
        "all_three_treatment_gains_positive": all(row["simultaneous_ci_low"] > 0 for row in strict_results[:3]),
    }


def s1_contrasts(means):
    a = means[1] - means[0]
    b = means[3] - means[2]
    c = means[5] - means[4]
    return np.asarray([a, b, c, a - b, a - c, c - b], dtype=float)


def s1_analysis(rows, config):
    primary = [row for row in rows if row["execution_class"] == "primary"]
    duplicates = [row for row in rows if row["execution_class"] != "primary"]
    by_cell = {}
    for row in primary:
        key = (str(row["cluster_id"]), row["reference"], row["treatment"])
        if key in by_cell:
            raise ValueError(f"duplicate S1 primary cell: {key}")
        by_cell[key] = float(truth(row["correctness"]))
    cluster_ids = ordered_clusters(primary)
    expected_cells = {(ref, treatment) for ref, treatment in CELLS}
    for task in cluster_ids:
        if {(ref, treatment) for (tid, ref, treatment) in by_cell if tid == task} != expected_cells:
            raise ValueError(f"S1 task is missing one or more of the six primary cells: {task}")
    duplicate_matches = 0
    for row in duplicates:
        key = (str(row["cluster_id"]), row["reference"], row["treatment"])
        if key not in by_cell or float(truth(row["correctness"])) != by_cell[key]:
            raise ValueError(f"S1 integrity duplicate mismatch: {key}")
        duplicate_matches += 1
    if len(cluster_ids) != 300 or len(primary) != 1800 or len(duplicates) != 360:
        raise ValueError("S1 row/task counts do not match the frozen design")
    matrix = np.asarray([[by_cell[(task, ref, treatment)] for ref, treatment in CELLS] for task in cluster_ids], dtype=float)
    estimate = s1_contrasts(matrix.mean(axis=0))
    contrast_matrix = np.column_stack([
        matrix[:, 1] - matrix[:, 0], matrix[:, 3] - matrix[:, 2], matrix[:, 5] - matrix[:, 4],
        matrix[:, 1] - matrix[:, 0] - matrix[:, 3] + matrix[:, 2],
        matrix[:, 1] - matrix[:, 0] - matrix[:, 5] + matrix[:, 4],
        matrix[:, 5] - matrix[:, 4] - matrix[:, 3] + matrix[:, 2],
    ])
    se = contrast_matrix.std(axis=0, ddof=1) / math.sqrt(len(cluster_ids))
    rng = np.random.default_rng(config["s1"]["seed"])
    maxima = np.empty(config["bootstrap_replicates"], dtype=float)
    for start in range(0, len(maxima), 256):
        count = min(256, len(maxima) - start)
        indices = rng.integers(0, len(cluster_ids), (count, len(cluster_ids)))
        boot_means = np.empty((count, 6))
        boot_ses = np.empty((count, 6))
        for j in range(count):
            sample = matrix[indices[j]]
            boot_means[j] = s1_contrasts(sample.mean(axis=0))
            boot_ses[j] = np.column_stack([
                sample[:, 1] - sample[:, 0], sample[:, 3] - sample[:, 2], sample[:, 5] - sample[:, 4],
                sample[:, 1] - sample[:, 0] - sample[:, 3] + sample[:, 2],
                sample[:, 1] - sample[:, 0] - sample[:, 5] + sample[:, 4],
                sample[:, 5] - sample[:, 4] - sample[:, 3] + sample[:, 2],
            ]).std(axis=0, ddof=1) / math.sqrt(len(cluster_ids))
        centered = np.abs(boot_means - estimate)
        t = np.divide(centered, boot_ses, out=np.zeros_like(centered), where=boot_ses != 0)
        t[(boot_ses == 0) & (centered > 0)] = np.inf
        maxima[start:start + count] = t.max(axis=1)
    critical = float(np.quantile(maxima, 0.95, method="higher"))
    margin = critical * se
    lower, upper = estimate - margin, estimate + margin
    names = ["tau_BASE", "tau_MASK", "tau_METADATA", "Delta_BASE_MASK", "Delta_BASE_METADATA", "Delta_METADATA_MASK"]
    results = intervals(names, estimate, se, critical, len(cluster_ids))
    for i, row in enumerate(results):
        row["simultaneous_ci_low"] = float(lower[i])
        row["simultaneous_ci_high"] = float(upper[i])
    return {
        "unique_tasks": len(cluster_ids), "primary_rows": len(primary), "integrity_duplicates": len(duplicates),
        "integrity_duplicate_endpoint_matches": duplicate_matches,
        "cell_accuracies": {f"{ref}|{treatment}": float(matrix[:, i].mean()) for i, (ref, treatment) in enumerate(CELLS)},
        "strict_results": results, "max_t_critical": critical,
        "primary_category": "FORMAT_CONTROLLED_REFERENCE_DEPENDENCE_CONFIRMED" if lower[3] > 0.05 else "UNDERDETERMINED",
    }


def k_transition(rows):
    gains = losses = switches = 0
    tasks = defaultdict(bool)
    old = new = 0
    for row in rows:
        before = truth(row["o_correct"])
        after = truth(row["n_correct"])
        old += int(before)
        new += int(after)
        if not before and after:
            gains += 1
        if before and not after:
            losses += 1
        if before != after:
            switches += 1
            tasks[str(row["cluster_id"])] = True
        else:
            tasks.setdefault(str(row["cluster_id"]), False)
    return {
        "pairs": len(rows), "gains": gains, "losses": losses, "switches": switches,
        "turnover": switches / len(rows), "cancellation": 1 - abs(gains - losses) / switches,
        "old_accuracy": old / len(rows), "new_accuracy": new / len(rows),
        "aggregate_shift": (new - old) / len(rows), "tasks": len(tasks),
        "tasks_switching": sum(tasks.values()), "task_any_switch_rate": sum(tasks.values()) / len(tasks),
    }


def check(expected, actual, path="expected"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            check(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(expected), float(actual), rel_tol=0, abs_tol=2e-12):
            raise ValueError(f"canonical mismatch at {path}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"canonical mismatch at {path}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/reproduced_summary.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/analysis.json")
    parser.add_argument("--r2-outcomes", type=Path, help="optional fresh score projection from scripts/score_outputs.py")
    parser.add_argument("--confirmation-outcomes", type=Path, help="optional fresh score projection from scripts/score_outputs.py")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_files = verify_manifest(args.data_dir)
    k = k_transition(read_csv(args.data_dir / "k_transition_outcomes.csv"))
    r2_path = args.r2_outcomes or args.data_dir / "r2_scored_outcomes.csv"
    confirmation_path = args.confirmation_outcomes or args.data_dir / "confirmation_scored_outcomes.csv"
    r2 = r2_analysis(read_csv(r2_path), config)
    confirmation = confirmation_analysis(read_csv(confirmation_path), config)
    s1 = s1_analysis(read_csv(args.data_dir / "s1_scored_outcomes.csv"), config)
    study_status = json.loads((args.data_dir / "study_status.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS",
        "manifest_files_verified": manifest_files,
        "inference": "Task-clustered studentized max-t; frozen seeds and canonical anonymous analysis_order; integrity repetitions averaged or excluded as specified.",
        "new_model_generations": 0,
        "k_transition": k,
        "r2": r2,
        "gemma_confirmation": confirmation,
        "s1": s1,
        "study_status": study_status,
        "s1b": study_status["s1b_arm_b_v2"],
    }
    expected = config["expected"]
    check(expected["k_transition"], {key: k[key] for key in expected["k_transition"]}, "k_transition")

    def contrast_map(rows):
        return {
            row["contrast"]: {
                "estimate": row["estimate"],
                "ci_low": row["simultaneous_ci_low"],
                "ci_high": row["simultaneous_ci_high"],
            }
            for row in rows
        }

    r2_canonical = {}
    for row in r2["strict_results"]:
        r2_canonical.setdefault(row["model"], {})[row["contrast"]] = {
            "estimate": row["estimate"],
            "ci_low": row["simultaneous_ci_low"],
            "ci_high": row["simultaneous_ci_high"],
        }
    check(expected["r2_strict_contrasts"], r2_canonical, "r2_strict_contrasts")
    check(expected["confirmation_strict_contrasts"], contrast_map(confirmation["strict_results"]), "confirmation_strict_contrasts")
    check(expected["confirmation_semantic_diagnostic"], contrast_map(confirmation["semantic_diagnostic_results"]), "confirmation_semantic_diagnostic")
    check(expected["s1_contrasts"], contrast_map(s1["strict_results"]), "s1_contrasts")
    check(expected["max_t_critical"], {
        "r2_strict": r2["max_t_critical"],
        "confirmation_strict": confirmation["strict_max_t_critical"],
        "confirmation_semantic_diagnostic": confirmation["semantic_diagnostic_max_t_critical"],
        "s1": s1["max_t_critical"],
    }, "max_t_critical")
    for name, section in (("r2_gemma_base_minus_mask", [r for r in r2["strict_results"] if r["model"] == "gemma"]), ("confirmation_base_minus_mask", confirmation["strict_results"]), ("s1_base_minus_mask", s1["strict_results"])):
        key = "BASE_minus_MASK_SENTINEL" if name.startswith("r2_") else "Delta_BASE_MASK"
        row = next(item for item in section if item["contrast"] == key)
        check(expected[name], {"estimate": row["estimate"], "ci_low": row["simultaneous_ci_low"], "ci_high": row["simultaneous_ci_high"]}, name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out), "k_switches": k["switches"], "r2_gemma_delta_pp": 100 * next(r["estimate"] for r in r2["strict_results"] if r["model"] == "gemma" and r["contrast"] == "BASE_minus_MASK_SENTINEL"), "confirmation_delta_pp": 100 * confirmation["strict_results"][3]["estimate"], "s1_delta_pp": 100 * s1["strict_results"][3]["estimate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
