#!/usr/bin/env python3
"""Project canonical R2 and confirmation outputs into the anonymous CPU schemas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINAL = re.compile(r"FINAL\s*:\s*([A-J])\b", re.I)
SEMANTIC_PATTERNS = [
    ("final_letter_strict", re.compile(r"FINAL\s*:\s*([A-J])\b", re.I)),
    ("boxed", re.compile(r"\\boxed\{\s*\(?([A-J])\)?\s*[.)]?\s*\}", re.I)),
    ("letter_tag", re.compile(r"<LETTER>\s*\(?([A-J])\)?\s*</LETTER>", re.I)),
    ("final_answer_is", re.compile(r"final\s+answer\s*(?:is)?\s*[:\-]?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])", re.I)),
    ("the_answer_is", re.compile(r"\banswer\s+is\s*[:\-]?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])", re.I)),
    ("answer_colon", re.compile(r"\banswer\s*:\s*\**\s*\(?([A-J])\)?(?![A-Za-z])", re.I)),
    ("option_letter", re.compile(r"\boption\s*\(?([A-J])\)?(?![A-Za-z])", re.I)),
    ("bold_letter_terminal", re.compile(r"\*\*\s*\(?([A-J])\)?\s*[.)]?\s*\*\*\s*\.?\s*$", re.I | re.M)),
    ("bare_letter_terminal", re.compile(r"^\s*\(?([A-J])\)?\s*[.)]?\s*$", re.I | re.M)),
]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def verified_package(root: Path) -> dict[str, Any]:
    path = root / "PREPARED_INPUTS_MANIFEST.json"
    if not path.is_file():
        raise ValueError("prepared input manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("local_only") is not True:
        raise ValueError("prepared package is not a verified local-only package")
    for rel, entry in manifest.get("files", {}).items():
        candidate = root / rel
        if not candidate.is_file() or sha_file(candidate) != entry.get("sha256"):
            raise ValueError("prepared package hash mismatch")
    return manifest


def strict_score(text: str, gold: str) -> dict[str, Any]:
    # Frozen R2 and confirmation scorers: last FINAL:<A-J> occurrence only.
    matches = list(FINAL.finditer(text or ""))
    if not matches:
        return {"parsed": False, "pred": None, "gold": gold, "correct": False, "method": "UNPARSED"}
    pred = matches[-1].group(1).upper()
    return {"parsed": True, "pred": pred, "gold": gold, "correct": pred == gold, "method": "FINAL"}


def semantic_score(text: str, gold: str, strict: dict[str, Any], *, cap: bool) -> dict[str, Any]:
    # Frozen R2 semantic diagnostic, also used diagnostically for confirmation.
    candidates = []
    if not text:
        semantic = {"letter": None, "determinate": False, "reason": "EMPTY"}
    else:
        for name, pattern in SEMANTIC_PATTERNS:
            for match in pattern.finditer(text):
                candidates.append((match.end(), match.group(1).upper(), name))
        if not candidates:
            semantic = {"letter": None, "determinate": False, "reason": "NO_MATCH"}
        else:
            best = max(candidate[0] for candidate in candidates)
            letters = {candidate[1] for candidate in candidates if candidate[0] == best}
            if len(letters) > 1:
                semantic = {"letter": None, "determinate": False, "reason": "CONFLICT_AT_SAME_POSITION"}
            else:
                letter = letters.pop()
                semantic = {"letter": letter, "determinate": True, "reason": "OK"}
    determinate = bool(semantic["determinate"])
    sem_correct = bool(determinate and semantic["letter"] == gold)
    if bool(strict["correct"]):
        category = "STRICT_CORRECT"
    elif bool(strict["parsed"]):
        category = "SEMANTIC_WRONG"
    elif cap:
        category = "TRUNCATED"
    elif not determinate:
        category = "SEMANTIC_UNDETERMINED"
    elif sem_correct:
        category = "SEMANTIC_CORRECT_FORMAT_FAIL"
    else:
        category = "SEMANTIC_WRONG"
    return {"determinate": determinate, "correct": sem_correct, "pred": semantic["letter"], "category": category}


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def projection_clusters(data_dir: Path, filename: str) -> dict[int, str]:
    rows = csv.DictReader((data_dir / filename).open(encoding="utf-8", newline=""))
    result: dict[int, str] = {}
    for row in rows:
        order = int(row["analysis_order"])
        prior = result.setdefault(order, row["cluster_id"])
        if prior != row["cluster_id"]:
            raise ValueError("anonymous cluster order is inconsistent")
    if sorted(result) != list(range(len(result))):
        raise ValueError("anonymous analysis order is not contiguous")
    return result


def score_r2(prepared: Path, outputs: list[Path], data_dir: Path, out: Path) -> int:
    verified_package(prepared)
    expected = list(csv.DictReader((prepared / "07_EXPERIMENTS/R2/EXPECTED_ROWS.csv").open(encoding="utf-8", newline="")))
    tasks = {str(row["task_id"]): row for row in jsonl(prepared / "R2_TASK_GOLD.jsonl")}
    if len(tasks) != 300 or len(expected) != 16200:
        raise ValueError("R2 prepared keyspace is incomplete")
    panels = json.loads((data_dir / "input_panel_manifest.json").read_text(encoding="utf-8"))["panels"]
    panel = panels["r2_primary"]
    analysis_hashes = panel.get("analysis_order_hashes", panel["ordered_hashes"])
    order_by_hash = {value: index for index, value in enumerate(analysis_hashes)}
    if len(order_by_hash) != 300:
        raise ValueError("R2 anonymous analysis-order manifest is malformed")
    clusters = projection_clusters(data_dir, "r2_scored_outcomes.csv")
    if len(clusters) != 300:
        raise ValueError("R2 anonymous cluster projection is incomplete")
    gold_by_key = {}
    order_by_task = {}
    for task_id, task in tasks.items():
        digest = str(task["task_hash"])
        order = order_by_hash.get(digest)
        if order is None or task.get("analysis_order") != order:
            raise ValueError("prepared R2 task does not map to the frozen anonymous order")
        order_by_task[task_id] = order
    for row in expected:
        task_id = str(row["task_id"])
        task = tasks.get(task_id)
        if task is None:
            raise ValueError("R2 expected rows include an unprepared task")
        key = (row["model_id"], task_id, row["reference_id"], row["treatment"], str(row["seed"]))
        gold_by_key[key] = str(task["gold"])
    if len(gold_by_key) != len(expected):
        raise ValueError("R2 gold projection is incomplete")
    locks = json.loads((ROOT / "configs/experiment_locks.json").read_text(encoding="utf-8"))["r2"]
    models = {row["id"]: row["revision"] for row in locks["models"]}
    short = {"google/gemma-4-12B-it": "gemma", "meta-llama/Llama-3.1-8B-Instruct": "llama", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek"}
    contract = {}
    for row in jsonl(prepared / "02_R2/R2_EXACT_RENDER_LEDGER.jsonl"):
        contract[(row["model_id"], str(row["task_id"]), row["reference_id"], row["treatment"])] = row
    raw: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for source in outputs:
        for row in jsonl(source):
            key = (str(row.get("model_id", "")), str(row.get("task_id", "")), str(row.get("reference_id", "")), str(row.get("treatment", "")), str(row.get("seed", "")))
            if key in raw or key not in gold_by_key:
                raise ValueError("R2 raw output has a duplicate or unexpected scientific key")
            raw[key] = row
    if set(raw) != set(gold_by_key):
        raise ValueError("R2 raw output is missing frozen rows")
    scored = []
    for expected_row in expected:
        key = (expected_row["model_id"], str(expected_row["task_id"]), expected_row["reference_id"], expected_row["treatment"], str(expected_row["seed"]))
        row = raw[key]
        expected_raw_key = "|".join(key)
        if row.get("key") not in (None, expected_raw_key):
            raise ValueError("R2 raw output key differs from the frozen row identity")
        if row.get("model_revision") != models.get(key[0]):
            raise ValueError("R2 model revision differs from frozen lock")
        if int(row.get("max_output_tokens", -1)) != 8192 or row.get("raw_response") is None:
            raise ValueError("R2 raw output is missing the frozen token limit or response")
        response = str(row["raw_response"])
        if row.get("response_hash") not in (None, hashlib.sha256(response.encode()).hexdigest()):
            raise ValueError("R2 raw response hash mismatch")
        if row.get("request_config") and "seed" in row["request_config"]:
            raise ValueError("R2 request config claims a seed parameter absent from the canonical transport")
        if row.get("seed_sent_to_backend") not in (None, False, 0, "false", "False"):
            raise ValueError("R2 output metadata contradicts the canonical seed transport")
        render = contract.get((key[0], key[1], key[2], key[3]))
        if render is None or row.get("prompt_hash") != render.get("prompt_hash"):
            raise ValueError("R2 prompt does not match the frozen render ledger")
        if row.get("input_token_ids_sha256") not in (None, render.get("token_ids_sha256")):
            raise ValueError("R2 input token hash differs from the frozen render ledger")
        output_tokens = row.get("output_tokens")
        if output_tokens is None or not str(output_tokens).isdigit():
            raise ValueError("R2 output token count is missing or invalid")
        gold = gold_by_key[key]
        strict = strict_score(response, gold)
        runtime_error = truth(row.get("runtime_error"))
        cap_proxy = int(output_tokens) == 8192 and int(row["max_output_tokens"]) == 8192
        semantic = semantic_score(response, gold, strict, cap=cap_proxy)
        order = order_by_task[key[1]]
        scored.append({
            "cluster_id": clusters[order], "analysis_order": order, "model_short": short[key[0]], "model_id": key[0],
            "reference_id": key[2], "treatment": key[3], "seed": int(key[4]),
            "strict_parseable": int(bool(strict["parsed"])), "strict_correct": int(bool(strict["correct"]) and not runtime_error),
            "semantic_determinate": int(semantic["determinate"]), "semantic_correct": int(semantic["correct"]),
            "cap_proxy": int(cap_proxy), "output_tokens": int(output_tokens),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cluster_id", "analysis_order", "model_short", "model_id", "reference_id", "treatment", "seed", "strict_parseable", "strict_correct", "semantic_determinate", "semantic_correct", "cap_proxy", "output_tokens"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(scored)
    print(json.dumps({"status": "PASS", "study": "R2", "rows": len(scored), "models": len(models), "task_ids_in_projection": False, "out": str(out)}, sort_keys=True))
    return 0


def score_confirmation(prepared: Path, outputs: list[Path], data_dir: Path, out: Path) -> int:
    verified_package(prepared)
    requests = jsonl(prepared / "CONFIRMATION_REQUESTS.jsonl")
    if len(requests) != 2160:
        raise ValueError("confirmation prepared keyspace is incomplete")
    req_by_key = {str(row["request_key"]): row for row in requests}
    identity = lambda row: (str(row.get("phase", "")), str(row.get("task_id", "")), str(row.get("reference_id", "")), str(row.get("treatment_id", row.get("treatment", ""))), str(row.get("execution_class", "")))
    req_by_identity = {identity(row): row for row in requests}
    if len(req_by_key) != len(requests) or len(req_by_identity) != len(requests):
        raise ValueError("confirmation request ledger has duplicate identities")
    panels = json.loads((data_dir / "input_panel_manifest.json").read_text(encoding="utf-8"))["panels"]
    hashes = panels["confirmation_primary"].get("analysis_order_hashes", panels["confirmation_primary"]["ordered_hashes"])
    order_by_hash = {value: index for index, value in enumerate(hashes)}
    clusters = projection_clusters(data_dir, "confirmation_scored_outcomes.csv")
    if len(order_by_hash) != 300 or len(clusters) != 300:
        raise ValueError("confirmation anonymous analysis order is incomplete")
    raw_rows = []
    seen_requests = set()
    for source in outputs:
        for row in jsonl(source):
            request = req_by_key.get(str(row.get("request_key", "")))
            exact_key = request is not None
            if request is None:
                request = req_by_identity.get(identity(row))
            if request is None:
                raise ValueError("confirmation raw output has an unexpected identity")
            key = str(request["request_key"])
            if key in seen_requests:
                raise ValueError("confirmation raw output contains duplicate request rows")
            seen_requests.add(key)
            for field in ("phase", "execution_class", "task_id", "reference_id", "model_id", "model_revision", "tokenizer_revision", "prompt_hash"):
                raw_field = "treatment_id" if field == "treatment_id" else field
                if field == "treatment_id":
                    continue
                if row.get(raw_field) != request.get(field):
                    raise ValueError("confirmation output metadata differs from the frozen request")
            if row.get("treatment_id", row.get("treatment")) != request.get("treatment_id"):
                raise ValueError("confirmation treatment differs from the frozen request")
            if exact_key and row.get("request_sha256") != request.get("request_sha256"):
                raise ValueError("confirmation request hash differs from the frozen request")
            response = row.get("raw_response")
            if not isinstance(response, str):
                raise ValueError("confirmation raw response is missing")
            if row.get("response_hash") not in (None, hashlib.sha256(response.encode()).hexdigest()):
                raise ValueError("confirmation raw response hash mismatch")
            try:
                output_tokens = int(row["output_tokens"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("confirmation output token count is missing or invalid") from None
            finish = row.get("finish_reason")
            if not isinstance(finish, str) or not finish:
                raise ValueError("confirmation finish reason is missing")
            strict = strict_score(response, str(request["gold"]))
            cap = output_tokens == int(request["expected_generation_parameters"]["max_tokens"])
            semantic = semantic_score(response, str(request["gold"]), strict, cap=cap)
            order = order_by_hash.get(str(request.get("task_hash", "")))
            if order is None:
                raise ValueError("confirmation task does not map to the frozen anonymous order")
            raw_rows.append({
                "cluster_id": clusters[order], "analysis_order": order,
                "reference_id": request["reference_id"], "treatment_id": request["treatment_id"],
                "execution_class": request["execution_class"], "strict_correct": bool(strict["correct"]),
                "semantic_correct": bool(semantic["correct"]), "semantic_determinate": bool(semantic["determinate"]),
                "output_tokens": output_tokens, "finish_reason": finish,
            })
    if len(seen_requests) != len(requests):
        raise ValueError("confirmation raw output is missing frozen requests")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cluster_id", "analysis_order", "reference_id", "treatment_id", "execution_class", "strict_correct", "semantic_correct", "semantic_determinate", "output_tokens", "finish_reason"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(raw_rows)
    print(json.dumps({"status": "PASS", "study": "confirmation", "rows": len(raw_rows), "task_ids_in_projection": False, "out": str(out)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="study", required=True)
    for name, default_package in (("r2", "data/local/prepared/r2"), ("confirmation", "data/local/prepared/confirmation")):
        command = sub.add_parser(name)
        command.add_argument("--prepared", type=Path, default=ROOT / default_package)
        command.add_argument("--outputs", type=Path, nargs="+", required=True)
        command.add_argument("--data-dir", type=Path, default=ROOT / "data")
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.study == "r2":
        return score_r2(args.prepared, args.outputs, args.data_dir, args.out)
    return score_confirmation(args.prepared, args.outputs, args.data_dir, args.out)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"scoring failed closed: {exc}") from None
