#!/usr/bin/env python3
"""Acquire and locally prepare frozen BlackwellBench MMLU-Pro inputs.

Benchmark text and task identifiers are materialized only under data/local/.
The repository stores a pinned official source location, a source hash, and
opaque content commitments; it does not redistribute the source dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "input_panel_manifest.json"
LOCAL = ROOT / "data" / "local"
SOURCE_DEFAULT = LOCAL / "source" / "MMLU-Pro_test_b189ec765aa7ed75c8acfea42df31fdae71f97be.parquet"
SOURCE_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet?download=true"
SOURCE_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
SOURCE_ROWS = 12032
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
GEMMA = "google/gemma-4-12B-it"
GEMMA_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "81cee02dd020268dced5fa1327e8555acce9c63c",
    GEMMA: GEMMA_REVISION,
}
TOKENIZER_DIRS = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b_it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek_r1_distill_llama8b",
    GEMMA: "gemma4_12b_it",
}
CHAT_TEMPLATE_SHA256 = {
    "meta-llama/Llama-3.1-8B-Instruct": "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "b6835114b7303ddd78919a82e4d9f7d8c26ed0d7dfc36beeb12d524f6144eab1",
    GEMMA: "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
}
REFERENCES = ("BASE", "MASK_SENTINEL", "METADATA_TOKEN_MATCHED_TO_MASK")
TREATMENTS = ("T0_OFF", "T1_SILENT_CHECK")
LETTERS = "ABCDEFGHIJ"
MASK = "[MASK]"
META = "[META]"
TREATMENT_TEXT = "Silently verify the selected option before giving the required final line."
R2_TREATMENT_TEXT = "Silently verify the selected option before giving the required final line."
S1_TREATMENT_TEXT = "Silently verify the selected option before giving the required final line."
S1_UNIFORM = "Return exactly one allowed answer label."
CONFIRMATION_SUFFIX = "End with FINAL: <LETTER>."


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(stable_json(row) + "\n" for row in rows), encoding="utf-8")


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "question": normalize(task["question"]),
        "options": [normalize(value) for value in task["options"]],
        "answer_index": int(task["answer_index"]),
        "category": normalize(task["category"]),
    }
    return sha_bytes(stable_json(payload).encode("utf-8"))


def load_source(path: Path, *, acknowledge_terms: bool) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not acknowledge_terms:
        raise SystemExit(
            "refusing to read MMLU-Pro: review the official dataset terms and pass --accept-source-terms. "
            "This acknowledgement is not a permission grant."
        )
    if not path.is_file() or sha_file(path) != SOURCE_SHA256:
        raise SystemExit(f"source is absent or its SHA-256 differs from the frozen official revision: {path}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("install anonymous_repo/requirements-inputs.txt to read the pinned Parquet source") from exc
    frame = pd.read_parquet(path)
    if len(frame) != SOURCE_ROWS:
        raise SystemExit(f"source record count drift: expected {SOURCE_ROWS}, got {len(frame)}")
    needed = {"question_id", "question", "options", "answer_index", "category", "src"}
    if not needed.issubset(frame.columns):
        raise SystemExit("pinned source schema drift: required MMLU-Pro columns are absent")
    tasks: list[dict[str, Any]] = []
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for source_index, (_, series) in enumerate(frame.iterrows()):
        row = series.to_dict()
        task = {
            "task_id": str(int(row["question_id"])),
            "source_question_id": str(int(row["question_id"])),
            "source_index": source_index,
            "source": str(row["src"]),
            "category": str(row["category"]),
            "question": str(row["question"]),
            "options": [str(value) for value in row["options"]],
            "answer_index": int(row["answer_index"]),
        }
        if not 0 <= task["answer_index"] < len(task["options"]) or len(task["options"]) > len(LETTERS):
            raise SystemExit("pinned source contains an invalid answer index or option count")
        task["gold"] = LETTERS[task["answer_index"]]
        fingerprint = task_fingerprint(task)
        if fingerprint in by_fingerprint:
            raise SystemExit("pinned source has a duplicate frozen task fingerprint")
        tasks.append(task)
        by_fingerprint[fingerprint] = task
    return tasks, by_fingerprint


def select_panels(by_fingerprint: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("sha256") != SOURCE_SHA256:
        raise SystemExit("input-panel manifest and source dataset locks disagree")
    selected: dict[str, list[dict[str, Any]]] = {}
    for name, panel in manifest["panels"].items():
        hashes = panel["ordered_hashes"]
        if len(hashes) != int(panel["count"]) or len(set(hashes)) != len(hashes):
            raise SystemExit(f"frozen panel manifest is malformed: {name}")
        analysis_hashes = panel.get("analysis_order_hashes", hashes)
        if len(analysis_hashes) != len(hashes) or set(analysis_hashes) != set(hashes):
            raise SystemExit(f"frozen anonymous analysis order is malformed: {name}")
        missing = [digest for digest in hashes if digest not in by_fingerprint]
        if missing:
            raise SystemExit(f"official pinned source does not reproduce panel {name}: {len(missing)} task commitments missing")
        selected[name] = [by_fingerprint[digest] for digest in hashes]
    return selected


def make_tokenizer(model: str, revision: str, *, local_only: bool, tokenizer_cache_dir: Path | None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install anonymous_repo/requirements-inputs.txt to build tokenizer-bound request locks") from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        if tokenizer_cache_dir is not None:
            tokenizer_path = tokenizer_cache_dir / TOKENIZER_DIRS[model]
            if not tokenizer_path.is_dir():
                raise FileNotFoundError(f"tokenizer directory absent: {tokenizer_path.name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=True,
                token=token,
                local_files_only=local_only,
            )
    except Exception as exc:
        mode = "provided tokenizer cache" if tokenizer_cache_dir is not None else ("local tokenizer cache" if local_only else "pinned model repository")
        raise SystemExit(f"cannot load {model}@{revision} from {mode}: {type(exc).__name__}") from None
    observed_template_sha = sha_bytes(str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8"))
    if observed_template_sha != CHAT_TEMPLATE_SHA256[model]:
        raise SystemExit(f"chat template hash mismatch for {model}@{revision}")
    return tokenizer


def chat_ids(tokenizer, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return str(rendered), [int(value) for value in ids]


def reference_prefix(reference: str) -> str:
    return {"BASE": "", "MASK_SENTINEL": MASK, "METADATA_TOKEN_MATCHED_TO_MASK": META}[reference]


def user_text(task: dict[str, Any], reference: str, suffix: str, *, strip_source_fields: bool = False) -> str:
    prefix = reference_prefix(reference)
    question = str(task["question"]).strip() if strip_source_fields else str(task["question"])
    options = "\n".join(
        f"{LETTERS[index]}. {str(value).strip() if strip_source_fields else str(value)}"
        for index, value in enumerate(task["options"])
    )
    head = f"{prefix}\n\n" if prefix else ""
    return f"{head}{question}\n\n{options}\n\n{suffix}"


def r2_messages(task: dict[str, Any], reference: str, treatment: str) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": user_text(task, reference, CONFIRMATION_SUFFIX, strip_source_fields=True)}]
    if treatment == "T1_SILENT_CHECK":
        messages.insert(0, {"role": "system", "content": R2_TREATMENT_TEXT})
    return messages


def build_r2(tasks: list[dict[str, Any]], output: Path, *, local_only: bool, tokenizer_cache_dir: Path | None) -> dict[str, Any]:
    r2_root = output / "r2"
    expected_dir = r2_root / "07_EXPERIMENTS" / "R2"
    ledger_path = r2_root / "02_R2" / "R2_EXACT_RENDER_LEDGER.jsonl"
    contracts: list[dict[str, Any]] = []
    template_hashes: dict[str, str] = {}
    panel_commitment_sha = sha_bytes("\n".join(task_fingerprint(task) for task in tasks).encode("utf-8"))
    token_match_count = 0
    for model, revision in MODELS.items():
        tokenizer = make_tokenizer(model, revision, local_only=local_only, tokenizer_cache_dir=tokenizer_cache_dir)
        template = str(getattr(tokenizer, "chat_template", "") or "")
        template_hashes[model] = sha_bytes(template.encode("utf-8"))
        local_contracts: dict[tuple[str, str], tuple[int, int]] = {}
        for task in tasks:
            for treatment in TREATMENTS:
                mask_render, mask_ids = chat_ids(tokenizer, r2_messages(task, "MASK_SENTINEL", treatment))
                meta_render, meta_ids = chat_ids(tokenizer, r2_messages(task, "METADATA_TOKEN_MATCHED_TO_MASK", treatment))
                if len(mask_ids) != len(meta_ids):
                    raise SystemExit(f"R2 [MASK]/[META] full-chat token-count lock failed for {model}")
                token_match_count += 1
                local_contracts[(task["task_id"], treatment)] = (len(mask_ids), len(meta_ids))
            for reference in REFERENCES:
                for treatment in TREATMENTS:
                    messages = r2_messages(task, reference, treatment)
                    rendered, ids = chat_ids(tokenizer, messages)
                    contracts.append({
                        "study": "R2_MMLUPRO300",
                        "task_id": task["task_id"],
                        "model_id": model,
                        "model_revision": revision,
                        "tokenizer_revision": revision,
                        "reference_id": reference,
                        "treatment": treatment,
                        "messages": messages,
                        "rendered_chat_text": rendered,
                        "rendered_chat_sha256": sha_bytes(rendered.encode("utf-8")),
                        "prompt_hash": sha_bytes(stable_json(messages).encode("utf-8")),
                        "token_ids_sha256": sha_bytes(json.dumps(ids, separators=(",", ":")).encode("utf-8")),
                        "full_chat_token_count": len(ids),
                        "panel_commitment_sha256": panel_commitment_sha,
                    })
        if len(local_contracts) != 600:
            raise SystemExit(f"R2 tokenizer contract count mismatch for {model}")
    if len(contracts) != 5400:
        raise SystemExit("R2 rendered ledger row count mismatch")
    write_jsonl(ledger_path, contracts)
    expected_dir.mkdir(parents=True, exist_ok=True)
    expected_path = expected_dir / "EXPECTED_ROWS.csv"
    with expected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["study_id", "benchmark", "model_id", "task_id", "reference_id", "treatment", "seed", "max_output_tokens", "shard_id"])
        writer.writeheader()
        for model in MODELS:
            for task in tasks:
                for reference in REFERENCES:
                    for treatment in TREATMENTS:
                        for seed in (2701, 2702, 2703):
                            writer.writerow({"study_id": "R2_MMLUPRO300", "benchmark": "MMLU-Pro_common300", "model_id": model, "task_id": task["task_id"], "reference_id": reference, "treatment": treatment, "seed": seed, "max_output_tokens": 8192, "shard_id": "local"})
    meta = {"status": "PASS", "models": len(MODELS), "panel_tasks": len(tasks), "rendered_contract_rows": len(contracts), "token_match_pairs": token_match_count, "tokenizer_chat_template_sha256": template_hashes, "ledger_sha256": sha_file(ledger_path), "expected_rows_sha256": sha_file(expected_path)}
    write_json(r2_root / "R2_PREPARED_MANIFEST.json", meta)
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))["panels"]["r2_primary"]
    analysis_hashes = panel.get("analysis_order_hashes", panel["ordered_hashes"])
    analysis_order = {digest: index for index, digest in enumerate(analysis_hashes)}
    write_jsonl(
        r2_root / "R2_TASK_GOLD.jsonl",
        [
            {
                "task_id": task["task_id"],
                "task_hash": task_fingerprint(task),
                "analysis_order": analysis_order[task_fingerprint(task)],
                "gold": task["gold"],
            }
            for task in tasks
        ],
    )
    prepared = write_prepared_manifest(
        r2_root,
        ["07_EXPERIMENTS/R2/EXPECTED_ROWS.csv", "02_R2/R2_EXACT_RENDER_LEDGER.jsonl", "R2_TASK_GOLD.jsonl"],
        {"panel_tasks": len(tasks), "rendered_contract_rows": len(contracts), "models": len(MODELS)},
    )
    return {**meta, "prepared_manifest_sha256": sha_file(r2_root / "PREPARED_INPUTS_MANIFEST.json"), "status": prepared["status"]}


def confirmation_messages(task: dict[str, Any], reference: str, treatment: str) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": user_text(task, reference, CONFIRMATION_SUFFIX)}]
    if treatment == "T1_SILENT_CHECK":
        messages.insert(0, {"role": "system", "content": TREATMENT_TEXT})
    return messages


def make_confirmation_request(task: dict[str, Any], tokenizer, phase: str, execution_class: str, reference: str, treatment: str, duplicate_of: str | None = None) -> dict[str, Any]:
    messages = confirmation_messages(task, reference, treatment)
    rendered, ids = chat_ids(tokenizer, messages)
    params = {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 8192, "max_model_len": 10496, "seed": 2701}
    key = f"{phase.upper()}::{task['task_id']}::{reference}::{treatment}::{execution_class.upper()}"
    canonical = {"model": GEMMA, "messages": messages, "generation_parameters": params, "rendered_chat_sha256": sha_bytes(rendered.encode("utf-8"))}
    return {
        "schema_version": "GEMMA_RTX5090_REQUEST_V1",
        "request_key": key,
        "request_id": key,
        "phase": phase,
        "execution_class": execution_class,
        "task_id": task["task_id"],
        "source_question_id": task["source_question_id"],
        "source_index": task["source_index"],
        "category": task["category"],
        "task_hash": task_fingerprint(task),
        "task_prompt_hash": sha_bytes(normalize(task["question"]).encode("utf-8")),
        "model_id": GEMMA,
        "model_revision": GEMMA_REVISION,
        "tokenizer_revision": GEMMA_REVISION,
        "reference_id": reference,
        "treatment_id": treatment,
        "reference_literal": reference_prefix(reference),
        "messages": messages,
        "exact_system_text": next((message["content"] for message in messages if message["role"] == "system"), None),
        "exact_user_text": messages[-1]["content"],
        "rendered_chat_text": rendered,
        "rendered_chat_sha256": sha_bytes(rendered.encode("utf-8")),
        "prompt_hash": sha_bytes(stable_json(messages).encode("utf-8")),
        "expected_generation_parameters": params,
        "canonical_serialized_request": canonical,
        "request_sha256": sha_bytes(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
        "input_token_count": len(ids),
        "full_chat_token_count": len(ids),
        "token_ids_sha256": sha_bytes(json.dumps(ids, separators=(",", ":")).encode("utf-8")),
        "gold": task["gold"],
        "is_inferential_primary": execution_class == "primary",
        "integrity_duplicate_of": duplicate_of,
        "task_snapshot": {"task_id": task["task_id"], "category": task["category"], "question": task["question"], "options": task["options"], "gold": task["gold"]},
    }


def build_confirmation(panels: dict[str, list[dict[str, Any]]], output: Path, *, local_only: bool, tokenizer_cache_dir: Path | None) -> dict[str, Any]:
    root = output / "confirmation"
    tokenizer = make_tokenizer(GEMMA, GEMMA_REVISION, local_only=local_only, tokenizer_cache_dir=tokenizer_cache_dir)
    requests: dict[str, list[dict[str, Any]]] = {"qualification": [], "confirmation": []}
    specs = (
        ("qualification", "confirmation_qualification_primary", "primary"),
        ("confirmation", "confirmation_primary", "primary"),
    )
    for phase, panel_name, execution_class in specs:
        for task in panels[panel_name]:
            for reference in REFERENCES:
                for treatment in TREATMENTS:
                    requests[phase].append(make_confirmation_request(task, tokenizer, phase, execution_class, reference, treatment))
    duplicate_specs = (
        ("qualification", "confirmation_qualification_integrity_duplicates"),
        ("confirmation", "confirmation_integrity_duplicates"),
    )
    for phase, panel_name in duplicate_specs:
        primary_by_fingerprint = {task_fingerprint(task): task for task in panels["confirmation_qualification_primary" if phase == "qualification" else "confirmation_primary"]}
        for task in panels[panel_name]:
            primary = primary_by_fingerprint.get(task_fingerprint(task))
            if primary is None:
                raise SystemExit(f"{phase} integrity duplicate is not in its primary task panel")
            for reference in REFERENCES:
                for treatment in TREATMENTS:
                    parent = f"{phase.upper()}::{primary['task_id']}::{reference}::{treatment}::PRIMARY"
                    requests[phase].append(make_confirmation_request(task, tokenizer, phase, "integrity_duplicate", reference, treatment, parent))
    if (len(requests["qualification"]), len(requests["confirmation"])) != (180, 2160):
        raise SystemExit("confirmation request ledger counts do not match frozen qualification/confirmation design")
    write_jsonl(root / "QUALIFICATION_REQUESTS.jsonl", requests["qualification"])
    write_jsonl(root / "CONFIRMATION_REQUESTS.jsonl", requests["confirmation"])
    return write_prepared_manifest(root, ["QUALIFICATION_REQUESTS.jsonl", "CONFIRMATION_REQUESTS.jsonl"], {"qualification_rows": 180, "confirmation_rows": 2160, "model_id": GEMMA, "model_revision": GEMMA_REVISION})


def s1_contract() -> dict[str, Any]:
    return {
        "schema_version": "BWB_S1_CONSTRAINED_ENDPOINT_V1",
        "status": "VERIFIED_RUNTIME_MECHANISM",
        "study_id": "S1_FORMAT_CONTROLLED_CONFIRMATION_20260913",
        "mechanism": "structured_outputs.choice",
        "runtime_version": "vllm==0.28.0",
        "request_field": "extra_body.structured_outputs.choice",
        "response_field": "choices[0].message.content",
        "finite_choice_required": True,
        "fallback_allowed": False,
        "live_smoke_required": True,
    }


def s1_user(task: dict[str, Any], reference: str) -> str:
    return user_text(task, reference, S1_UNIFORM)


def s1_messages(task: dict[str, Any], reference: str, treatment: str) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": s1_user(task, reference)}]
    if treatment == "T1_SILENT_CHECK":
        messages.insert(0, {"role": "system", "content": S1_TREATMENT_TEXT})
    return messages


def s1_request(task: dict[str, Any], reference: str, treatment: str, phase: str, kind: str, key: str, contract_sha: str, duplicate_of: str | None = None) -> dict[str, Any]:
    messages = s1_messages(task, reference, treatment)
    shown = "\n".join(f"<|{message['role']}|>\n{message['content']}\n<|end_{message['role']}|>" for message in messages)
    allowed = list(LETTERS[:len(task["options"])])
    params = {"do_sample": False, "max_model_len": 10496, "max_tokens": 2, "seed": 2701, "temperature": 0.0, "top_k": -1, "top_p": 1.0}
    payload = {"study_id": "S1_FORMAT_CONTROLLED_CONFIRMATION_20260913", "task_id": task["task_id"], "reference": reference, "treatment": treatment, "messages": messages, "allowed_labels": allowed, "generation_parameters": params, "extra_body": {"structured_outputs": {"choice": allowed}}}
    ce = s1_contract()
    ce["allowed_labels"] = allowed
    return {
        "schema_version": "BWB_S1_REQUEST_V1",
        "study_id": "S1_FORMAT_CONTROLLED_CONFIRMATION_20260913",
        "phase": phase,
        "execution_class": kind,
        "request_key": key,
        "request_id": key,
        "task_id": task["task_id"],
        "category": task["category"],
        "dataset": "MMLU-Pro",
        "dataset_revision": REVISION,
        "reference": reference,
        "reference_literal": reference_prefix(reference),
        "treatment": treatment,
        "prompt_hash": sha_bytes(stable_json(messages).encode("utf-8")),
        "rendered_prompt_sha256": sha_bytes(shown.encode("utf-8")),
        "rendered_prompt_text": shown,
        "messages": messages,
        "allowed_labels": allowed,
        "extra_body": {"structured_outputs": {"choice": allowed}},
        "gold_label": task["gold"],
        "gold_answer_index": task["answer_index"],
        "task_prompt_hash": sha_bytes(normalize(task["question"]).encode("utf-8")),
        "task_hash": task_fingerprint(task),
        "model_id": GEMMA,
        "model_revision": GEMMA_REVISION,
        "tokenizer_revision": GEMMA_REVISION,
        "expected_generation_parameters": params,
        "constrained_endpoint": ce,
        "endpoint_mechanism": "structured_outputs.choice",
        "constrained_endpoint_contract_sha256": contract_sha,
        "canonical_payload_sha256": sha_bytes(stable_json(payload).encode("utf-8")),
        "is_inferential_primary": kind == "primary",
        "integrity_duplicate_of": duplicate_of,
    }


def build_s1(panels: dict[str, list[dict[str, Any]]], output: Path) -> dict[str, Any]:
    root = output / "s1"
    primary = panels["s1_primary"]
    qualification = panels["s1_qualification"]
    contract = s1_contract()
    contract_sha = sha_bytes(json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    write_json(root / "CONSTRAINED_ENDPOINT_CONTRACT.json", contract)

    def serialized_task(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "BWB_S1_TASK_V1",
            "study_id": "S1_FORMAT_CONTROLLED_CONFIRMATION_20260913",
            "dataset": "MMLU-Pro",
            "dataset_revision": REVISION,
            "task_id": task["task_id"],
            "source_question_id": task["source_question_id"],
            "source_index": task["source_index"],
            "source": task["source"],
            "category": task["category"],
            "question": task["question"],
            "options": task["options"],
            "answer_index": task["answer_index"],
            "gold": task["gold"],
            "normalized_prompt_hash": sha_bytes(normalize(task["question"] + "\n" + "\n".join(task["options"])).encode("utf-8")),
            "normalized_task_hash": task_fingerprint(task),
        }

    write_jsonl(root / "S1_PRIMARY_TASKS_300.jsonl", [serialized_task(task) for task in primary])
    write_jsonl(root / "S1_QUALIFICATION_TASKS_30.jsonl", [serialized_task(task) for task in qualification])
    quals = []
    for task in qualification:
        for reference in REFERENCES:
            for treatment in TREATMENTS:
                quals.append(s1_request(task, reference, treatment, "qualification", "qualification", f"S1Q::{task['task_id']}::{reference}::{treatment}", contract_sha))
    primaries = []
    for task in primary:
        for reference in REFERENCES:
            for treatment in TREATMENTS:
                primaries.append(s1_request(task, reference, treatment, "production", "primary", f"S1::{task['task_id']}::{reference}::{treatment}::PRIMARY", contract_sha))
    task_by_id = {task["task_id"]: task for task in primary}
    duplicates = []
    for row in sorted(primaries, key=lambda value: value["request_key"])[:360]:
        task = task_by_id[row["task_id"]]
        key = f"S1D::{row['request_key']}::DUP"
        duplicate = s1_request(task, row["reference"], row["treatment"], "production", "integrity_duplicate", key, contract_sha, row["request_key"])
        duplicate["canonical_payload_sha256"] = row["canonical_payload_sha256"]
        duplicates.append(duplicate)
    if (len(quals), len(primaries), len(duplicates)) != (180, 1800, 360):
        raise SystemExit("S1 request counts differ from frozen study design")
    write_jsonl(root / "qualification_requests.jsonl", quals)
    write_jsonl(root / "primary_requests.jsonl", primaries)
    write_jsonl(root / "integrity_duplicate_requests.jsonl", duplicates)
    return write_prepared_manifest(root, ["CONSTRAINED_ENDPOINT_CONTRACT.json", "S1_PRIMARY_TASKS_300.jsonl", "S1_QUALIFICATION_TASKS_30.jsonl", "qualification_requests.jsonl", "primary_requests.jsonl", "integrity_duplicate_requests.jsonl"], {"qualification_rows": 180, "primary_rows": 1800, "integrity_duplicate_rows": 360, "model_id": GEMMA, "model_revision": GEMMA_REVISION, "endpoint": "structured_outputs.choice"})


def write_prepared_manifest(root: Path, names: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    files = {name: {"sha256": sha_file(root / name), "bytes": (root / name).stat().st_size} for name in names}
    result = {"schema_version": "blackwellbench-prepared-local-inputs-v1", "status": "PASS", "contains_source_task_text": True, "contains_task_ids": True, "local_only": True, "files": files, **extra}
    write_json(root / "PREPARED_INPUTS_MANIFEST.json", result)
    return result


def prepare(source: Path, output: Path, *, acknowledge_terms: bool, local_only: bool, tokenizer_cache_dir: Path | None) -> None:
    _, by_fingerprint = load_source(source, acknowledge_terms=acknowledge_terms)
    panels = select_panels(by_fingerprint)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"dataset": "TIGER-Lab/MMLU-Pro", "revision": REVISION, "source_sha256": sha_file(source), "source_url": SOURCE_URL, "local_only": True, "source_terms_reviewed_by_operator": True}
    write_json(output / "SOURCE_PROVENANCE.json", provenance)
    r2 = build_r2(panels["r2_primary"], output, local_only=local_only, tokenizer_cache_dir=tokenizer_cache_dir)
    confirmation = build_confirmation(panels, output, local_only=local_only, tokenizer_cache_dir=tokenizer_cache_dir)
    s1 = build_s1(panels, output)
    write_json(output / "PREPARED_INPUTS_SUMMARY.json", {"source": provenance, "r2": r2, "confirmation": confirmation, "s1": s1, "new_gpu_generations": 0})
    print(json.dumps({"status": "PREPARED_LOCALLY", "output": str(output), "panels": {key: len(value) for key, value in panels.items()}, "new_gpu_generations": 0}, sort_keys=True))


def acquire(destination: Path, *, acknowledge_terms: bool) -> None:
    if not acknowledge_terms:
        raise SystemExit("review the current official source terms and pass --accept-source-terms; this flag does not grant permission")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha_file(destination) != SOURCE_SHA256:
            raise SystemExit("destination already exists with a different hash; remove or rename it manually")
        print(json.dumps({"status": "ALREADY_PRESENT", "sha256": SOURCE_SHA256}, sort_keys=True))
        return
    fd, temp_name = tempfile.mkstemp(prefix=".mmlu-pro.", dir=str(destination.parent))
    os.close(fd)
    temp = Path(temp_name)
    try:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "BlackwellBench-local-preparation/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        observed = sha_file(temp)
        if observed != SOURCE_SHA256:
            raise SystemExit(f"official source hash mismatch; expected {SOURCE_SHA256}, got {observed}")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"status": "ACQUIRED_LOCALLY", "path": str(destination), "sha256": SOURCE_SHA256}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("acquire", help="download the pinned official source into ignored data/local/")
    fetch.add_argument("--destination", type=Path, default=SOURCE_DEFAULT)
    fetch.add_argument("--accept-source-terms", action="store_true", help="operator confirms they reviewed current upstream terms; this is not a permission grant")
    prep = sub.add_parser("prepare", help="build local request ledgers from the pinned source and content commitments")
    prep.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    prep.add_argument("--output", type=Path, default=LOCAL / "prepared")
    prep.add_argument("--accept-source-terms", action="store_true", help="operator confirms they reviewed current upstream terms; this is not a permission grant")
    prep.add_argument("--local-tokenizers-only", action="store_true", help="do not contact model repositories for tokenizer files")
    prep.add_argument("--tokenizer-cache-dir", type=Path, help="optional directory containing the three locked tokenizer subdirectories")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.destination, acknowledge_terms=args.accept_source_terms)
    else:
        prepare(args.source, args.output, acknowledge_terms=args.accept_source_terms, local_only=args.local_tokenizers_only, tokenizer_cache_dir=args.tokenizer_cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
