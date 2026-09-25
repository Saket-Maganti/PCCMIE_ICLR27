#!/usr/bin/env python3
"""Run S1 through the OpenAI-compatible vLLM endpoint.

The live smoke is deliberately mandatory and non-scientific. Any version,
health, or structured-output failure stops the run before a frozen S1 row is
sent. Responses are scored by direct string equality only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

FAIL = "S1_RUNTIME_STRUCTURED_OUTPUT_VERIFICATION_FAIL"
MODEL = "google/gemma-4-12B-it"
VLLM_VERSION = "0.28.0"
CONCURRENCY = 8
TOY_CHOICES = ["ALPHA", "BETA"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def runtime_identity(model: str = MODEL) -> dict[str, str]:
    return {
        "host_id": os.environ.get("S1_HOST_ID", socket.gethostname()),
        "gpu_uuid": os.environ.get("S1_GPU_UUID", "RECORD_ON_HOST"),
        "runtime_id": os.environ.get("S1_RUNTIME_ID", f"vllm-{VLLM_VERSION}"),
        "model_id": model,
        "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    }


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_gate_receipt(root: Path, identity: dict[str, str]) -> dict:
    receipt_path = root / "S1_GATE_RECEIPT.json"
    if not receipt_path.is_file():
        raise RuntimeError("S1 production is blocked until smoke, both invariance probes, and qualification pass")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS" or receipt.get("schema_version") != "bwb-s1-production-gate-v1":
        raise RuntimeError("S1 production gate receipt is not PASS")
    if receipt.get("prepared_manifest_sha256") != sha_file(root / "PREPARED_INPUTS_MANIFEST.json"):
        raise RuntimeError("S1 prepared input package changed after gate validation")
    if receipt.get("endpoint_contract_sha256") != sha_file(root / "CONSTRAINED_ENDPOINT_CONTRACT.json"):
        raise RuntimeError("S1 endpoint contract changed after gate validation")
    for field, value in identity.items():
        if receipt.get("identity", {}).get(field) != value:
            raise RuntimeError("S1 production host or model identity differs from the passing gate receipt")
    for filename, field in (
        ("LIVE_SMOKE.json", "smoke_sha256"),
        ("LIVE_INVARIANCE_sequential.json", "sequential_invariance_sha256"),
        ("LIVE_INVARIANCE_concurrent.json", "concurrent_invariance_sha256"),
        ("qualification_results.jsonl", "qualification_results_sha256"),
        ("qualification_requests.jsonl", "qualification_requests_sha256"),
    ):
        path = root / filename
        if not path.is_file() or sha_file(path) != receipt.get(field):
            raise RuntimeError("S1 gate evidence changed after validation")
    return receipt


def require_invariance_receipts(root: Path) -> None:
    smoke = read_json(root / "LIVE_SMOKE.json")
    identity = runtime_identity(str(smoke.get("model_id", MODEL)))
    for name in ("sequential", "concurrent"):
        probe = read_json(root / f"LIVE_INVARIANCE_{name}.json")
        if probe.get("status") != "PASS" or probe.get("n") != 32 or len(probe.get("outputs", [])) != 32:
            raise RuntimeError("S1 qualification requires passing 32-run invariance receipts")
        if not probe.get("finite_choice_verified") or len(set(probe["outputs"])) != 1:
            raise RuntimeError("S1 invariance receipt is not deterministic finite-choice")
        for field, value in identity.items():
            if probe.get(field) != value:
                raise RuntimeError("S1 invariance receipt identity differs from the smoke run")
    sequential = read_json(root / "LIVE_INVARIANCE_sequential.json")["outputs"]
    concurrent = read_json(root / "LIVE_INVARIANCE_concurrent.json")["outputs"]
    if sequential != concurrent:
        raise RuntimeError("S1 sequential and concurrent invariance outputs differ")


def fail(root: Path, detail: str) -> int:
    payload = {"status": FAIL, "detail": detail, "scientific_requests_sent": 0, "new_gpu_generations": 0}
    write_json(root / "LIVE_SMOKE.json", payload)
    status = read_json(root / "LIVE_STATUS.json") if (root / "LIVE_STATUS.json").exists() else {}
    status.update({"phase": FAIL, "endpoint_invalid_rows": 0, "in_flight_requests": 0})
    write_json(root / "LIVE_STATUS.json", status)
    print(f"{FAIL}: {detail}", file=sys.stderr)
    return 2


def load_contract(root: Path) -> dict:
    contract = read_json(root / "CONSTRAINED_ENDPOINT_CONTRACT.json")
    forbidden = ("guided_choice", "guided_regex", "guided_json", "guided_grammar", "guided_whitespace_pattern")
    request_blob = json.dumps({"mechanism": contract.get("mechanism"), "request_field": contract.get("request_field")}, sort_keys=True)
    if contract.get("status") != "VERIFIED_RUNTIME_MECHANISM":
        raise RuntimeError("contract is not VERIFIED_RUNTIME_MECHANISM")
    if contract.get("mechanism") != "structured_outputs.choice":
        raise RuntimeError("contract mechanism mismatch")
    if contract.get("request_field") != "extra_body.structured_outputs.choice":
        raise RuntimeError("contract request field mismatch")
    if contract.get("fallback_allowed") is not False or any(x in request_blob for x in forbidden):
        raise RuntimeError("legacy or fallback mechanism present")
    return contract


def task_allowed_labels(task: dict) -> list[str]:
    options = task.get("options")
    if not isinstance(options, list) or not options or len(options) > 26:
        raise RuntimeError(f"invalid frozen option list for task {task.get('task_id')}")
    return [chr(ord("A") + i) for i in range(len(options))]


def build_extra_body(row: dict, task: dict) -> dict:
    derived = task_allowed_labels(task)
    allowed = row.get("allowed_labels")
    if allowed != derived:
        raise RuntimeError(f"allowed_labels drift for task {row.get('task_id')}")
    return {"structured_outputs": {"choice": list(allowed)}}


def build_openai_payload(row: dict, task: dict) -> dict:
    return {
        "model": row["model_id"],
        "messages": row["messages"],
        "max_tokens": row["expected_generation_parameters"]["max_tokens"],
        "temperature": row["expected_generation_parameters"]["temperature"],
        "top_p": row["expected_generation_parameters"]["top_p"],
        "seed": row["expected_generation_parameters"]["seed"],
        "extra_body": build_extra_body(row, task),
    }


def endpoint_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value[:-3].rstrip("/") if value.endswith("/v1") else value


def server_version(base_url: str, api_key: str) -> str:
    url = os.environ.get("VLLM_VERSION_URL", endpoint_root(base_url) + "/version")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return str(data.get("version") or data.get("data", {}).get("version") or "")
    return ""


def completion_text(completion) -> str:
    # Direct field access; no normalization, regex, or semantic parsing.
    return completion.choices[0].message.content


def call_choice(client, model: str, messages: list[dict], choices: list[str], *, max_tokens: int = 2) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=2701,
        extra_body={"structured_outputs": {"choice": list(choices)}},
    )
    value = completion_text(completion)
    if value not in choices:
        raise RuntimeError(f"engine returned value outside finite choice set: {value!r}")
    return value


def open_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"OpenAI client unavailable: {exc}") from exc
    return OpenAI(base_url=base_url, api_key=api_key)


def smoke(root: Path, contract: dict, base_url: str, api_key: str):
    client = open_client(base_url, api_key)
    version = server_version(base_url, api_key)
    if version != VLLM_VERSION:
        raise RuntimeError(f"vLLM version reported {version!r}, expected {VLLM_VERSION}")
    models = client.models.list()
    model_ids = [getattr(x, "id", None) for x in getattr(models, "data", [])]
    model = os.environ.get("VLLM_SERVED_MODEL_NAME", MODEL)
    if model not in model_ids:
        raise RuntimeError(f"healthy server did not expose expected model {model!r}: {model_ids!r}")
    messages = [{"role": "user", "content": "NON-SCIENTIFIC SMOKE. Choose one token: ALPHA or BETA."}]
    outputs = [call_choice(client, model, messages, TOY_CHOICES) for _ in range(3)]
    result = {
        "status": "PASS",
        "vllm_version": version,
        "server_healthy": True,
        "model_id": model,
        **runtime_identity(model),
        "structured_output_field": "extra_body.structured_outputs.choice",
        "toy_choice_set": TOY_CHOICES,
        "toy_outputs": outputs,
        "finite_choice_verified": all(x in TOY_CHOICES for x in outputs),
        "guided_choice_present": False,
        "unconstrained_fallback": False,
        "scientific_requests_sent": 0,
        "new_gpu_generations": 0,
    }
    write_json(root / "LIVE_SMOKE.json", result)
    return client, model


def load_rows(root: Path, phase: str) -> list[dict]:
    names = {
        "qualification": ["qualification_requests.jsonl"],
        "production": ["primary_requests.jsonl", "integrity_duplicate_requests.jsonl"],
    }.get(phase)
    if names is None:
        raise RuntimeError(f"no scientific request ledger for phase {phase}")
    rows = []
    for name in names:
        rows.extend(json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def load_tasks(root: Path) -> dict[str, dict]:
    tasks = {}
    for name in ("S1_PRIMARY_TASKS_300.jsonl", "S1_QUALIFICATION_TASKS_30.jsonl"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                task = json.loads(line)
                tasks[task["task_id"]] = task
    return tasks


def result_row(row: dict, selected: str) -> dict:
    return {
        "schema_version": "BWB_S1_RESULT_V1",
        "request_key": row["request_key"],
        "request_id": row["request_id"],
        "phase": row["phase"],
        "execution_class": row["execution_class"],
        "task_id": row["task_id"],
        "category": row["category"],
        "reference": row["reference"],
        "treatment": row["treatment"],
        "allowed_labels": list(row["allowed_labels"]),
        "selected_constrained_label": selected,
        "gold_label": row["gold_label"],
        "correctness": selected == row["gold_label"],
        "constraint_active": True,
        "endpoint_mechanism": "structured_outputs.choice",
        "constrained_endpoint_contract_sha256": row["constrained_endpoint_contract_sha256"],
        "integrity_duplicate_of": row.get("integrity_duplicate_of"),
        "host_id": os.environ.get("S1_HOST_ID", socket.gethostname()),
        "gpu_uuid": os.environ.get("S1_GPU_UUID", "RECORD_ON_HOST"),
        "runtime_id": os.environ.get("S1_RUNTIME_ID", "vllm-0.28.0"),
        "model_id": row["model_id"],
        "model_revision": row["model_revision"],
    }


def run_scientific(root: Path, client, phase: str, concurrency: int) -> int:
    tasks = load_tasks(root)
    rows = load_rows(root, phase)
    for row in rows:
        if row["task_id"] not in tasks:
            raise RuntimeError(f"task not found in frozen manifests: {row['task_id']}")
        build_openai_payload(row, tasks[row["task_id"]])

    def one(row: dict) -> dict:
        payload = build_openai_payload(row, tasks[row["task_id"]])
        choices = payload["extra_body"]["structured_outputs"]["choice"]
        selected = call_choice(client, payload["model"], payload["messages"], choices, max_tokens=payload["max_tokens"])
        return result_row(row, selected)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, rows))
    out_name = "qualification_results.jsonl" if phase == "qualification" else "production_results.jsonl"
    (root / out_name).write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in results), encoding="utf-8")
    status = read_json(root / "LIVE_STATUS.json") if (root / "LIVE_STATUS.json").exists() else {}
    status.update({"phase": phase, "expected_rows": len(rows), "durable_rows": len(results), "in_flight_requests": 0, "endpoint_invalid_rows": 0})
    write_json(root / "LIVE_STATUS.json", status)
    return 0


def run_invariance(root: Path, client, model: str, concurrent_mode: bool) -> int:
    messages = [{"role": "user", "content": "NON-SCIENTIFIC INVARIANCE PROBE. Choose one token: ALPHA or BETA."}]
    if concurrent_mode:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            outputs = list(pool.map(lambda _: call_choice(client, model, messages, TOY_CHOICES), range(32)))
        key = "concurrent"
    else:
        outputs = [call_choice(client, model, messages, TOY_CHOICES) for _ in range(32)]
        key = "sequential"
    result = {"status": "PASS", "probe": key, "n": 32, "outputs": outputs, "finite_choice_verified": all(x in TOY_CHOICES for x in outputs), **runtime_identity(model), "scientific_requests_sent": 0, "new_gpu_generations": 0}
    write_json(root / f"LIVE_INVARIANCE_{key}.json", result)
    return 0


def preflight(root: Path, phase: str) -> int:
    import hashlib

    manifest_path = root / "PREPARED_INPUTS_MANIFEST.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("status") != "PASS" or manifest.get("local_only") is not True:
            raise RuntimeError("prepared input manifest is not a verified local-only package")
        for rel, record in manifest.get("files", {}).items():
            path = root / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
                raise RuntimeError(f"prepared input hash mismatch: {rel}")
    contract = load_contract(root)
    contract_sha = hashlib.sha256((root / "CONSTRAINED_ENDPOINT_CONTRACT.json").read_bytes()).hexdigest()
    tasks = load_tasks(root)
    if len(tasks) != 330:
        raise RuntimeError(f"frozen S1 task set has {len(tasks)} unique tasks; expected 330")
    rows = load_rows(root, phase)
    expected = 180 if phase == "qualification" else 2160
    if len(rows) != expected:
        raise RuntimeError(f"frozen S1 {phase} request count mismatch: expected {expected}, got {len(rows)}")
    seen = set()
    for row in rows:
        key = row.get("request_key")
        if not isinstance(key, str) or not key or key in seen:
            raise RuntimeError("frozen S1 request key is empty or duplicated")
        seen.add(key)
        task = tasks.get(row.get("task_id"))
        if task is None:
            raise RuntimeError("frozen S1 request does not map to a prepared task")
        build_openai_payload(row, task)
        if row.get("constrained_endpoint_contract_sha256") != contract_sha:
            raise RuntimeError("frozen S1 endpoint contract hash mismatch")
    if phase == "production":
        if sum(row.get("execution_class") == "primary" for row in rows) != 1800:
            raise RuntimeError("S1 production primary row count mismatch")
        if sum(row.get("execution_class") == "integrity_duplicate" for row in rows) != 360:
            raise RuntimeError("S1 production integrity duplicate row count mismatch")
    gate_valid = False
    if phase == "production":
        try:
            verify_gate_receipt(root, runtime_identity())
            gate_valid = True
        except Exception:
            gate_valid = False
    print(json.dumps({"status": "PREFLIGHT_PASS", "phase": phase, "frozen_requests": len(rows), "task_count": len(tasks), "endpoint": contract["mechanism"], "production_gate_valid": gate_valid, "network_requests": 0, "new_gpu_generations": 0}, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["smoke", "invariance_sequential", "invariance_concurrent", "qualification", "production"])
    ap.add_argument("--package-root", default=str(Path(__file__).parent))
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--preflight-only", action="store_true", help="validate frozen ledgers without contacting the GPU endpoint")
    args = ap.parse_args()
    root = Path(args.package_root).resolve()
    try:
        contract = load_contract(root)
        if args.preflight_only:
            if args.phase not in {"qualification", "production"}:
                raise RuntimeError("--preflight-only requires qualification or production phase")
            return preflight(root, args.phase)
        if args.phase in {"qualification", "production"}:
            preflight(root, args.phase)
        receipt = None
        if args.phase == "production":
            receipt = verify_gate_receipt(root, runtime_identity(os.environ.get("VLLM_SERVED_MODEL_NAME", MODEL)))
        client, model = smoke(root, contract, args.base_url, args.api_key)
        if args.phase == "smoke":
            print("S1_RUNTIME_STRUCTURED_OUTPUT_VERIFICATION_PASS")
            return 0
        if args.phase == "invariance_sequential":
            return run_invariance(root, client, model, False)
        if args.phase == "invariance_concurrent":
            return run_invariance(root, client, model, True)
        if args.phase == "qualification":
            require_invariance_receipts(root)
        if receipt is not None and model != receipt.get("identity", {}).get("model_id"):
            raise RuntimeError("live served model differs from the S1 gate receipt")
        return run_scientific(root, client, args.phase, args.concurrency)
    except Exception as exc:
        return fail(root, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
