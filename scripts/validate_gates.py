#!/usr/bin/env python3
"""Validate the frozen confirmation and S1 production gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

from score_outputs import jsonl, projection_clusters, sha_file, strict_score, verified_package

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
VLLM_VERSION = "0.28.0"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def stable_sha(value: dict[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_confirmation_runtime(path: Path) -> None:
    runtime = json.loads(path.read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "configs/experiment_locks.json").read_text(encoding="utf-8"))["gemma_confirmation"]
    expected = lock["runtime"].split("==", 1)[1]
    observed = runtime.get("vllm_version") or runtime.get("vllm")
    for container in (runtime.get("versions"), runtime.get("expected"), runtime.get("lock")):
        if observed is None and isinstance(container, dict):
            observed = container.get("vllm_version") or container.get("vllm")
    if str(observed) != expected:
        raise SystemExit("runtime manifest does not attest the frozen vLLM version")


def confirmation_qualification(args: argparse.Namespace) -> int:
    errors: list[str] = []
    package = args.package.resolve()
    try:
        manifest = verified_package(package)
        requests = jsonl(package / "QUALIFICATION_REQUESTS.jsonl")
        outputs = jsonl(args.results)
        lock_rows = jsonl(args.data_dir / "confirmation_qualification_lock.jsonl")
        panel = json.loads((args.data_dir / "input_panel_manifest.json").read_text(encoding="utf-8"))["panels"]["r2_primary"]
        order_hashes = panel.get("analysis_order_hashes", panel["ordered_hashes"])
        order_by_hash = {digest: order for order, digest in enumerate(order_hashes)}
        r2_clusters = projection_clusters(args.data_dir, "r2_scored_outcomes.csv")
        expected = {
            (row["cluster_id"], row["reference_id"], row["treatment_id"], row["execution_class"]): row["strict_endpoint_sha256"]
            for row in lock_rows
        }
        request_by_key = {str(row["request_key"]): row for row in requests}
        identity = lambda row: (str(row.get("phase", "")), str(row.get("task_id", "")), str(row.get("reference_id", "")), str(row.get("treatment_id", row.get("treatment", ""))), str(row.get("execution_class", "")))
        request_by_identity = {identity(row): row for row in requests}
        output_by_key: dict[str, dict[str, Any]] = {}
        endpoints: dict[tuple[str, str, str], dict[str, tuple[bool, str | None, bool]]] = {}
        for output in outputs:
            key = str(output.get("request_key", ""))
            request = request_by_key.get(key)
            exact_key = request is not None
            if request is None:
                request = request_by_identity.get(identity(output))
            if request is None:
                errors.append("qualification output contains an unexpected request identity")
                continue
            frozen_key = str(request["request_key"])
            if frozen_key in output_by_key:
                errors.append("qualification output contains duplicate request identities")
                continue
            output_by_key[frozen_key] = output
            if identity(output) != identity(request):
                errors.append("qualification output metadata differs from the frozen request")
            for field in ("model_id", "model_revision", "tokenizer_revision", "prompt_hash"):
                if output.get(field) != request.get(field):
                    errors.append("qualification output request metadata differs from the frozen request")
            if exact_key and output.get("request_sha256") != request.get("request_sha256"):
                errors.append("qualification request hash differs from the frozen request")
            raw = output.get("raw_response")
            if not isinstance(raw, str):
                errors.append("qualification raw response is missing")
                continue
            if output.get("response_hash") not in (None, hashlib.sha256(raw.encode("utf-8")).hexdigest()):
                errors.append("qualification response hash mismatch")
            strict = strict_score(raw, str(request["gold"]))
            endpoint = {"strict_parsed": bool(strict["parsed"]), "strict_pred": strict["pred"], "strict_correct": bool(strict["correct"])}
            order = order_by_hash.get(str(request.get("task_hash", "")))
            if order is None or order not in r2_clusters:
                errors.append("qualification task is absent from the frozen R2 anonymous panel")
                continue
            cell = (r2_clusters[order], request["reference_id"], request["treatment_id"], request["execution_class"])
            digest = stable_sha(endpoint)
            if expected.get(cell) != digest:
                errors.append("qualification strict endpoint differs from the frozen endpoint commitment")
            task_cell = (str(request["task_id"]), str(request["reference_id"]), str(request["treatment_id"]))
            endpoints.setdefault(task_cell, {})[str(request["execution_class"])] = (endpoint["strict_parsed"], endpoint["strict_pred"], endpoint["strict_correct"])
        missing = len(requests) - len(output_by_key)
        if len(requests) != 180 or len(request_by_key) != len(requests) or len(request_by_identity) != len(requests):
            errors.append("frozen qualification request keyspace is malformed")
        if len(outputs) != 180 or missing:
            errors.append("qualification output does not contain the complete 180-row keyspace")
        if len(lock_rows) != 180 or len(expected) != 180:
            errors.append("frozen qualification endpoint lock is malformed")
        duplicate_checks = 0
        for cells in endpoints.values():
            primary = cells.get("primary")
            duplicate = cells.get("integrity_duplicate")
            if duplicate is not None:
                duplicate_checks += 1
                if primary is None or duplicate != primary:
                    errors.append("qualification integrity duplicate strict endpoint mismatch")
        if duplicate_checks != 36:
            errors.append("qualification integrity duplicate keyspace is incomplete")
        if args.confirmation_results and args.confirmation_results.is_file():
            for row in jsonl(args.confirmation_results):
                if row.get("phase") == "confirmation":
                    errors.append("confirmation outputs exist before qualification authorization")
                    break
        status = "PASS" if not errors else "FAIL"
        result = {
            "schema_version": "bwb-confirmation-qualification-validation-v1",
            "status": status,
            "errors": sorted(set(errors)),
            "expected_requests": 180,
            "observed_outputs": len(outputs),
            "missing_outputs": max(0, missing),
            "integrity_duplicate_endpoint_checks": duplicate_checks,
            "text_equality_is_diagnostic_only": True,
            "confirmation_outputs_forbidden_until_pass": True,
            "qualification_results": str(args.results.resolve()),
            "qualification_results_sha256": sha_file(args.results),
            "prepared_manifest_sha256": sha_file(package / "PREPARED_INPUTS_MANIFEST.json"),
            "qualification_requests_sha256": sha_file(package / "QUALIFICATION_REQUESTS.jsonl"),
            "confirmation_requests_sha256": sha_file(package / "CONFIRMATION_REQUESTS.jsonl"),
            "input_panel_manifest_sha256": sha_file(args.data_dir / "input_panel_manifest.json"),
            "r2_projection_sha256": sha_file(args.data_dir / "r2_scored_outcomes.csv"),
            "qualification_lock_sha256": sha_file(args.data_dir / "confirmation_qualification_lock.jsonl"),
            "model_id": manifest.get("model_id", MODEL),
            "model_revision": manifest.get("model_revision", REVISION),
        }
        write_json(args.out, result)
        print(json.dumps({"status": status, "expected_requests": 180, "observed_outputs": len(outputs), "duplicate_checks": duplicate_checks, "errors": len(result["errors"]), "out": str(args.out)}, sort_keys=True))
        return 0 if status == "PASS" else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        write_json(args.out, {"schema_version": "bwb-confirmation-qualification-validation-v1", "status": "FAIL", "errors": [str(exc)], "expected_requests": 180, "observed_outputs": 0, "missing_outputs": 180})
        print(json.dumps({"status": "FAIL", "errors": 1, "out": str(args.out)}, sort_keys=True))
        return 1


def confirmation_lock(args: argparse.Namespace) -> int:
    package = args.package.resolve()
    manifest = verified_package(package)
    validation_path = args.validation.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise SystemExit("production authorization requires a passing qualification validation")
    results_path = Path(validation["qualification_results"])
    if not results_path.is_file() or sha_file(results_path) != validation.get("qualification_results_sha256"):
        raise SystemExit("qualification results changed after validation")
    recheck_path = package / ".qualification_validation_recheck.json"
    recheck_args = argparse.Namespace(package=package, results=results_path, confirmation_results=None, data_dir=ROOT / "data", out=recheck_path)
    try:
        recheck_status = confirmation_qualification(recheck_args)
        rechecked = json.loads(recheck_path.read_text(encoding="utf-8"))
    finally:
        recheck_path.unlink(missing_ok=True)
    if recheck_status != 0 or rechecked != validation:
        raise SystemExit("qualification criteria do not reproduce the supplied PASS validation")
    if validation.get("prepared_manifest_sha256") != sha_file(package / "PREPARED_INPUTS_MANIFEST.json"):
        raise SystemExit("prepared input package changed after qualification validation")
    if validation.get("qualification_requests_sha256") != sha_file(package / "QUALIFICATION_REQUESTS.jsonl") or validation.get("confirmation_requests_sha256") != sha_file(package / "CONFIRMATION_REQUESTS.jsonl"):
        raise SystemExit("frozen confirmation requests changed after qualification validation")
    if not all((args.host_id, args.instance_id, args.gpu_uuid)):
        raise SystemExit("start lock requires physical host, provider instance, and GPU identities")
    runtime = args.runtime_manifest.resolve()
    if not runtime.is_file():
        raise SystemExit("runtime manifest is missing")
    verify_confirmation_runtime(runtime)
    payload = {
        "schema_version": "bwb-confirmation-production-start-lock-v1",
        "status": "LOCKED",
        "model_id": manifest.get("model_id", MODEL),
        "model_revision": manifest.get("model_revision", REVISION),
        "tokenizer_revision": REVISION,
        "host_id": args.host_id,
        "instance_id": args.instance_id,
        "gpu_uuid": args.gpu_uuid,
        "qualification_validation_sha256": sha_file(validation_path),
        "qualification_results_sha256": validation["qualification_results_sha256"],
        "prepared_manifest_sha256": sha_file(package / "PREPARED_INPUTS_MANIFEST.json"),
        "qualification_requests_sha256": sha_file(package / "QUALIFICATION_REQUESTS.jsonl"),
        "confirmation_requests_sha256": sha_file(package / "CONFIRMATION_REQUESTS.jsonl"),
        "input_panel_manifest_sha256": sha_file(ROOT / "data/input_panel_manifest.json"),
        "qualification_lock_sha256": sha_file(ROOT / "data/confirmation_qualification_lock.jsonl"),
        "runtime_manifest_sha256": sha_file(runtime),
        "runtime_manifest": str(runtime),
    }
    if args.out.exists():
        if json.loads(args.out.read_text(encoding="utf-8")) != payload:
            raise SystemExit("existing start lock is immutable and differs from this authorization")
    else:
        write_json(args.out, payload)
    print(json.dumps({"status": "LOCKED", "new_gpu_generations": 0, "out": str(args.out)}, sort_keys=True))
    return 0


def s1_identity() -> dict[str, str]:
    return {
        "host_id": os.environ.get("S1_HOST_ID", socket.gethostname()),
        "gpu_uuid": os.environ.get("S1_GPU_UUID", "RECORD_ON_HOST"),
        "runtime_id": os.environ.get("S1_RUNTIME_ID", f"vllm-{VLLM_VERSION}"),
        "model_id": MODEL,
        "model_revision": REVISION,
    }


def check_s1_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verified_package(root)
    contract_path = root / "CONSTRAINED_ENDPOINT_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("mechanism") != "structured_outputs.choice" or contract.get("fallback_allowed") is not False:
        raise ValueError("S1 endpoint contract does not match the frozen constrained-choice mechanism")
    if manifest.get("model_id") != MODEL or manifest.get("model_revision") != REVISION:
        raise ValueError("S1 model identity differs from the frozen lock")
    return manifest, contract


def s1_receipt(args: argparse.Namespace) -> int:
    root = args.package_root.resolve()
    manifest, contract = check_s1_inputs(root)
    expected_identity = s1_identity()
    if expected_identity["gpu_uuid"] == "RECORD_ON_HOST":
        raise SystemExit("S1 gate receipt requires an explicit S1_GPU_UUID")
    smoke = json.loads((root / "LIVE_SMOKE.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS" or smoke.get("vllm_version") != VLLM_VERSION or smoke.get("model_id") != MODEL or smoke.get("model_revision") != REVISION:
        raise SystemExit("S1 smoke did not pass the frozen runtime and model checks")
    if smoke.get("structured_output_field") != "extra_body.structured_outputs.choice" or smoke.get("guided_choice_present") is not False or smoke.get("unconstrained_fallback") is not False:
        raise SystemExit("S1 smoke did not verify the frozen structured-output mechanism")
    if len(smoke.get("toy_outputs", [])) != 3 or not smoke.get("finite_choice_verified") or any(value not in {"ALPHA", "BETA"} for value in smoke["toy_outputs"]):
        raise SystemExit("S1 smoke toy choices are invalid")
    for field, value in expected_identity.items():
        if smoke.get(field) != value:
            raise SystemExit("S1 smoke identity differs from the current release host")
    identities = [expected_identity]
    probe_files = {}
    for name in ("sequential", "concurrent"):
        path = root / f"LIVE_INVARIANCE_{name}.json"
        probe = json.loads(path.read_text(encoding="utf-8"))
        if probe.get("status") != "PASS" or probe.get("n") != 32 or len(probe.get("outputs", [])) != 32 or not probe.get("finite_choice_verified"):
            raise SystemExit("S1 invariance probe is incomplete")
        if any(value not in {"ALPHA", "BETA"} for value in probe["outputs"]) or len(set(probe["outputs"])) != 1:
            raise SystemExit("S1 invariance output is not deterministic finite-choice")
        for field, value in expected_identity.items():
            if probe.get(field) != value:
                raise SystemExit("S1 invariance identity differs from the current release host")
        probe_files[name] = sha_file(path)
        identities.append({field: probe[field] for field in expected_identity})
    if smoke["toy_outputs"][0] != json.loads((root / "LIVE_INVARIANCE_sequential.json").read_text(encoding="utf-8"))["outputs"][0]:
        raise SystemExit("S1 smoke and invariance probes disagree")
    sequential = json.loads((root / "LIVE_INVARIANCE_sequential.json").read_text(encoding="utf-8"))["outputs"]
    concurrent = json.loads((root / "LIVE_INVARIANCE_concurrent.json").read_text(encoding="utf-8"))["outputs"]
    if sequential != concurrent:
        raise SystemExit("S1 sequential and concurrent invariance outputs differ")
    requests = jsonl(root / "qualification_requests.jsonl")
    results_path = root / "qualification_results.jsonl"
    results = jsonl(results_path)
    by_key = {str(row.get("request_key", "")): row for row in results}
    expected_by_key = {str(row["request_key"]): row for row in requests}
    if len(requests) != 180 or len(expected_by_key) != 180 or len(results) != 180 or set(by_key) != set(expected_by_key):
        raise SystemExit("S1 qualification does not contain the exact frozen 180-row keyspace")
    contract_sha = sha_file(root / "CONSTRAINED_ENDPOINT_CONTRACT.json")
    for key, request in expected_by_key.items():
        result = by_key[key]
        if result.get("phase") != "qualification" or result.get("task_id") != request.get("task_id") or result.get("reference") != request.get("reference") or result.get("treatment") != request.get("treatment"):
            raise SystemExit("S1 qualification output identity differs from its request")
        if result.get("selected_constrained_label") not in request.get("allowed_labels", []):
            raise SystemExit("S1 qualification output escaped its finite label set")
        if result.get("constraint_active") is not True or result.get("endpoint_mechanism") != "structured_outputs.choice":
            raise SystemExit("S1 qualification output did not use the frozen constrained endpoint")
        if result.get("constrained_endpoint_contract_sha256") != contract_sha or request.get("constrained_endpoint_contract_sha256") != contract_sha:
            raise SystemExit("S1 qualification endpoint contract hash mismatch")
        for field, value in expected_identity.items():
            output_field = field
            if result.get(output_field) != value:
                raise SystemExit("S1 qualification identity differs from the current release host")
        if result.get("correctness") is not (result.get("selected_constrained_label") == result.get("gold_label")):
            raise SystemExit("S1 qualification correctness field is inconsistent")
    status_path = root / "LIVE_STATUS.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("phase") != "qualification" or status.get("durable_rows") != 180 or status.get("endpoint_invalid_rows") != 0:
            raise SystemExit("S1 qualification runner status is not a complete PASS")
    receipt = {
        "schema_version": "bwb-s1-production-gate-v1",
        "status": "PASS",
        "identity": expected_identity,
        "vllm_version": VLLM_VERSION,
        "endpoint_mechanism": contract["mechanism"],
        "prepared_manifest_sha256": sha_file(root / "PREPARED_INPUTS_MANIFEST.json"),
        "endpoint_contract_sha256": contract_sha,
        "smoke_sha256": sha_file(root / "LIVE_SMOKE.json"),
        "sequential_invariance_sha256": probe_files["sequential"],
        "concurrent_invariance_sha256": probe_files["concurrent"],
        "qualification_results_sha256": sha_file(results_path),
        "qualification_requests_sha256": sha_file(root / "qualification_requests.jsonl"),
        "smoke_toy_generations": 3,
        "invariance_runs_each": 32,
        "qualification_rows": 180,
        "new_gpu_generations": 0,
    }
    write_json(args.out or root / "S1_GATE_RECEIPT.json", receipt)
    print(json.dumps({"status": "PASS", "smoke": 3, "sequential": 32, "concurrent": 32, "qualification": 180, "new_gpu_generations": 0, "out": str(args.out or root / 'S1_GATE_RECEIPT.json')}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    qual = sub.add_parser("confirmation-qualification")
    qual.add_argument("--package", type=Path, default=ROOT / "data/local/prepared/confirmation")
    qual.add_argument("--results", type=Path, required=True)
    qual.add_argument("--confirmation-results", type=Path)
    qual.add_argument("--data-dir", type=Path, default=ROOT / "data")
    qual.add_argument("--out", type=Path, required=True)
    lock = sub.add_parser("confirmation-lock")
    lock.add_argument("--package", type=Path, default=ROOT / "data/local/prepared/confirmation")
    lock.add_argument("--validation", type=Path, required=True)
    lock.add_argument("--runtime-manifest", type=Path, required=True)
    lock.add_argument("--host-id", required=True)
    lock.add_argument("--instance-id", required=True)
    lock.add_argument("--gpu-uuid", required=True)
    lock.add_argument("--out", type=Path, required=True)
    s1 = sub.add_parser("s1")
    s1.add_argument("--package-root", type=Path, default=ROOT / "data/local/prepared/s1")
    s1.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "confirmation-qualification":
        return confirmation_qualification(args)
    if args.command == "confirmation-lock":
        return confirmation_lock(args)
    return s1_receipt(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"gate validation failed closed: {exc}") from None
