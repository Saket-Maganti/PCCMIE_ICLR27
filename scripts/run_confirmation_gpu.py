#!/usr/bin/env python3
"""Run one frozen Gemma request phase against the local vLLM endpoint.

Only completed raw response rows are durable.  This runner never computes or
prints strict/semantic effects; those are added by the frozen validators and
analyzer after the gated phase completes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from checkpoint_manager import (  # noqa: E402
    IDENTITY_FIELDS,
    append_row,
    load_jsonl,
    request_map,
    sha256_file,
    validate_ledger_identity,
    write_checkpoint_manifest,
    write_json_atomic,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc: BaseException) -> str:
    # Do not persist a server body or a credential-bearing URL.
    value = f"{type(exc).__name__}: {exc}"
    for needle in ("hf_", "Bearer ", "Authorization:"):
        if needle in value:
            value = value.split(needle, 1)[0] + "[REDACTED]"
    return value[:500]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        # Read and discard the body; it can contain request data or credentials.
        try:
            exc.read()
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}") from None
    if status < 200 or status >= 300:
        raise RuntimeError(f"HTTP {status}")
    try:
        value = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON response: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise RuntimeError("JSON response is not an object")
    return value


def check_request_contract(request: dict[str, Any]) -> None:
    expected = request.get("expected_generation_parameters")
    if expected != {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 8192,
        "max_model_len": 10496,
        "seed": 2701,
    }:
        raise ValueError(f"frozen generation parameters mismatch for {request.get('request_key')}")
    canonical = request.get("canonical_serialized_request")
    stable = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    import hashlib
    if hashlib.sha256(stable.encode("utf-8")).hexdigest() != request.get("request_sha256"):
        raise ValueError(f"frozen request hash mismatch for {request.get('request_key')}")
    if canonical.get("messages") != request.get("messages"):
        raise ValueError(f"frozen message mismatch for {request.get('request_key')}")


def one_request(
    request: dict[str, Any],
    *,
    base_url: str,
    timeout: float,
    retries: int,
    attempt_id: str,
    instance_id: str,
    host_id: str,
    gpu_uuid: str,
    runtime_manifest_sha256: str,
) -> dict[str, Any]:
    check_request_contract(request)
    params = request["expected_generation_parameters"]
    payload = {
        "model": request["model_id"],
        "messages": request["messages"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "max_tokens": params["max_tokens"],
        "seed": params["seed"],
        "stream": False,
    }
    last: BaseException | None = None
    response: dict[str, Any] | None = None
    for retry in range(retries + 1):
        try:
            response = post_json(f"{base_url.rstrip('/')}/chat/completions", payload, timeout)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if retry < retries:
                time.sleep(min(60.0, 2.0 ** retry))
    if response is None:
        raise RuntimeError(safe_error(last or RuntimeError("request failed")))
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("response has no choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("response choice has no text content")
    raw_response = message["content"]
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("response has no usage object")
    try:
        output_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("response has no completion_tokens") from None
    input_tokens = usage.get("prompt_tokens")
    try:
        input_tokens = int(input_tokens) if input_tokens is not None else None
    except (TypeError, ValueError):
        input_tokens = None
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise RuntimeError("response has no finish_reason")
    import hashlib
    row = {field: request.get(field) for field in IDENTITY_FIELDS}
    row.update(
        {
            "schema_version": "GEMMA_RTX5090_OUTPUT_ROW_V1",
            "raw_response": raw_response,
            "response_hash": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "finish_reason": finish_reason,
            "completed_at_utc": now_utc(),
            "attempt_id": attempt_id,
            "instance_id": str(instance_id),
            "host_id": str(host_id),
            "gpu_uuid": gpu_uuid,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "scientific_outcomes_written": False,
        }
    )
    return row


def write_status(path: Path, phase: str, *, started: str, expected: int, observed: int, failed: int, tokens: int, errors: list[str]) -> None:
    payload = {
        "schema_version": "GEMMA_RTX5090_PHASE_STATUS_V1",
        "phase": phase,
        "started_at_utc": started,
        "updated_at_utc": now_utc(),
        "expected_rows": expected,
        "observed_rows": observed,
        "failed_requests": failed,
        "output_tokens": tokens,
        "scientific_outcomes_written": False,
        "errors": errors[-5:],
    }
    write_json_atomic(path, payload)


def validate_confirmation_start_lock(
    package: Path,
    validation_path: Path,
    start_lock_path: Path,
    runtime_manifest_path: Path,
    *,
    host_id: str | None = None,
    instance_id: str | None = None,
    gpu_uuid: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    start_lock = json.loads(start_lock_path.read_text(encoding="utf-8"))
    manifest_path = package / "PREPARED_INPUTS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or start_lock.get("status") != "LOCKED":
        raise ValueError("confirmation requires passing qualification validation and a locked start")
    if start_lock.get("qualification_validation_sha256") != sha256_file(validation_path):
        raise ValueError("qualification validation changed after confirmation start lock")
    results_path = Path(str(validation.get("qualification_results", "")))
    if not results_path.is_file() or sha256_file(results_path) != validation.get("qualification_results_sha256"):
        raise ValueError("qualification results changed after confirmation start lock")
    for field, path in (
        ("prepared_manifest_sha256", manifest_path),
        ("qualification_requests_sha256", package / "QUALIFICATION_REQUESTS.jsonl"),
        ("confirmation_requests_sha256", package / "CONFIRMATION_REQUESTS.jsonl"),
        ("input_panel_manifest_sha256", HERE.parent / "data/input_panel_manifest.json"),
        ("qualification_lock_sha256", HERE.parent / "data/confirmation_qualification_lock.jsonl"),
        ("runtime_manifest_sha256", runtime_manifest_path),
    ):
        if not path.is_file() or start_lock.get(field) != sha256_file(path):
            raise ValueError(f"confirmation start-lock binding changed: {field}")
    runtime = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads((HERE.parent / "configs/experiment_locks.json").read_text(encoding="utf-8"))["gemma_confirmation"]
    expected_vllm = frozen["runtime"].split("==", 1)[1]
    observed_vllm = runtime.get("vllm_version") or runtime.get("vllm")
    for container in (runtime.get("versions"), runtime.get("expected"), runtime.get("lock")):
        if observed_vllm is None and isinstance(container, dict):
            observed_vllm = container.get("vllm_version") or container.get("vllm")
    if str(observed_vllm) != expected_vllm:
        raise ValueError("runtime manifest does not attest the frozen vLLM version")
    if validation.get("prepared_manifest_sha256") != start_lock.get("prepared_manifest_sha256"):
        raise ValueError("qualification validation is bound to a different prepared package")
    if validation.get("qualification_results_sha256") != start_lock.get("qualification_results_sha256"):
        raise ValueError("qualification results differ from the passing validation")
    if manifest.get("status") != "PASS" or manifest.get("local_only") is not True:
        raise ValueError("prepared confirmation package is not a verified local-only package")
    if start_lock.get("model_id") != manifest.get("model_id") or start_lock.get("model_revision") != manifest.get("model_revision"):
        raise ValueError("confirmation start-lock model identity differs from the request package")
    for field, value in (("host_id", host_id), ("instance_id", instance_id), ("gpu_uuid", gpu_uuid)):
        if value is not None and start_lock.get(field) != value:
            raise ValueError(f"confirmation start lock is bound to a different {field}")
    return validation, start_lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("qualification", "confirmation"), required=True)
    parser.add_argument("--input-package", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--attempt-id")
    parser.add_argument("--instance-id")
    parser.add_argument("--host-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--qualification-validation", type=Path)
    parser.add_argument("--confirmation-start-lock", type=Path)
    parser.add_argument("--preflight-only", action="store_true", help="validate frozen requests and gates without sending any request")
    args = parser.parse_args()
    package = args.input_package.resolve()
    result_root = package / "runtime_outputs"
    args.results = args.results or result_root / f"{args.phase}_responses.jsonl"
    args.status = args.status or result_root / f"{args.phase}_status.json"
    args.checkpoint_manifest = args.checkpoint_manifest or result_root / f"{args.phase}_checkpoint.json"
    args.runtime_manifest = args.runtime_manifest or package / "runtime_manifest.json"
    args.attempt_id = args.attempt_id or "LOCAL_PREFLIGHT"
    args.instance_id = args.instance_id or "LOCAL_PREFLIGHT"
    args.host_id = args.host_id or "LOCAL_PREFLIGHT"
    if args.concurrency < 1 or args.concurrency > 8:
        raise SystemExit("--concurrency must be between 1 and 8")
    validation: dict[str, Any] | None = None
    start_lock: dict[str, Any] | None = None
    if args.phase == "confirmation" and not args.preflight_only:
        if args.qualification_validation is None or not args.qualification_validation.is_file():
            raise SystemExit("confirmation requires a qualification validation file")
        if args.confirmation_start_lock is None or not args.confirmation_start_lock.is_file():
            raise SystemExit("confirmation requires an immutable confirmation start lock")
        if not all((args.attempt_id, args.instance_id, args.host_id, args.gpu_uuid)):
            raise SystemExit("live confirmation requires attempt, instance, host, and GPU identities")
        try:
            validation, start_lock = validate_confirmation_start_lock(
                package,
                args.qualification_validation.resolve(),
                args.confirmation_start_lock.resolve(),
                args.runtime_manifest.resolve(),
                host_id=args.host_id,
                instance_id=args.instance_id,
                gpu_uuid=args.gpu_uuid,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"confirmation production is unauthorized: {exc}") from None
    elif args.phase == "confirmation" and (args.qualification_validation is not None or args.confirmation_start_lock is not None):
        if args.qualification_validation is None or args.confirmation_start_lock is None:
            raise SystemExit("preflight gate check needs both qualification validation and start lock")
        if args.qualification_validation.is_file() and args.confirmation_start_lock.is_file() and args.runtime_manifest.is_file():
            try:
                validation, start_lock = validate_confirmation_start_lock(
                    package,
                    args.qualification_validation.resolve(),
                    args.confirmation_start_lock.resolve(),
                    args.runtime_manifest.resolve(),
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                validation, start_lock = None, None
    requests = request_map(package / ("QUALIFICATION_REQUESTS.jsonl" if args.phase == "qualification" else "CONFIRMATION_REQUESTS.jsonl"))
    _, errors = validate_ledger_identity(args.results.resolve(), requests)
    if errors:
        raise SystemExit("existing ledger identity validation failed: " + "; ".join(errors[:5]))
    completed = {str(row["request_key"]) for row in load_jsonl(args.results.resolve())}
    pending = [request for key, request in requests.items() if key not in completed]
    for request in requests.values():
        check_request_contract(request)
    expected_rows = {"qualification": 180, "confirmation": 2160}[args.phase]
    if len(requests) != expected_rows:
        raise SystemExit(f"frozen {args.phase} request count mismatch: expected {expected_rows}, got {len(requests)}")
    prepared_manifest_path = package / "PREPARED_INPUTS_MANIFEST.json"
    if prepared_manifest_path.exists():
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
        if prepared_manifest.get("status") != "PASS" or prepared_manifest.get("local_only") is not True:
            raise SystemExit("prepared input manifest is not a verified local-only package")
        for rel, record in prepared_manifest.get("files", {}).items():
            candidate = package / rel
            if not candidate.is_file() or sha256_file(candidate) != record.get("sha256"):
                raise SystemExit(f"prepared request hash mismatch: {rel}")
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "phase": args.phase, "frozen_requests": len(requests), "completed_requests": len(completed), "production_authorized": bool(validation and start_lock), "network_requests": 0, "new_gpu_generations": 0}, sort_keys=True))
        return 0
    if not all((args.attempt_id, args.instance_id, args.host_id, args.gpu_uuid)):
        raise SystemExit("live inference requires --attempt-id, --instance-id, --host-id, and --gpu-uuid")
    if not args.runtime_manifest.exists():
        raise SystemExit("runtime manifest is missing")
    runtime_sha = sha256_file(args.runtime_manifest.resolve())
    if args.phase == "confirmation":
        if start_lock is None or start_lock.get("runtime_manifest_sha256") != runtime_sha:
            raise SystemExit("runtime manifest changed after confirmation start lock")
    started = now_utc()
    args.status.parent.mkdir(parents=True, exist_ok=True)
    write_status(args.status, args.phase, started=started, expected=len(requests), observed=len(completed), failed=0, tokens=0, errors=[])
    if not pending:
        write_checkpoint_manifest(
            phase=args.phase,
            ledger_path=args.results.resolve(),
            frozen_requests=requests,
            manifest_path=args.checkpoint_manifest.resolve(),
            attempt_id=args.attempt_id,
            instance_id=args.instance_id,
            host_id=args.host_id,
            gpu_uuid=args.gpu_uuid,
        )
        print(json.dumps({"status": "COMPLETE", "phase": args.phase, "rows": len(completed)}, sort_keys=True))
        return 0

    observed = len(completed)
    failed = 0
    tokens = 0
    errors_seen: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                one_request,
                request,
                base_url=args.base_url,
                timeout=args.timeout_seconds,
                retries=args.retries,
                attempt_id=args.attempt_id,
                instance_id=args.instance_id,
                host_id=args.host_id,
                gpu_uuid=args.gpu_uuid,
                runtime_manifest_sha256=runtime_sha,
            ): request
            for request in pending
        }
        for future in as_completed(futures):
            request = futures[future]
            try:
                row = future.result()
                append_row(args.results.resolve(), row, requests)
                observed += 1
                tokens += int(row.get("output_tokens") or 0)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors_seen.append(f"{request.get('request_key')}: {safe_error(exc)}")
            # The operational batch is eight concurrent requests.  Persist the
            # durable ledger and checkpoint after every such batch (and on the
            # final partial batch), so a disconnect never leaves a completed
            # batch without a checkpoint manifest.
            if observed % 8 == 0 or (observed + failed) == len(requests):
                write_status(
                    args.status,
                    args.phase,
                    started=started,
                    expected=len(requests),
                    observed=observed,
                    failed=failed,
                    tokens=tokens,
                    errors=errors_seen,
                )
                write_checkpoint_manifest(
                    phase=args.phase,
                    ledger_path=args.results.resolve(),
                    frozen_requests=requests,
                    manifest_path=args.checkpoint_manifest.resolve(),
                    attempt_id=args.attempt_id,
                    instance_id=args.instance_id,
                    host_id=args.host_id,
                    gpu_uuid=args.gpu_uuid,
                )
                print(json.dumps({"phase": args.phase, "completed": observed, "failed": failed}, sort_keys=True), flush=True)
    write_status(
        args.status,
        args.phase,
        started=started,
        expected=len(requests),
        observed=observed,
        failed=failed,
        tokens=tokens,
        errors=errors_seen,
    )
    write_checkpoint_manifest(
        phase=args.phase,
        ledger_path=args.results.resolve(),
        frozen_requests=requests,
        manifest_path=args.checkpoint_manifest.resolve(),
        attempt_id=args.attempt_id,
        instance_id=args.instance_id,
        host_id=args.host_id,
        gpu_uuid=args.gpu_uuid,
    )
    result = {"status": "PASS" if failed == 0 and observed == len(requests) else "INCOMPLETE", "phase": args.phase, "completed": observed, "failed": failed}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
