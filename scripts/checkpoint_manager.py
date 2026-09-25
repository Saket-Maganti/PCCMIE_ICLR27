#!/usr/bin/env python3
"""Atomic, identity-checked append/checkpoint helpers for the Gemma lane.

This module deliberately knows nothing about strict or semantic outcomes.  It
only protects the frozen request keyspace and durable response rows.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


IDENTITY_FIELDS = (
    "request_key",
    "phase",
    "execution_class",
    "task_id",
    "reference_id",
    "treatment_id",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "prompt_hash",
    "rendered_chat_sha256",
    "request_sha256",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def request_map(request_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(request_path):
        key = str(row.get("request_key", ""))
        if not key or key in result:
            raise ValueError(f"duplicate or empty frozen request key in {request_path}: {key!r}")
        result[key] = row
    return result


def validate_ledger_identity(
    ledger_path: Path,
    frozen_requests: dict[str, dict[str, Any]],
    *,
    allow_missing: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate durable rows already present, without computing outcomes."""

    rows = load_jsonl(ledger_path)
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("request_key", ""))
        if not key:
            errors.append("ledger row has empty request_key")
            continue
        if key in seen:
            errors.append(f"duplicate ledger key: {key}")
        seen.add(key)
        request = frozen_requests.get(key)
        if request is None:
            errors.append(f"ledger key is not frozen: {key}")
            continue
        for field in IDENTITY_FIELDS:
            if row.get(field) != request.get(field):
                errors.append(f"{key}: {field} mismatch")
        if row.get("raw_response") is not None and not isinstance(row.get("raw_response"), str):
            errors.append(f"{key}: raw_response must be text")
    if not allow_missing and set(seen) != set(frozen_requests):
        errors.append(f"ledger keyspace incomplete: {len(seen)}/{len(frozen_requests)}")
    return rows, errors


def append_row(path: Path, row: dict[str, Any], frozen_requests: dict[str, dict[str, Any]]) -> None:
    """Append exactly one new row while holding a process-safe file lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            existing = load_jsonl(path)
            _, errors = validate_ledger_identity(path, frozen_requests)
            if errors:
                raise ValueError("existing ledger failed identity validation: " + "; ".join(errors[:5]))
            key = str(row.get("request_key", ""))
            if key not in frozen_requests:
                raise ValueError(f"row key is not frozen: {key}")
            if any(str(item.get("request_key")) == key for item in existing):
                raise ValueError(f"row key already exists: {key}")
            request = frozen_requests[key]
            for field in IDENTITY_FIELDS:
                if row.get(field) != request.get(field):
                    raise ValueError(f"{key}: {field} does not match frozen request")
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_checkpoint_manifest(
    *,
    phase: str,
    ledger_path: Path,
    frozen_requests: dict[str, dict[str, Any]],
    manifest_path: Path,
    attempt_id: str,
    instance_id: str,
    host_id: str,
    gpu_uuid: str,
) -> dict[str, Any]:
    rows, errors = validate_ledger_identity(ledger_path, frozen_requests)
    keys = {str(row.get("request_key")) for row in rows}
    payload = {
        "schema_version": "GEMMA_RTX5090_CHECKPOINT_MANIFEST_V1",
        "phase": phase,
        "attempt_id": attempt_id,
        "instance_id": str(instance_id),
        "host_id": str(host_id),
        "gpu_uuid": gpu_uuid,
        "ledger_path": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.exists() else None,
        "observed_rows": len(rows),
        "expected_rows": len(frozen_requests),
        "missing_rows": len(set(frozen_requests) - keys),
        "extra_rows": len(keys - set(frozen_requests)),
        "identity_validation_errors": errors,
        "complete_keyspace": not errors and keys == set(frozen_requests),
        "updated_at_utc": now_utc(),
        "scientific_outcomes_written": False,
    }
    write_json_atomic(manifest_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("qualification", "confirmation"), required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    requests = request_map(args.requests.resolve())
    result = write_checkpoint_manifest(
        phase=args.phase,
        ledger_path=args.ledger.resolve(),
        frozen_requests=requests,
        manifest_path=args.manifest.resolve(),
        attempt_id=args.attempt_id,
        instance_id=args.instance_id,
        host_id=args.host_id,
        gpu_uuid=args.gpu_uuid,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["identity_validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
