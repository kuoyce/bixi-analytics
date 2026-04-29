from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARK_PIPELINES_DIR = PROJECT_ROOT / "src" / "spark-pipelines"
if SPARK_PIPELINES_DIR.exists() and str(SPARK_PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(SPARK_PIPELINES_DIR))

from inference_artifacts import get_inference_run_dir
from sparkutils import resolve_data_path


def _default_request_timestamp() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    rounded = now.replace(minute=0, second=0, microsecond=0)
    return rounded.isoformat().replace("+00:00", "Z")


def _build_run_context(run_id: str | None, run_ts: str | None) -> tuple[str, str]:
    now = datetime.datetime.now(datetime.timezone.utc)
    if not run_id:
        run_id = f"run_{now.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if not run_ts:
        run_ts = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return run_id, run_ts


def _run_reference_pipeline(
    run_id: str,
    run_ts: str,
    request_json: str,
    script_path: Path,
) -> None:
    env = os.environ.copy()
    env["INFERENCE_REQUEST_JSON"] = request_json
    command = [
        sys.executable,
        str(script_path),
        "--run-id",
        run_id,
        "--run-ts",
        run_ts,
    ]
    subprocess.run(command, env=env, check=True)


def _read_reference_output(run_id: str, station_id: str) -> dict[str, Any]:
    base_path = resolve_data_path()
    run_dir = Path(get_inference_run_dir(base_path, run_id))
    output_path = run_dir / "output" / f"station_{station_id}.json"
    if not output_path.exists():
        raise FileNotFoundError(f"Reference output not found: {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _fetch_api_output(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/inference/run", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_values(
    reference: Any,
    candidate: Any,
    path: str,
    tolerance: float,
    ignore_keys: set[str],
    mismatches: list[str],
) -> None:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        keys = set(reference.keys()) | set(candidate.keys())
        for key in sorted(keys):
            if key in ignore_keys:
                continue
            if key not in reference:
                mismatches.append(f"{path}.{key}: missing in reference")
                continue
            if key not in candidate:
                mismatches.append(f"{path}.{key}: missing in api")
                continue
            _compare_values(
                reference[key],
                candidate[key],
                f"{path}.{key}",
                tolerance,
                ignore_keys,
                mismatches,
            )
        return

    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            mismatches.append(f"{path}: length {len(reference)} != {len(candidate)}")
            return
        for idx, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _compare_values(
                ref_item,
                cand_item,
                f"{path}[{idx}]",
                tolerance,
                ignore_keys,
                mismatches,
            )
        return

    if _is_number(reference) and _is_number(candidate):
        if not math.isclose(float(reference), float(candidate), rel_tol=tolerance, abs_tol=tolerance):
            mismatches.append(f"{path}: {reference} != {candidate}")
        return

    if reference != candidate:
        mismatches.append(f"{path}: {reference} != {candidate}")


def compare_outputs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float,
    compare_live_status: bool,
) -> list[str]:
    ignore_keys = set()
    if not compare_live_status:
        ignore_keys.update({"num_bikes_available", "num_docks_available"})

    mismatches: list[str] = []
    _compare_values(reference, candidate, "output", tolerance, ignore_keys, mismatches)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FastAPI inference output to run_10 pipeline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--name")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--request-timestamp")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--compare-live-status", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-ts")
    args = parser.parse_args()

    if (args.lat is None) != (args.lon is None):
        raise ValueError("--lat and --lon must be provided together")

    run_id, run_ts = _build_run_context(args.run_id, args.run_ts)

    request_timestamp = args.request_timestamp or _default_request_timestamp()
    request_payload: dict[str, Any] = {"station_id": args.station_id, "request_timestamp": request_timestamp}
    if args.name:
        request_payload["name"] = args.name
    if args.lat is not None:
        request_payload["lat"] = args.lat
    if args.lon is not None:
        request_payload["lon"] = args.lon

    request_json = json.dumps(request_payload, ensure_ascii=True)

    script_path = PROJECT_ROOT / "src" / "spark-pipelines" / "run_10_inference_pipeline.py"
    _run_reference_pipeline(run_id, run_ts, request_json, script_path)

    reference_output = _read_reference_output(run_id, args.station_id)
    api_output = _fetch_api_output(args.base_url, request_payload, args.timeout)

    mismatches = compare_outputs(
        reference=reference_output,
        candidate=api_output,
        tolerance=args.tolerance,
        compare_live_status=args.compare_live_status,
    )

    if mismatches:
        print("Outputs differ:")
        for item in mismatches[:20]:
            print(f"- {item}")
        if len(mismatches) > 20:
            print(f"... and {len(mismatches) - 20} more")
        raise SystemExit(1)

    print("Outputs are similar.")


if __name__ == "__main__":
    main()
