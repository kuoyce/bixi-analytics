"""
Step 01: materialize live station + canonical + champion context artifact.

Artifact output:
- models/inference/run_*/live_station/live_station.json
"""

from __future__ import annotations

import argparse
import json
import os

from inference_artifacts import (
    build_request_from_env,
    build_spark_session,
    build_default_run_id,
    build_default_run_ts,
    get_base_path,
    load_and_validate_champion_row_for_station,
    load_champion_station_ids,
    resolve_canonical_station_id,
    resolve_inference_run_context,
    to_serializable_dict,
    to_utc_hour,
    validate_canonical_station_against_champion_scope,
    write_step_artifact,
    build_live_station_snapshot,
)


def run_live_station_step(
    run_id: str,
    run_ts: str,
) -> tuple[dict, str]:
    base_path = get_base_path()

    request = build_request_from_env()
    request_ts = to_utc_hour(request.request_timestamp)

    live_station = build_live_station_snapshot(request)
    request.name = live_station.name

    resolver_result = resolve_canonical_station_id(base_path, live_station, request)
    canonical_station_id = str(resolver_result["canonical_station_id"])

    spark = build_spark_session()
    try:
        champion_station_ids = load_champion_station_ids(spark, base_path)
        validate_canonical_station_against_champion_scope(
            canonical_station_id=canonical_station_id,
            champion_station_ids=champion_station_ids,
        )
        champion_row = load_and_validate_champion_row_for_station(
            spark=spark,
            base_path=base_path,
            canonical_station_id=canonical_station_id,
        )
    finally:
        spark.stop()

    payload = {
        "run_id": run_id,
        "run_ts": run_ts,
        "request": {
            "station_id": request.station_id,
            "name": request.name,
            "lat": request.lat,
            "lon": request.lon,
        },
        "request_timestamp": request_ts.isoformat().replace("+00:00", "Z"),
        "live_station": to_serializable_dict(live_station),
        "canonical_station_id": canonical_station_id,
        "resolver_result": resolver_result,
        "champion_row": champion_row,
    }

    artifact_path = write_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="live_station",
        payload=payload,
        filename="live_station.json",
    )
    return payload, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Step 01: live station context")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-ts", default=None)
    parser.add_argument(
        "--request-json",
        default=None,
        help="Optional INFERENCE_REQUEST_JSON override",
    )
    args = parser.parse_args()

    if args.request_json and args.request_json.strip():
        os.environ["INFERENCE_REQUEST_JSON"] = args.request_json.strip()

    run_id, run_ts = resolve_inference_run_context(args.run_id, args.run_ts)

    payload, artifact_path = run_live_station_step(run_id=run_id, run_ts=run_ts)
    print(
        json.dumps(
            {
                "step": "live_station",
                "run_id": run_id,
                "run_ts": run_ts,
                "artifact_path": artifact_path,
                "station_id": payload["live_station"]["station_id"],
                "canonical_station_id": payload["canonical_station_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
