"""
Step 03: materialize synthetic station history artifact.

Artifact output:
- models/inference/run_*/synthetic_history/synthetic_history.json
"""

from __future__ import annotations

import argparse
import json

from inference_artifacts import (
    build_spark_session,
    get_base_path,
    parse_positive_int_env,
    parse_request_timestamp,
    parse_synthesis_mode_env,
    read_step_artifact,
    resolve_inference_run_context,
    synthesize_station_history,
    write_step_artifact,
)


def run_history_step(
    run_id: str,
    run_ts: str,
    lookback_hours: int,
    warmup_hours: int,
    synthesis_mode: str,
) -> tuple[dict, str]:
    base_path = get_base_path()

    live_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="live_station",
        filename="live_station.json",
    )

    request_ts = parse_request_timestamp(live_payload["request_timestamp"])
    canonical_station_id = str(live_payload["canonical_station_id"])

    spark = build_spark_session()
    try:
        history_result = synthesize_station_history(
            spark=spark,
            base_path=base_path,
            canonical_station_id=canonical_station_id,
            request_ts_utc=request_ts,
            lookback_hours=lookback_hours,
            requested_mode=synthesis_mode,
            warmup_hours=warmup_hours,
        )
    finally:
        spark.stop()

    payload = {
        "run_id": run_id,
        "run_ts": run_ts,
        "request_timestamp": request_ts.isoformat().replace("+00:00", "Z"),
        "station_id": live_payload["live_station"]["station_id"],
        "canonical_station_id": canonical_station_id,
        "mode_requested": history_result.mode_requested,
        "mode_used": history_result.mode_used,
        "lookback_hours": history_result.lookback_hours,
        "window_start": history_result.window_start,
        "window_end": history_result.window_end,
        "missing_before_fill": history_result.missing_before_fill,
        "synthesized_count": history_result.synthesized_count,
        "rows": history_result.rows,
    }

    artifact_path = write_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="synthetic_history",
        payload=payload,
        filename="synthetic_history.json",
    )
    return payload, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Step 03: synthetic history")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-ts", default=None)
    parser.add_argument("--lookback-hours", type=int, default=None)
    parser.add_argument("--warmup-hours", type=int, default=None)
    parser.add_argument("--synthesis-mode", default=None)
    args = parser.parse_args()

    run_id, run_ts = resolve_inference_run_context(args.run_id, args.run_ts)
    lookback_hours = args.lookback_hours or parse_positive_int_env(
        "INFERENCE_HISTORY_LOOKBACK_HOURS",
        default=168,
    )
    warmup_hours = args.warmup_hours or parse_positive_int_env(
        "INFERENCE_HISTORY_WARMUP_HOURS",
        default=336,
    )
    synthesis_mode = args.synthesis_mode or parse_synthesis_mode_env(
        "INFERENCE_SYNTHESIS_MODE",
        default="auto",
    )

    payload, artifact_path = run_history_step(
        run_id=run_id,
        run_ts=run_ts,
        lookback_hours=lookback_hours,
        warmup_hours=warmup_hours,
        synthesis_mode=synthesis_mode,
    )

    print(
        json.dumps(
            {
                "step": "synthetic_history",
                "run_id": run_id,
                "run_ts": run_ts,
                "artifact_path": artifact_path,
                "row_count": len(payload["rows"]),
                "mode_used": payload["mode_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
