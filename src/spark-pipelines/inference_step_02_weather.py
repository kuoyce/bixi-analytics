"""
Step 02: materialize weather horizon artifact.

Artifact output:
- models/inference/run_*/weather/weather.json
"""

from __future__ import annotations

import argparse
import json

from inference_artifacts import (
    fetch_forecast_weather_rows,
    get_base_path,
    parse_positive_int_env,
    parse_request_timestamp,
    read_step_artifact,
    resolve_inference_run_context,
    write_step_artifact,
)


def run_weather_step(
    run_id: str,
    run_ts: str,
    horizon_steps: int,
) -> tuple[dict, str]:
    base_path = get_base_path()

    live_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="live_station",
        filename="live_station.json",
    )

    request_ts = parse_request_timestamp(live_payload["request_timestamp"])
    live_station = live_payload["live_station"]

    weather_horizon = fetch_forecast_weather_rows(
        lat=float(live_station["lat"]),
        lon=float(live_station["lon"]),
        request_ts_utc=request_ts,
        horizon_steps=horizon_steps,
    )

    payload = {
        "run_id": run_id,
        "run_ts": run_ts,
        "request_timestamp": request_ts.isoformat().replace("+00:00", "Z"),
        "station_id": live_station["station_id"],
        "canonical_station_id": live_payload["canonical_station_id"],
        "horizon_steps": horizon_steps,
        "rows": weather_horizon.rows,
        "fallback_applied": weather_horizon.fallback_applied,
        "missing_timestamps": weather_horizon.missing_timestamps,
    }

    artifact_path = write_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="weather",
        payload=payload,
        filename="weather.json",
    )
    return payload, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Step 02: weather horizon")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-ts", default=None)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Optional horizon override. Defaults to INFERENCE_HORIZON_STEPS or 6.",
    )
    args = parser.parse_args()

    run_id, run_ts = resolve_inference_run_context(args.run_id, args.run_ts)
    horizon_steps = args.horizon_steps or parse_positive_int_env(
        "INFERENCE_HORIZON_STEPS",
        default=6,
    )

    payload, artifact_path = run_weather_step(
        run_id=run_id,
        run_ts=run_ts,
        horizon_steps=horizon_steps,
    )

    print(
        json.dumps(
            {
                "step": "weather",
                "run_id": run_id,
                "run_ts": run_ts,
                "artifact_path": artifact_path,
                "row_count": len(payload["rows"]),
                "fallback_applied": payload["fallback_applied"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
