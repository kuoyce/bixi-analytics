"""
Step 05: build final inference output from step artifacts.

Reads:
- live_station/live_station.json
- weather/weather.json
- synthetic_history/synthetic_history.json
- feature_rows/feature_rows.json

Writes:
- models/inference/run_*/output/station_<station_id>.json

Note:
- Prediction arrays are produced by champion model scoring (Milestone 6).
- Historical arrays are sourced from synthetic_history and controlled by HISTORICAL_RETURN_STEPS.
"""

from __future__ import annotations

import argparse
import json

from inference_artifacts import (
    build_spark_session,
    get_base_path,
    parse_positive_int_env,
    read_step_artifact,
    resolve_inference_run_context,
    score_champion_models_from_feature_rows,
    write_step_artifact,
)


def _to_nonnegative_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def _select_history_tail_rows(
    synthetic_history_payload: dict,
    historical_return_steps: int,
) -> list[dict]:
    rows = synthetic_history_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("synthetic_history artifact has no rows")

    history_rows = [row for row in rows if isinstance(row, dict) and row.get("timestamp")]
    if not history_rows:
        raise ValueError("synthetic_history artifact rows are missing timestamp values")

    history_rows.sort(key=lambda row: str(row["timestamp"]))
    return history_rows[-historical_return_steps:]


def run_output_step(
    run_id: str,
    run_ts: str,
    historical_return_steps: int,
) -> tuple[dict, str]:
    base_path = get_base_path()

    live_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="live_station",
        filename="live_station.json",
    )
    # The final step intentionally reads all prior step artifacts by run_id.
    read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="weather",
        filename="weather.json",
    )
    synthetic_history_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="synthetic_history",
        filename="synthetic_history.json",
    )
    feature_rows_payload = read_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="feature_rows",
        filename="feature_rows.json",
    )

    if feature_rows_payload.get("missing_gold_columns"):
        raise ValueError(
            "Milestone 6 blocked: feature rows are missing Stage-06 gold columns: "
            f"{feature_rows_payload['missing_gold_columns']}"
        )
    if feature_rows_payload.get("missing_model_input_columns"):
        raise ValueError(
            "Milestone 6 blocked: feature rows are missing model input columns: "
            f"{feature_rows_payload['missing_model_input_columns']}"
        )

    champion_row = live_payload["champion_row"]
    model_input_path = str(feature_rows_payload["model_input_path"])

    spark = build_spark_session()
    try:
        prediction_result = score_champion_models_from_feature_rows(
            spark=spark,
            model_input_path=model_input_path,
            inflow_model_path=str(champion_row["inflow_model_path"]),
            outflow_model_path=str(champion_row["outflow_model_path"]),
        )
    finally:
        spark.stop()

    expected_row_count = int(feature_rows_payload.get("row_count") or 0)
    if expected_row_count and prediction_result["row_count"] != expected_row_count:
        raise ValueError(
            "Milestone 6 scoring row count mismatch: "
            f"predictions={prediction_result['row_count']} expected={expected_row_count}"
        )

    history_tail_rows = _select_history_tail_rows(
        synthetic_history_payload=synthetic_history_payload,
        historical_return_steps=historical_return_steps,
    )
    station_timestamps = [str(row["timestamp"]) for row in history_tail_rows]
    station_inflow = [_to_nonnegative_int(row.get("station_inflow")) for row in history_tail_rows]
    station_outflow = [_to_nonnegative_int(row.get("station_outflow")) for row in history_tail_rows]

    model_timestamps = [str(ts) for ts in prediction_result.get("timestamps", [])]
    model_inflow_int = [_to_nonnegative_int(value) for value in prediction_result["model_inflow"]]
    model_outflow_int = [_to_nonnegative_int(value) for value in prediction_result["model_outflow"]]

    if len(model_timestamps) != len(model_inflow_int) or len(model_timestamps) != len(model_outflow_int):
        raise ValueError(
            "Milestone 6 scoring timestamp/value length mismatch: "
            f"timestamps={len(model_timestamps)} inflow={len(model_inflow_int)} "
            f"outflow={len(model_outflow_int)}"
        )

    joined_timestamp = station_timestamps + model_timestamps
    joined_inflow = station_inflow + model_inflow_int
    joined_outflow = station_outflow + model_outflow_int

    live_station = live_payload["live_station"]
    station_id = str(live_station["station_id"])

    output_payload = {
        "request_timestamp": live_payload["request_timestamp"],
        "station_id": station_id,
        "capacity": live_station["capacity"],
        "num_bikes_available": live_station["num_bikes_available"],
        "num_docks_available": live_station["num_docks_available"],
        "canonical_station_id": live_payload["canonical_station_id"],
        "station_inflow": station_inflow,
        "station_outflow": station_outflow,
        "model_inflow": prediction_result["model_inflow"],
        "model_outflow": prediction_result["model_outflow"],
        "joined_timestamp": joined_timestamp,
        "joined_inflow": joined_inflow,
        "joined_outflow": joined_outflow,
    }

    artifact_path = write_step_artifact(
        base_path=base_path,
        run_id=run_id,
        step_name="output",
        payload=output_payload,
        filename=f"station_{station_id}.json",
    )
    return output_payload, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Step 05: final inference output")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-ts", default=None)
    parser.add_argument(
        "--historical-return-steps",
        type=int,
        default=None,
        help="Historical rows to include from synthetic_history. Defaults to HISTORICAL_RETURN_STEPS or 6.",
    )
    args = parser.parse_args()

    if args.historical_return_steps is not None and args.historical_return_steps <= 0:
        raise ValueError("--historical-return-steps must be > 0")

    historical_return_steps = (
        args.historical_return_steps
        if args.historical_return_steps is not None
        else parse_positive_int_env("HISTORICAL_RETURN_STEPS", default=6)
    )

    run_id, run_ts = resolve_inference_run_context(args.run_id, args.run_ts)
    output_payload, artifact_path = run_output_step(
        run_id=run_id,
        run_ts=run_ts,
        historical_return_steps=historical_return_steps,
    )

    print(
        json.dumps(
            {
                "step": "output",
                "run_id": run_id,
                "run_ts": run_ts,
                "historical_return_steps": historical_return_steps,
                "artifact_path": artifact_path,
                "output": output_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
