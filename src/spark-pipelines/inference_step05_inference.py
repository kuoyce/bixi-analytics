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
"""

from __future__ import annotations

import argparse
import json

from inference_artifacts import (
    build_spark_session,
    get_base_path,
    read_step_artifact,
    resolve_inference_run_context,
    score_champion_models_from_feature_rows,
    write_step_artifact,
)


def run_output_step(run_id: str, run_ts: str) -> tuple[dict, str]:
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
    read_step_artifact(
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

    live_station = live_payload["live_station"]
    station_id = str(live_station["station_id"])

    output_payload = {
        "request_timestamp": live_payload["request_timestamp"],
        "station_id": station_id,
        "capacity": live_station["capacity"],
        "num_bikes_available": live_station["num_bikes_available"],
        "num_docks_available": live_station["num_docks_available"],
        "canonical_station_id": live_payload["canonical_station_id"],
        "model_inflow": prediction_result["model_inflow"],
        "model_outflow": prediction_result["model_outflow"],
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
    args = parser.parse_args()

    run_id, run_ts = resolve_inference_run_context(args.run_id, args.run_ts)
    output_payload, artifact_path = run_output_step(run_id=run_id, run_ts=run_ts)

    print(
        json.dumps(
            {
                "step": "output",
                "run_id": run_id,
                "run_ts": run_ts,
                "artifact_path": artifact_path,
                "output": output_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
