"""
Stage 08: evaluate all inflow/outflow model combinations (combi) for netflow.

This stage reads stage-07 solo summaries and model artifacts, scores all combinations on
station test data, and saves per-run combi results to models/summary/combi/{run_id}.
"""

import os

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from sparkutils import (
    append_run_df_to_delta_table,
    apply_local_spark_defaults,
    asymmetric_loss_mean,
    get_spark,
    is_production_mode,
    resolve_data_path,
    resolve_summary_table_target,
    should_write_summary_tables,
)

HARD_CODED_STATION_IDS: list[str] = ['STN_0001', 'STN_0002', 'STN_0003', 'STN_0004', 'STN_0005', 'STN_0006']

def build_storage_path(base_path: str, *parts: str) -> str:
    base = base_path.rstrip("/")
    suffix = "/".join(str(p).strip("/") for p in parts)
    return f"{base}/{suffix}" if suffix else base


def list_summary_run_paths(summary_root_path: str) -> list[str]:
    if not os.path.isdir(summary_root_path):
        return []

    run_paths = []
    for name in sorted(os.listdir(summary_root_path)):
        path = build_storage_path(summary_root_path, name)
        if os.path.isdir(path):
            run_paths.append(path)

    return run_paths


def load_gold_inputs(spark, base_path: str) -> tuple[DataFrame, DataFrame]:
    flow_df = spark.read.parquet(f"{base_path}/gold/station_flow")
    stations_df = spark.read.parquet(f"{base_path}/silver/station_cleaning/station_canonical_summary")
    return flow_df, stations_df


def parse_station_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return sorted({token.strip() for token in raw_value.split(",") if token.strip()})


def resolve_target_station_ids(stations_df: DataFrame) -> list[str]:
    env_station_ids = parse_station_ids(os.environ.get("STAGE8_STATION_IDS"))
    hardcoded_station_ids = sorted({sid.strip() for sid in HARD_CODED_STATION_IDS if sid and sid.strip()})

    if env_station_ids:
        requested_station_ids = env_station_ids
        source = "env:STAGE8_STATION_IDS"
    elif hardcoded_station_ids:
        requested_station_ids = hardcoded_station_ids
        source = "HARD_CODED_STATION_IDS"
    else:
        requested_station_ids = [
            row["canonical_station_id"]
            for row in stations_df.select("canonical_station_id").dropna().dropDuplicates().collect()
        ]
        source = "all canonical station ids"

    if not requested_station_ids:
        raise ValueError("No station ids resolved. Provide STAGE8_STATION_IDS or populate HARD_CODED_STATION_IDS.")

    existing_station_ids = {
        row["canonical_station_id"]
        for row in stations_df.select("canonical_station_id").dropna().dropDuplicates().collect()
    }
    missing_station_ids = sorted([sid for sid in requested_station_ids if sid not in existing_station_ids])
    if missing_station_ids:
        raise ValueError(f"Unknown station ids: {missing_station_ids}")

    print(f"Resolved {len(requested_station_ids)} target station(s) from {source}")
    return requested_station_ids


def get_columns() -> tuple[list[str], list[str], str]:
    all_columns = [
        "ts_hour", "station_inflow", "station_outflow", "station_netflow",
        "radius100m_inflow_lag1", "radius100m_outflow_lag1", "radius100m_inflow_lag12", "radius100m_outflow_lag12",
        "radius100m_inflow_rollmean6", "radius100m_outflow_rollmean6", "radius100m_inflow_rollmean12", "radius100m_outflow_rollmean12",
        "radius100m_inflow_rollsum6", "radius100m_outflow_rollsum6", "radius100m_inflow_rollsum12", "radius100m_outflow_rollsum12",
        "radius200m_inflow_lag1", "radius200m_outflow_lag1", "radius200m_inflow_lag12", "radius200m_outflow_lag12",
        "radius200m_inflow_rollmean6", "radius200m_outflow_rollmean6", "radius200m_inflow_rollmean12", "radius200m_outflow_rollmean12",
        "radius200m_inflow_rollsum6", "radius200m_outflow_rollsum6", "radius200m_inflow_rollsum12", "radius200m_outflow_rollsum12",
        "radius500m_inflow_lag1", "radius500m_outflow_lag1", "radius500m_inflow_lag12", "radius500m_outflow_lag12",
        "radius500m_inflow_rollmean6", "radius500m_outflow_rollmean6", "radius500m_inflow_rollmean12", "radius500m_outflow_rollmean12",
        "radius500m_inflow_rollsum6", "radius500m_outflow_rollsum6", "radius500m_inflow_rollsum12", "radius500m_outflow_rollsum12",
        "temp", "precip", "station_inflow_lag1", "station_outflow_lag1", "station_inflow_lag12", "station_outflow_lag12",
        "precip_rollmean3", "station_inflow_rollmean6", "station_outflow_rollmean6", "station_inflow_rollmean12", "station_outflow_rollmean12",
        "precip_rollsum3", "station_inflow_rollsum6", "station_outflow_rollsum6", "station_inflow_rollsum12", "station_outflow_rollsum12",
        "temp_bin", "dow", "is_weekday", "hod", "moy", "dow_cos", "hod_cos", "moy_cos",
    ]

    time_col = "ts_hour"
    base_exclude = [time_col, "station_inflow", "station_outflow", "station_netflow"]
    categorical_cols = ["temp_bin"]
    bool_cols = ["is_weekday"]
    numeric_cols = [c for c in all_columns if c not in base_exclude + categorical_cols + bool_cols]
    return numeric_cols, bool_cols, time_col


def fillna_numerics_and_booleans(df: DataFrame, num_cols: list[str], bool_cols: list[str]) -> DataFrame:
    fill_map = {c: 0.0 for c in num_cols}
    fill_map.update({c: 0 for c in bool_cols})
    return df.fillna(fill_map)


def split_train_test_by_cutoff(df: DataFrame, cutoff_date: str = "2025-08-01") -> tuple[DataFrame, DataFrame]:
    train_df = df.filter(F.col("ts_hour") < cutoff_date)
    test_df = df.filter(F.col("ts_hour") >= cutoff_date)
    return train_df, test_df


def load_solo_summary_history(spark, base_path: str) -> DataFrame:
    summary_root = build_storage_path(base_path, "models", "summary", "solo")
    run_paths = list_summary_run_paths(summary_root)
    if not run_paths:
        raise FileNotFoundError(
            f"No solo summary runs found under {summary_root}. "
            "Run stage 07 first to generate models/summary/solo/{run_id}."
        )
    return spark.read.parquet(*run_paths)


def resolve_current_run_id(summary_df: DataFrame) -> tuple[str, str]:
    env_run_id = os.environ.get("PIPELINE_RUN_ID") or os.environ.get("CHAMP_RUN_ID")
    if not env_run_id:
        job_run_id = os.environ.get("PIPELINE_JOB_RUN_ID")
        if job_run_id and job_run_id.strip():
            repair_count = os.environ.get("PIPELINE_REPAIR_COUNT", "0").strip() or "0"
            env_run_id = f"job_{job_run_id.strip()}_repair_{repair_count}"

    if env_run_id:
        row = (
            summary_df.where(F.col("run_id") == env_run_id)
            .orderBy(F.col("run_ts").desc_nulls_last())
            .select("run_id", "run_ts")
            .limit(1)
            .collect()
        )
        if not row:
            raise ValueError(f"Configured run_id={env_run_id} was not found in solo summary history")
        return row[0]["run_id"], row[0]["run_ts"]

    if is_production_mode():
        raise ValueError(
            "PIPELINE_RUN_ID is required in production mode for stage 08. "
            "Pass Databricks {{job.run_id}} plus {{job.repair_count}} as PIPELINE_RUN_ID, "
            "or provide PIPELINE_JOB_RUN_ID and PIPELINE_REPAIR_COUNT."
        )

    latest = (
        summary_df.orderBy(F.col("run_ts").desc_nulls_last(), F.col("run_id").desc_nulls_last())
        .select("run_id", "run_ts")
        .limit(1)
        .collect()
    )
    if not latest:
        raise ValueError("No run metadata found in solo summary history")

    return latest[0]["run_id"], latest[0]["run_ts"]


def get_available_models_per_station(
    summary_df: DataFrame,
    station_ids: list[str],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    target_df = summary_df.where(F.col("target_col").isin(["station_inflow", "station_outflow"]))
    target_df = target_df.where(F.col("station_id").isin(station_ids))

    model_rows = (
        target_df.select("station_id", "target_col", "model_name", "best_params_json")
        .dropna(subset=["station_id", "target_col", "model_name", "best_params_json"])
        .dropDuplicates(["station_id", "target_col", "model_name", "best_params_json"])
        .collect()
    )

    models_by_station: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in model_rows:
        sid = row["station_id"]
        tcol = row["target_col"]
        model_dt = {
            "model_name": row["model_name"],
            "best_params_json": row["best_params_json"],
        }
        models_by_station.setdefault(sid, {}).setdefault(tcol, []).append(model_dt)

    for sid in models_by_station:
        for tcol in models_by_station[sid]:
            models_by_station[sid][tcol] = sorted(
                models_by_station[sid][tcol],
                key=lambda x: (x["model_name"], x["best_params_json"]),
            )

    return models_by_station


def evaluate_station(
    run_id: str,
    run_ts: str,
    station_id: str,
    test_df: DataFrame,
    inflow_model_name: str,
    inflow_best_params_json: str,
    outflow_model_name: str,
    outflow_best_params_json: str,
    model_root_path: str,
) -> dict:
    inflow_model_path = build_storage_path(model_root_path, run_id, station_id, "station_inflow", inflow_model_name)
    outflow_model_path = build_storage_path(model_root_path, run_id, station_id, "station_outflow", outflow_model_name)

    inflow_model = PipelineModel.load(inflow_model_path)
    outflow_model = PipelineModel.load(outflow_model_path)

    inflow_pred = inflow_model.transform(test_df).select(
        "ts_hour",
        "station_netflow",
        F.col("prediction").alias("predicted_inflow"),
    )
    outflow_pred = outflow_model.transform(test_df).select(
        "ts_hour",
        F.col("prediction").alias("predicted_outflow"),
    )

    joined_pred = inflow_pred.join(outflow_pred, on=["ts_hour"], how="inner").withColumn(
        "model_netflow", F.col("predicted_inflow") - F.col("predicted_outflow")
    )

    asymmetric_mae_mean = asymmetric_loss_mean(
        joined_pred,
        y_true_col="station_netflow",
        y_pred_col="model_netflow",
        loss_type="mae",
    )
    asymmetric_rmse_mean = asymmetric_loss_mean(
        joined_pred,
        y_true_col="station_netflow",
        y_pred_col="model_netflow",
        loss_type="rmse",
    )

    return {
        "run_id": run_id,
        "run_ts": run_ts,
        "station_id": station_id,
        "target_col": "station_netflow",
        "inflow_model_name": inflow_model_name,
        "inflow_best_params_json": inflow_best_params_json,
        "outflow_model_name": outflow_model_name,
        "outflow_best_params_json": outflow_best_params_json,
        "param_signature": f"in:{inflow_best_params_json}|out:{outflow_best_params_json}",
        "test_row_count": joined_pred.count(),
        "asymmetric_mae_mean": float(asymmetric_mae_mean) if asymmetric_mae_mean is not None else None,
        "asymmetric_rmse_mean": float(asymmetric_rmse_mean) if asymmetric_rmse_mean is not None else None,
        "inflow_model_path": inflow_model_path,
        "outflow_model_path": outflow_model_path,
    }


def main() -> None:
    spark = get_spark()
    apply_local_spark_defaults(spark)

    base_path = resolve_data_path()
    model_root_path = build_storage_path(base_path, "models")
    flow_df, stations_df = load_gold_inputs(spark, base_path)
    station_ids = resolve_target_station_ids(stations_df)

    numeric_cols, bool_cols, _time_col = get_columns()
    summary_df = load_solo_summary_history(spark, base_path)
    required = {"run_id", "run_ts", "station_id", "target_col", "model_name", "best_params_json"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(
            f"Solo summary is missing required columns {sorted(missing)}. "
            "Run stage 07 again to regenerate summary with run metadata."
        )

    run_id, run_ts = resolve_current_run_id(summary_df)
    current_run_summary_df = summary_df.where(F.col("run_id") == run_id)
    models_by_station = get_available_models_per_station(current_run_summary_df, station_ids)

    print("=== Stage 08 Combi Eval start: inflow/outflow model combinations ===")
    print(f"Base path: {base_path}")
    print(f"Solo summary path: {build_storage_path(base_path, 'models', 'summary', 'solo')}")
    print(f"Model root path: {model_root_path}")
    print(f"Current run id: {run_id}")
    print(f"Current run ts: {run_ts}")
    print(f"Stations to evaluate: {len(station_ids)}")

    results = []
    for station_id in station_ids:
        station_models = models_by_station.get(station_id, {})
        inflow_models = station_models.get("station_inflow", [])
        outflow_models = station_models.get("station_outflow", [])

        if not inflow_models or not outflow_models:
            print(f"Skipping {station_id}: missing inflow/outflow models in summary")
            continue

        print(
            f"{station_id}: evaluating {len(inflow_models)} inflow x {len(outflow_models)} outflow "
            f"= {len(inflow_models) * len(outflow_models)} combinations"
        )

        station_flow_df = flow_df.filter(F.col("station_id") == station_id)
        station_df = fillna_numerics_and_booleans(station_flow_df, numeric_cols, bool_cols)
        _train_df, test_df = split_train_test_by_cutoff(station_df)

        if test_df.limit(1).count() == 0:
            print(f"Skipping {station_id}: no test rows after cutoff")
            continue

        for inflow_model in inflow_models:
            for outflow_model in outflow_models:
                results.append(
                    evaluate_station(
                        run_id=run_id,
                        run_ts=run_ts,
                        station_id=station_id,
                        test_df=test_df,
                        inflow_model_name=inflow_model["model_name"],
                        inflow_best_params_json=inflow_model["best_params_json"],
                        outflow_model_name=outflow_model["model_name"],
                        outflow_best_params_json=outflow_model["best_params_json"],
                        model_root_path=model_root_path,
                    )
                )

    if not results:
        print("No combi results produced.")
        spark.stop()
        return

    result_df = spark.createDataFrame(results).orderBy(
        F.col("station_id").asc(),
        F.col("asymmetric_rmse_mean").desc_nulls_last(),
    )
    result_df = result_df.cache()
    result_df.count()

    combi_summary_path = build_storage_path(base_path, "models", "summary", "combi", run_id)
    result_df.write.mode("overwrite").parquet(combi_summary_path)

    combi_table_name = None
    if should_write_summary_tables():
        catalog, schema, table, _ = resolve_summary_table_target(
            default_table="combi",
            env_key="PIPELINE_TABLE_COMBI",
        )
        combi_table_name = append_run_df_to_delta_table(
            spark=spark,
            df=result_df,
            catalog=catalog,
            schema=schema,
            table=table,
            run_id=run_id,
            run_id_col="run_id",
        )

    print("=== Stage 08 Combi Eval results (preview) ===")
    print(f"Saved combi summary path: {combi_summary_path}")
    if combi_table_name:
        print(f"Appended combi summary table: {combi_table_name}")
    result_df.show(200, truncate=False)
    result_df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
