"""
Stage 11: monitor station-level covariate and concept drift from gold test data.

Drift design:
- Covariate drift (per station): PSI over hod_cos weighted by station_inflow/outflow.
- Concept drift (per station): champion netflow model asymmetric loss ratio
  between recent and reference windows within the test set.

Data source:
- gold/station_flow (same feature source used by stages 07/08)
- models/summary/champion/current (champion model paths from stage 09)
"""

import argparse
import datetime
import math
import os

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from sparkutils import (
    append_run_df_to_delta_table,
    apply_local_spark_defaults,
    asymmetric_loss_mean,
    get_spark,
    resolve_data_path,
    resolve_pipeline_run_metadata,
    resolve_summary_table_target,
    should_write_summary_tables,
)

HARD_CODED_STATION_IDS: list[str] = ["STN_0001", "STN_0002", "STN_0003", "STN_0004",]
DEFAULT_CUTOFF_DATE = "2025-08-01"
DEFAULT_REFERENCE_WINDOW_DAYS = 28
DEFAULT_RECENT_WINDOW_DAYS = 7
DEFAULT_HOD_COS_BIN_COUNT = 20
DEFAULT_THRESHOLD_COVARIATE_DRIFT_SCORE = 0.25
DEFAULT_THRESHOLD_CONCEPT_RMSE_RATIO = 1.2
DEFAULT_THRESHOLD_CONCEPT_MAE_RATIO = 1.05
PSI_EPSILON = 1e-9


def _set_env_if_provided(env_key: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[env_key] = str(value)


def build_storage_path(base_path: str, *parts: str) -> str:
    base = base_path.rstrip("/")
    suffix = "/".join(str(p).strip("/") for p in parts)
    return f"{base}/{suffix}" if suffix else base


def resolve_cutoff_date(raw_value: str | None) -> str:
    cutoff_date = (raw_value or "").strip()
    return cutoff_date or DEFAULT_CUTOFF_DATE


def parse_station_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return sorted({token.strip() for token in raw_value.split(",") if token.strip()})


def resolve_positive_int(raw_value: str | None, default_value: int, field_name: str) -> int:
    value = (raw_value or "").strip()
    if not value:
        return default_value

    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive. Got {resolved}")
    return resolved


def resolve_positive_float(raw_value: str | None, default_value: float, field_name: str) -> float:
    value = (raw_value or "").strip()
    if not value:
        return default_value

    resolved = float(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive. Got {resolved}")
    return resolved


def resolve_target_station_ids(stations_df: DataFrame) -> list[str]:
    env_station_ids = parse_station_ids(os.environ.get("PIPELINE_STATION_ID"))
    hardcoded_station_ids = sorted({sid.strip() for sid in HARD_CODED_STATION_IDS if sid and sid.strip()})

    if env_station_ids:
        requested_station_ids = env_station_ids
        source = "env:PIPELINE_STATION_ID"
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
        raise ValueError("No station ids resolved. Provide PIPELINE_STATION_ID or populate HARD_CODED_STATION_IDS.")

    existing_station_ids = {
        row["canonical_station_id"]
        for row in stations_df.select("canonical_station_id").dropna().dropDuplicates().collect()
    }
    missing_station_ids = sorted([sid for sid in requested_station_ids if sid not in existing_station_ids])
    if missing_station_ids:
        raise ValueError(f"Unknown station ids: {missing_station_ids}")

    print(f"Resolved {len(requested_station_ids)} target station(s) from {source}")
    return requested_station_ids


def load_gold_inputs(spark, base_path: str) -> tuple[DataFrame, DataFrame]:
    flow_df = spark.read.parquet(f"{base_path}/gold/station_flow")
    stations_df = spark.read.parquet(f"{base_path}/silver/station_cleaning/station_canonical_summary")
    return flow_df, stations_df


def load_current_champion(spark, base_path: str) -> DataFrame:
    champion_path = build_storage_path(base_path, "models", "summary", "champion", "current")
    champion_df = spark.read.parquet(champion_path)

    required = {
        "station_id",
        "run_id",
        "run_ts",
        "inflow_model_path",
        "outflow_model_path",
    }
    missing = required - set(champion_df.columns)
    if missing:
        raise ValueError(
            f"Champion snapshot is missing required columns {sorted(missing)} at {champion_path}. "
            "Run stage 09 first."
        )

    return champion_df


def fill_missing_feature_values(df: DataFrame) -> DataFrame:
    numeric_types = (
        T.ByteType,
        T.ShortType,
        T.IntegerType,
        T.LongType,
        T.FloatType,
        T.DoubleType,
        T.DecimalType,
    )

    fill_map: dict[str, float | bool] = {}
    for field in df.schema.fields:
        if field.name in {"station_id", "ts_hour", "temp_bin"}:
            continue
        if isinstance(field.dataType, numeric_types):
            fill_map[field.name] = 0.0
        elif isinstance(field.dataType, T.BooleanType):
            fill_map[field.name] = False

    if not fill_map:
        return df
    return df.fillna(fill_map)


def split_train_test_by_cutoff(df: DataFrame, cutoff_date: str = DEFAULT_CUTOFF_DATE) -> tuple[DataFrame, DataFrame]:
    train_df = df.filter(F.col("ts_hour") < cutoff_date)
    test_df = df.filter(F.col("ts_hour") >= cutoff_date)
    return train_df, test_df


def split_reference_recent_from_test(
    test_df: DataFrame,
    reference_window_days: int,
    recent_window_days: int,
    timestamp_col: str = "ts_hour",
) -> tuple[DataFrame, DataFrame, datetime.datetime | None, datetime.datetime | None, datetime.datetime | None]:
    max_ts = test_df.select(F.max(F.col(timestamp_col)).alias("max_ts")).collect()[0]["max_ts"]
    if max_ts is None:
        return test_df.limit(0), test_df.limit(0), None, None, None

    recent_start = max_ts - datetime.timedelta(days=recent_window_days)
    reference_start = recent_start - datetime.timedelta(days=reference_window_days)

    reference_df = test_df.filter(
        (F.col(timestamp_col) >= F.lit(reference_start))
        & (F.col(timestamp_col) < F.lit(recent_start))
    )
    recent_df = test_df.filter(F.col(timestamp_col) >= F.lit(recent_start))

    return reference_df, recent_df, reference_start, recent_start, max_ts


def build_hod_cos_bin_expr(bin_count: int):
    hod_cos = F.col("hod_cos").cast("double")
    clipped = F.greatest(F.lit(-1.0), F.least(hod_cos, F.lit(1.0)))
    raw_bin = ((clipped + F.lit(1.0)) / F.lit(2.0)) * F.lit(float(bin_count))
    max_bin = F.lit(int(bin_count - 1))

    return (
        F.when(hod_cos.isNull(), None)
        .otherwise(F.least(max_bin, F.floor(raw_bin).cast("int")))
        .alias("bin_id")
    )


def collect_weight_by_bin(df: DataFrame, target_col: str, bin_col: str = "bin_id") -> dict[int, float]:
    rows = (
        df.groupBy(bin_col)
        .agg(F.sum(F.col(target_col).cast("double")).alias("weight"))
        .collect()
    )

    out: dict[int, float] = {}
    for row in rows:
        bin_id = row[bin_col]
        if bin_id is None:
            continue
        out[int(bin_id)] = float(row["weight"] or 0.0)

    return out


def compute_psi(reference_bins: dict[int, float], recent_bins: dict[int, float]) -> float | None:
    all_bins = sorted(set(reference_bins) | set(recent_bins))
    if not all_bins:
        return None

    ref_total = sum(max(reference_bins.get(bin_id, 0.0), 0.0) for bin_id in all_bins)
    recent_total = sum(max(recent_bins.get(bin_id, 0.0), 0.0) for bin_id in all_bins)
    if ref_total <= 0 or recent_total <= 0:
        return None

    ref_denom = ref_total + PSI_EPSILON * len(all_bins)
    recent_denom = recent_total + PSI_EPSILON * len(all_bins)

    psi = 0.0
    for bin_id in all_bins:
        ref_share = (max(reference_bins.get(bin_id, 0.0), 0.0) + PSI_EPSILON) / ref_denom
        recent_share = (max(recent_bins.get(bin_id, 0.0), 0.0) + PSI_EPSILON) / recent_denom
        psi += (recent_share - ref_share) * math.log(recent_share / ref_share)

    return float(psi)


def compute_covariate_drift_metrics(
    reference_df: DataFrame,
    recent_df: DataFrame,
    hod_cos_bin_count: int,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "covariate_hod_cos_psi_inflow": None,
        "covariate_hod_cos_psi_outflow": None,
        "covariate_drift_score": None,
    }

    for target_col in ["station_inflow", "station_outflow"]:
        ref_hod_cos = collect_weight_by_bin(
            reference_df.select(build_hod_cos_bin_expr(hod_cos_bin_count), F.col(target_col)),
            target_col=target_col,
        )
        recent_hod_cos = collect_weight_by_bin(
            recent_df.select(build_hod_cos_bin_expr(hod_cos_bin_count), F.col(target_col)),
            target_col=target_col,
        )
        hod_cos_psi = compute_psi(ref_hod_cos, recent_hod_cos)

        if target_col == "station_inflow":
            metrics["covariate_hod_cos_psi_inflow"] = hod_cos_psi
        else:
            metrics["covariate_hod_cos_psi_outflow"] = hod_cos_psi

    psi_values = [
        metrics["covariate_hod_cos_psi_inflow"],
        metrics["covariate_hod_cos_psi_outflow"],
    ]
    psi_values = [value for value in psi_values if value is not None]
    if psi_values:
        metrics["covariate_drift_score"] = float(max(psi_values))

    return metrics


def score_netflow_with_champion(
    station_df: DataFrame,
    inflow_model_path: str,
    outflow_model_path: str,
) -> DataFrame:
    inflow_model = PipelineModel.load(inflow_model_path)
    outflow_model = PipelineModel.load(outflow_model_path)

    inflow_pred = inflow_model.transform(station_df).select(
        "ts_hour",
        "station_netflow",
        F.col("prediction").alias("predicted_inflow"),
    )
    outflow_pred = outflow_model.transform(station_df).select(
        "ts_hour",
        F.col("prediction").alias("predicted_outflow"),
    )

    return inflow_pred.join(outflow_pred, on=["ts_hour"], how="inner").withColumn(
        "model_netflow",
        F.col("predicted_inflow") - F.col("predicted_outflow"),
    )


def compute_concept_drift_metrics(
    reference_df: DataFrame,
    recent_df: DataFrame,
    inflow_model_path: str,
    outflow_model_path: str,
) -> dict[str, float | int | None]:
    reference_pred_df = score_netflow_with_champion(
        station_df=reference_df,
        inflow_model_path=inflow_model_path,
        outflow_model_path=outflow_model_path,
    )
    recent_pred_df = score_netflow_with_champion(
        station_df=recent_df,
        inflow_model_path=inflow_model_path,
        outflow_model_path=outflow_model_path,
    )

    reference_count = int(reference_pred_df.count())
    recent_count = int(recent_pred_df.count())
    if reference_count == 0 or recent_count == 0:
        return {
            "concept_reference_row_count": reference_count,
            "concept_recent_row_count": recent_count,
            "concept_reference_asymmetric_rmse": None,
            "concept_recent_asymmetric_rmse": None,
            "concept_reference_asymmetric_mae": None,
            "concept_recent_asymmetric_mae": None,
            "concept_rmse_ratio": None,
            "concept_mae_ratio": None,
        }

    reference_rmse = asymmetric_loss_mean(
        reference_pred_df,
        y_true_col="station_netflow",
        y_pred_col="model_netflow",
        loss_type="rmse",
    )
    recent_rmse = asymmetric_loss_mean(
        recent_pred_df,
        y_true_col="station_netflow",
        y_pred_col="model_netflow",
        loss_type="rmse",
    )
    reference_mae = asymmetric_loss_mean(
        reference_pred_df,
        y_true_col="station_netflow",
        y_pred_col="model_netflow",
        loss_type="mae",
    )
    recent_mae = asymmetric_loss_mean(
        recent_pred_df,
        y_true_col="station_netflow",
        y_pred_col="model_netflow",
        loss_type="mae",
    )

    rmse_ratio = None
    mae_ratio = None
    if reference_rmse is not None and reference_rmse > 0:
        rmse_ratio = float(recent_rmse / reference_rmse)
    if reference_mae is not None and reference_mae > 0:
        mae_ratio = float(recent_mae / reference_mae)

    return {
        "concept_reference_row_count": reference_count,
        "concept_recent_row_count": recent_count,
        "concept_reference_asymmetric_rmse": float(reference_rmse) if reference_rmse is not None else None,
        "concept_recent_asymmetric_rmse": float(recent_rmse) if recent_rmse is not None else None,
        "concept_reference_asymmetric_mae": float(reference_mae) if reference_mae is not None else None,
        "concept_recent_asymmetric_mae": float(recent_mae) if recent_mae is not None else None,
        "concept_rmse_ratio": rmse_ratio,
        "concept_mae_ratio": mae_ratio,
    }


def write_monitoring_rows(
    spark,
    monitoring_rows: list[dict],
    monitoring_path: str,
    run_id: str,
) -> str | None:
    if not monitoring_rows:
        return None

    monitoring_df = spark.createDataFrame(monitoring_rows)
    dedupe_cols = ["run_id", "station_id"]
    monitoring_df = monitoring_df.dropDuplicates(dedupe_cols)

    monitoring_df.write.mode("overwrite").parquet(monitoring_path)

    table_name = None
    if should_write_summary_tables():
        catalog, schema, table, full_table_name = resolve_summary_table_target(
            default_table="monitoring",
            env_key="PIPELINE_TABLE_MONITORING",
        )
        try:
            table_name = append_run_df_to_delta_table(
                spark=spark,
                df=monitoring_df,
                catalog=catalog,
                schema=schema,
                table=table,
                run_id=run_id,
                run_id_col="run_id",
            )
        except ValueError as exc:
            if "Duplicate run id detected" not in str(exc):
                raise
            run_id_sql = str(run_id).replace("'", "''")
            spark.sql(f"DELETE FROM {full_table_name} WHERE run_id = '{run_id_sql}'")
            monitoring_df.write.format("delta").mode("append").saveAsTable(full_table_name)
            table_name = full_table_name

    return table_name


def main() -> None:
    spark = get_spark()
    apply_local_spark_defaults(spark)

    base_path = resolve_data_path()
    run_id, run_ts = resolve_pipeline_run_metadata(
        fallback_envs=("CHAMP_RUN_ID",),
        require_run_id_in_production=False,
    )
    cutoff_date = resolve_cutoff_date(os.environ.get("CUTOFF_DATE"))
    reference_window_days = resolve_positive_int(
        os.environ.get("MONITOR_REFERENCE_WINDOW_DAYS"),
        DEFAULT_REFERENCE_WINDOW_DAYS,
        "MONITOR_REFERENCE_WINDOW_DAYS",
    )
    recent_window_days = resolve_positive_int(
        os.environ.get("MONITOR_RECENT_WINDOW_DAYS"),
        DEFAULT_RECENT_WINDOW_DAYS,
        "MONITOR_RECENT_WINDOW_DAYS",
    )
    hod_cos_bin_count = resolve_positive_int(
        os.environ.get("MONITOR_HOD_COS_BIN_COUNT"),
        DEFAULT_HOD_COS_BIN_COUNT,
        "MONITOR_HOD_COS_BIN_COUNT",
    )
    threshold_covariate_drift_score = resolve_positive_float(
        os.environ.get("MONITOR_THRESHOLD_COVARIATE_DRIFT_SCORE"),
        DEFAULT_THRESHOLD_COVARIATE_DRIFT_SCORE,
        "MONITOR_THRESHOLD_COVARIATE_DRIFT_SCORE",
    )
    threshold_concept_rmse_ratio = resolve_positive_float(
        os.environ.get("MONITOR_THRESHOLD_CONCEPT_RMSE_RATIO"),
        DEFAULT_THRESHOLD_CONCEPT_RMSE_RATIO,
        "MONITOR_THRESHOLD_CONCEPT_RMSE_RATIO",
    )
    threshold_concept_mae_ratio = resolve_positive_float(
        os.environ.get("MONITOR_THRESHOLD_CONCEPT_MAE_RATIO"),
        DEFAULT_THRESHOLD_CONCEPT_MAE_RATIO,
        "MONITOR_THRESHOLD_CONCEPT_MAE_RATIO",
    )

    flow_df, stations_df = load_gold_inputs(spark, base_path)
    station_ids = resolve_target_station_ids(stations_df)

    champion_df = load_current_champion(spark, base_path).where(F.col("station_id").isin(station_ids))
    champion_rows = champion_df.select(
        "station_id",
        "run_id",
        "run_ts",
        "inflow_model_path",
        "outflow_model_path",
    ).collect()
    champion_by_station = {row["station_id"]: row for row in champion_rows}

    print("=== Stage 11 Monitoring start ===")
    print(f"Base path: {base_path}")
    print(f"Run id: {run_id}")
    print(f"Run ts: {run_ts}")
    print(f"Cutoff date: {cutoff_date}")
    print(f"Reference window days: {reference_window_days}")
    print(f"Recent window days: {recent_window_days}")
    print(f"Hod_cos bin count: {hod_cos_bin_count}")
    print(f"Threshold covariate_drift_score: {threshold_covariate_drift_score}")
    print(f"Threshold concept_rmse_ratio: {threshold_concept_rmse_ratio}")
    print(f"Threshold concept_mae_ratio: {threshold_concept_mae_ratio}")
    print(f"Target station count: {len(station_ids)}")
    print(f"Champion station count: {len(champion_by_station)}")

    monitoring_rows: list[dict] = []
    for station_id in station_ids:
        print(f"Monitoring station: {station_id}")
        station_flow_df = flow_df.filter(F.col("station_id") == station_id)
        station_flow_df = fill_missing_feature_values(station_flow_df)
        _train_df, test_df = split_train_test_by_cutoff(station_flow_df, cutoff_date=cutoff_date)

        test_row_count = int(test_df.count())
        if test_row_count == 0:
            monitoring_rows.append(
                {
                    "run_id": run_id,
                    "run_ts": run_ts,
                    "station_id": station_id,
                    "cutoff_date": cutoff_date,
                    "reference_window_days": reference_window_days,
                    "recent_window_days": recent_window_days,
                    "hod_cos_bin_count": hod_cos_bin_count,
                    "threshold_covariate_drift_score": threshold_covariate_drift_score,
                    "threshold_concept_rmse_ratio": threshold_concept_rmse_ratio,
                    "threshold_concept_mae_ratio": threshold_concept_mae_ratio,
                    "champion_run_id": None,
                    "champion_run_ts": None,
                    "reference_start_ts": None,
                    "recent_start_ts": None,
                    "max_test_ts": None,
                    "test_row_count": 0,
                    "reference_row_count": 0,
                    "recent_row_count": 0,
                    "covariate_hod_cos_psi_inflow": None,
                    "covariate_hod_cos_psi_outflow": None,
                    "covariate_drift_score": None,
                    "concept_reference_row_count": 0,
                    "concept_recent_row_count": 0,
                    "concept_reference_asymmetric_rmse": None,
                    "concept_recent_asymmetric_rmse": None,
                    "concept_reference_asymmetric_mae": None,
                    "concept_recent_asymmetric_mae": None,
                    "concept_rmse_ratio": None,
                    "concept_mae_ratio": None,
                    "concept_inflow_model_path": None,
                    "concept_outflow_model_path": None,
                    "monitoring_status": "no_test_rows_after_cutoff",
                    "monitoring_remarks": "No test rows after cutoff date.",
                }
            )
            continue

        reference_df, recent_df, reference_start, recent_start, max_test_ts = split_reference_recent_from_test(
            test_df,
            reference_window_days=reference_window_days,
            recent_window_days=recent_window_days,
        )

        reference_row_count = int(reference_df.count())
        recent_row_count = int(recent_df.count())

        cov_metrics = {
            "covariate_hod_cos_psi_inflow": None,
            "covariate_hod_cos_psi_outflow": None,
            "covariate_drift_score": None,
        }
        if reference_row_count > 0 and recent_row_count > 0:
            cov_metrics = compute_covariate_drift_metrics(
                reference_df=reference_df,
                recent_df=recent_df,
                hod_cos_bin_count=hod_cos_bin_count,
            )

        champion_row = champion_by_station.get(station_id)
        concept_metrics = {
            "concept_reference_row_count": 0,
            "concept_recent_row_count": 0,
            "concept_reference_asymmetric_rmse": None,
            "concept_recent_asymmetric_rmse": None,
            "concept_reference_asymmetric_mae": None,
            "concept_recent_asymmetric_mae": None,
            "concept_rmse_ratio": None,
            "concept_mae_ratio": None,
        }

        monitoring_status = "ok"
        monitoring_remarks = "Passed all configured thresholds."
        if reference_row_count == 0 or recent_row_count == 0:
            monitoring_status = "insufficient_reference_or_recent_rows"
            monitoring_remarks = "Insufficient reference or recent rows for drift checks."
        elif champion_row is None:
            monitoring_status = "missing_champion_for_station"
            monitoring_remarks = "Champion model paths missing for station."
        else:
            concept_metrics = compute_concept_drift_metrics(
                reference_df=reference_df,
                recent_df=recent_df,
                inflow_model_path=champion_row["inflow_model_path"],
                outflow_model_path=champion_row["outflow_model_path"],
            )

            fail_reasons: list[str] = []
            covariate_drift_score = cov_metrics["covariate_drift_score"]
            concept_rmse_ratio = concept_metrics["concept_rmse_ratio"]
            concept_mae_ratio = concept_metrics["concept_mae_ratio"]

            if (
                covariate_drift_score is not None
                and covariate_drift_score > threshold_covariate_drift_score
            ):
                fail_reasons.append(
                    "covariate_drift_score "
                    f"{covariate_drift_score:.6f} > threshold {threshold_covariate_drift_score:.6f}"
                )
            if (
                concept_rmse_ratio is not None
                and concept_rmse_ratio > threshold_concept_rmse_ratio
            ):
                fail_reasons.append(
                    "concept_rmse_ratio "
                    f"{concept_rmse_ratio:.6f} > threshold {threshold_concept_rmse_ratio:.6f}"
                )
            if (
                concept_mae_ratio is not None
                and concept_mae_ratio > threshold_concept_mae_ratio
            ):
                fail_reasons.append(
                    "concept_mae_ratio "
                    f"{concept_mae_ratio:.6f} > threshold {threshold_concept_mae_ratio:.6f}"
                )

            if fail_reasons:
                monitoring_status = "fail"
                monitoring_remarks = "; ".join(fail_reasons)

        monitoring_rows.append(
            {
                "run_id": run_id,
                "run_ts": run_ts,
                "station_id": station_id,
                "cutoff_date": cutoff_date,
                "reference_window_days": reference_window_days,
                "recent_window_days": recent_window_days,
                "hod_cos_bin_count": hod_cos_bin_count,
                "threshold_covariate_drift_score": threshold_covariate_drift_score,
                "threshold_concept_rmse_ratio": threshold_concept_rmse_ratio,
                "threshold_concept_mae_ratio": threshold_concept_mae_ratio,
                "champion_run_id": champion_row["run_id"] if champion_row is not None else None,
                "champion_run_ts": champion_row["run_ts"] if champion_row is not None else None,
                "reference_start_ts": reference_start,
                "recent_start_ts": recent_start,
                "max_test_ts": max_test_ts,
                "test_row_count": test_row_count,
                "reference_row_count": reference_row_count,
                "recent_row_count": recent_row_count,
                "covariate_hod_cos_psi_inflow": cov_metrics["covariate_hod_cos_psi_inflow"],
                "covariate_hod_cos_psi_outflow": cov_metrics["covariate_hod_cos_psi_outflow"],
                "covariate_drift_score": cov_metrics["covariate_drift_score"],
                "concept_reference_row_count": concept_metrics["concept_reference_row_count"],
                "concept_recent_row_count": concept_metrics["concept_recent_row_count"],
                "concept_reference_asymmetric_rmse": concept_metrics["concept_reference_asymmetric_rmse"],
                "concept_recent_asymmetric_rmse": concept_metrics["concept_recent_asymmetric_rmse"],
                "concept_reference_asymmetric_mae": concept_metrics["concept_reference_asymmetric_mae"],
                "concept_recent_asymmetric_mae": concept_metrics["concept_recent_asymmetric_mae"],
                "concept_rmse_ratio": concept_metrics["concept_rmse_ratio"],
                "concept_mae_ratio": concept_metrics["concept_mae_ratio"],
                "concept_inflow_model_path": champion_row["inflow_model_path"] if champion_row is not None else None,
                "concept_outflow_model_path": champion_row["outflow_model_path"] if champion_row is not None else None,
                "monitoring_status": monitoring_status,
                "monitoring_remarks": monitoring_remarks,
            }
        )

    monitoring_path = build_storage_path(base_path, "models", "summary", "monitoring", run_id)
    monitoring_table_name = write_monitoring_rows(
        spark=spark,
        monitoring_rows=monitoring_rows,
        monitoring_path=monitoring_path,
        run_id=run_id,
    )

    print("=== Stage 11 Monitoring complete ===")
    print(f"Saved monitoring path: {monitoring_path}")
    if monitoring_table_name:
        print(f"Appended monitoring table: {monitoring_table_name}")

    spark.createDataFrame(monitoring_rows).orderBy(F.col("station_id").asc()).show(200, truncate=False)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 11 drift monitoring")
    parser.add_argument(
        "--pipeline-station-id",
        default=os.environ.get("PIPELINE_STATION_ID"),
        help="Comma-separated station IDs to monitor (maps to PIPELINE_STATION_ID)",
    )
    parser.add_argument(
        "--pipeline-run-id",
        default=os.environ.get("PIPELINE_RUN_ID"),
        help="Run identifier (maps to PIPELINE_RUN_ID)",
    )
    parser.add_argument(
        "--pipeline-run-ts",
        default=os.environ.get("PIPELINE_RUN_TS"),
        help="Run timestamp UTC ISO (maps to PIPELINE_RUN_TS)",
    )
    parser.add_argument(
        "--champ-run-id",
        default=os.environ.get("CHAMP_RUN_ID"),
        help="Alternate run identifier (maps to CHAMP_RUN_ID)",
    )
    parser.add_argument(
        "--pipeline-job-run-id",
        default=os.environ.get("PIPELINE_JOB_RUN_ID"),
        help="Databricks job run id (maps to PIPELINE_JOB_RUN_ID)",
    )
    parser.add_argument(
        "--pipeline-repair-count",
        default=os.environ.get("PIPELINE_REPAIR_COUNT", "0"),
        help="Databricks repair count (maps to PIPELINE_REPAIR_COUNT)",
    )
    parser.add_argument(
        "--pipeline-mode",
        default=os.environ.get("PIPELINE_MODE"),
        help="Pipeline mode such as production/local (maps to PIPELINE_MODE)",
    )
    parser.add_argument(
        "--pipeline-enable-table-writes",
        default=os.environ.get("PIPELINE_ENABLE_TABLE_WRITES"),
        help="Enable summary table writes (maps to PIPELINE_ENABLE_TABLE_WRITES)",
    )
    parser.add_argument(
        "--pipeline-table-catalog",
        default=os.environ.get("PIPELINE_TABLE_CATALOG", "workspace"),
        help="Summary table catalog (maps to PIPELINE_TABLE_CATALOG)",
    )
    parser.add_argument(
        "--pipeline-table-schema",
        default=os.environ.get("PIPELINE_TABLE_SCHEMA", "bixi-fs"),
        help="Summary table schema (maps to PIPELINE_TABLE_SCHEMA)",
    )
    parser.add_argument(
        "--pipeline-table-monitoring",
        default=os.environ.get("PIPELINE_TABLE_MONITORING", "monitoring"),
        help="Stage 11 summary table name (maps to PIPELINE_TABLE_MONITORING)",
    )
    parser.add_argument(
        "--cutoff-date",
        default=os.environ.get("CUTOFF_DATE", DEFAULT_CUTOFF_DATE),
        help="Train/test split cutoff date (maps to CUTOFF_DATE)",
    )
    parser.add_argument(
        "--monitor-reference-window-days",
        default=os.environ.get("MONITOR_REFERENCE_WINDOW_DAYS", str(DEFAULT_REFERENCE_WINDOW_DAYS)),
        help="Reference window length in days (maps to MONITOR_REFERENCE_WINDOW_DAYS)",
    )
    parser.add_argument(
        "--monitor-recent-window-days",
        default=os.environ.get("MONITOR_RECENT_WINDOW_DAYS", str(DEFAULT_RECENT_WINDOW_DAYS)),
        help="Recent window length in days (maps to MONITOR_RECENT_WINDOW_DAYS)",
    )
    parser.add_argument(
        "--monitor-hod-cos-bin-count",
        default=os.environ.get("MONITOR_HOD_COS_BIN_COUNT", str(DEFAULT_HOD_COS_BIN_COUNT)),
        help="Bin count for hod_cos PSI (maps to MONITOR_HOD_COS_BIN_COUNT)",
    )
    parser.add_argument(
        "--monitor-threshold-covariate-drift-score",
        default=os.environ.get(
            "MONITOR_THRESHOLD_COVARIATE_DRIFT_SCORE",
            str(DEFAULT_THRESHOLD_COVARIATE_DRIFT_SCORE),
        ),
        help="Fail threshold for covariate_drift_score (maps to MONITOR_THRESHOLD_COVARIATE_DRIFT_SCORE)",
    )
    parser.add_argument(
        "--monitor-threshold-concept-rmse-ratio",
        default=os.environ.get(
            "MONITOR_THRESHOLD_CONCEPT_RMSE_RATIO",
            str(DEFAULT_THRESHOLD_CONCEPT_RMSE_RATIO),
        ),
        help="Fail threshold for concept_rmse_ratio (maps to MONITOR_THRESHOLD_CONCEPT_RMSE_RATIO)",
    )
    parser.add_argument(
        "--monitor-threshold-concept-mae-ratio",
        default=os.environ.get(
            "MONITOR_THRESHOLD_CONCEPT_MAE_RATIO",
            str(DEFAULT_THRESHOLD_CONCEPT_MAE_RATIO),
        ),
        help="Fail threshold for concept_mae_ratio (maps to MONITOR_THRESHOLD_CONCEPT_MAE_RATIO)",
    )
    args = parser.parse_args()

    _set_env_if_provided("PIPELINE_STATION_ID", args.pipeline_station_id)
    _set_env_if_provided("PIPELINE_RUN_ID", args.pipeline_run_id)
    _set_env_if_provided("PIPELINE_RUN_TS", args.pipeline_run_ts)
    _set_env_if_provided("CHAMP_RUN_ID", args.champ_run_id)
    _set_env_if_provided("PIPELINE_JOB_RUN_ID", args.pipeline_job_run_id)
    _set_env_if_provided("PIPELINE_REPAIR_COUNT", args.pipeline_repair_count)
    _set_env_if_provided("PIPELINE_MODE", args.pipeline_mode)
    _set_env_if_provided("PIPELINE_ENABLE_TABLE_WRITES", args.pipeline_enable_table_writes)
    _set_env_if_provided("PIPELINE_TABLE_CATALOG", args.pipeline_table_catalog)
    _set_env_if_provided("PIPELINE_TABLE_SCHEMA", args.pipeline_table_schema)
    _set_env_if_provided("PIPELINE_TABLE_MONITORING", args.pipeline_table_monitoring)
    _set_env_if_provided("CUTOFF_DATE", args.cutoff_date)
    _set_env_if_provided("MONITOR_REFERENCE_WINDOW_DAYS", args.monitor_reference_window_days)
    _set_env_if_provided("MONITOR_RECENT_WINDOW_DAYS", args.monitor_recent_window_days)
    _set_env_if_provided("MONITOR_HOD_COS_BIN_COUNT", args.monitor_hod_cos_bin_count)
    _set_env_if_provided(
        "MONITOR_THRESHOLD_COVARIATE_DRIFT_SCORE",
        args.monitor_threshold_covariate_drift_score,
    )
    _set_env_if_provided(
        "MONITOR_THRESHOLD_CONCEPT_RMSE_RATIO",
        args.monitor_threshold_concept_rmse_ratio,
    )
    _set_env_if_provided(
        "MONITOR_THRESHOLD_CONCEPT_MAE_RATIO",
        args.monitor_threshold_concept_mae_ratio,
    )

    main()
