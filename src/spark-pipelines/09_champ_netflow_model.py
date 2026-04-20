"""
Stage 09: select and persist station-level netflow champions from Stage 08 combi outputs.

Inference usage:
1. Read models/summary/champion/current for one champion row per station.
2. Load inflow_model_path and outflow_model_path from that row.
3. Netflow prediction is predicted_inflow - predicted_outflow.

Champion is evaluated and persisted by this stage, not selected at inference runtime.
"""

import argparse
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sparkutils import (
    append_run_df_to_delta_table,
    apply_local_spark_defaults,
    get_spark,
    is_production_mode,
    resolve_data_path,
    resolve_summary_table_target,
    should_write_summary_tables,
)


def _set_env_if_provided(env_key: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[env_key] = str(value)


def is_missing_summary_path_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "path does not exist" in message
        or "path_not_found" in message
        or "no such file or directory" in message
    )


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


def load_combi_summary_history(spark, base_path: str, exclude_run_id: str | None = None) -> DataFrame | None:
    combi_root = build_storage_path(base_path, "models", "summary", "combi")
    run_paths = list_summary_run_paths(combi_root)
    if exclude_run_id:
        run_paths = [p for p in run_paths if os.path.basename(p) != exclude_run_id]
    if not run_paths:
        return None
    return spark.read.parquet(*run_paths)


def load_current_champion(spark, base_path: str) -> DataFrame | None:
    champion_path = build_storage_path(base_path, "models", "summary", "champion", "current")
    try:
        champion_df = spark.read.parquet(champion_path)
        # Spark reads are lazy; force tiny action so missing paths fail inside this try block.
        champion_df.limit(1).count()
        return champion_df
    except Exception as exc:
        if not is_missing_summary_path_error(exc):
            raise
        return None


def resolve_combi_run_id(combi_df: DataFrame) -> tuple[str, str]:
    env_run_id = os.environ.get("PIPELINE_RUN_ID") or os.environ.get("CHAMP_RUN_ID")
    if not env_run_id:
        job_run_id = os.environ.get("PIPELINE_JOB_RUN_ID")
        if job_run_id and job_run_id.strip():
            repair_count = os.environ.get("PIPELINE_REPAIR_COUNT", "0").strip() or "0"
            env_run_id = f"job_{job_run_id.strip()}_repair_{repair_count}"

    if env_run_id:
        row = (
            combi_df.where(F.col("run_id") == env_run_id)
            .orderBy(F.col("run_ts").desc_nulls_last())
            .select("run_id", "run_ts")
            .limit(1)
            .collect()
        )
        if not row:
            raise ValueError(f"Configured run_id={env_run_id} was not found in combi summary history")
        return row[0]["run_id"], row[0]["run_ts"]

    if is_production_mode():
        raise ValueError(
            "PIPELINE_RUN_ID is required in production mode for stage 09. "
            "Pass Databricks {{job.run_id}} plus {{job.repair_count}} as PIPELINE_RUN_ID, "
            "or provide PIPELINE_JOB_RUN_ID and PIPELINE_REPAIR_COUNT."
        )

    latest = (
        combi_df.orderBy(F.col("run_ts").desc_nulls_last(), F.col("run_id").desc_nulls_last())
        .select("run_id", "run_ts")
        .limit(1)
        .collect()
    )
    if not latest:
        raise ValueError("No run metadata found in combi summary history")

    return latest[0]["run_id"], latest[0]["run_ts"]


def select_best_combo_per_station(result_df: DataFrame) -> DataFrame:
    rank_window = Window.partitionBy("station_id", "target_col").orderBy(
        F.col("asymmetric_rmse_mean").asc_nulls_last(),
        F.col("asymmetric_mae_mean").asc_nulls_last(),
    )

    return (
        result_df
        .withColumn("_rn", F.row_number().over(rank_window))
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )


def build_champion_snapshot(
    latest_best_df: DataFrame,
    previous_champion_df: DataFrame | None,
    min_rmse_improvement: float,
) -> DataFrame:
    if previous_champion_df is None:
        return (
            latest_best_df
            .withColumn("decision", F.lit("init_no_previous"))
            .withColumn("previous_champion_run_id", F.lit(None).cast("string"))
            .withColumn("previous_champion_rmse", F.lit(None).cast("double"))
            .withColumn("rmse_improvement_vs_prev", F.lit(None).cast("double"))
            .withColumn("is_champion_replaced", F.lit(True))
        )

    prev_required = {
        "run_id", "run_ts", "station_id", "target_col",
        "asymmetric_rmse_mean", "asymmetric_mae_mean",
        "inflow_model_name", "inflow_best_params_json",
        "outflow_model_name", "outflow_best_params_json",
        "param_signature", "inflow_model_path", "outflow_model_path", "test_row_count",
    }
    missing_prev = prev_required - set(previous_champion_df.columns)
    if missing_prev:
        raise ValueError(f"Current champion snapshot missing columns: {sorted(missing_prev)}")

    key_cols = ["station_id", "target_col"]
    latest = latest_best_df.alias("latest")
    prev = previous_champion_df.alias("prev")
    joined = latest.join(prev, on=key_cols, how="fullouter")

    no_prev = F.col("prev.station_id").isNull()
    no_latest = F.col("latest.station_id").isNull()
    rmse_improvement = F.col("prev.asymmetric_rmse_mean") - F.col("latest.asymmetric_rmse_mean")
    significant = rmse_improvement > F.lit(float(min_rmse_improvement))

    use_latest = (
        F.when(no_prev, F.lit(True))
        .when(no_latest, F.lit(False))
        .when(significant, F.lit(True))
        .otherwise(F.lit(False))
    )

    decision = (
        F.when(no_prev, F.lit("init_no_previous"))
        .when(no_latest, F.lit("keep_previous_no_latest"))
        .when(significant, F.lit("replace_significant_improvement"))
        .otherwise(F.lit("keep_previous_minor_improvement"))
    )

    return joined.select(
        F.coalesce(F.col("latest.station_id"), F.col("prev.station_id")).alias("station_id"),
        F.coalesce(F.col("latest.target_col"), F.col("prev.target_col")).alias("target_col"),
        F.when(use_latest, F.col("latest.run_id")).otherwise(F.col("prev.run_id")).alias("run_id"),
        F.when(use_latest, F.col("latest.run_ts")).otherwise(F.col("prev.run_ts")).alias("run_ts"),
        F.when(use_latest, F.col("latest.inflow_model_name")).otherwise(F.col("prev.inflow_model_name")).alias("inflow_model_name"),
        F.when(use_latest, F.col("latest.inflow_best_params_json")).otherwise(F.col("prev.inflow_best_params_json")).alias("inflow_best_params_json"),
        F.when(use_latest, F.col("latest.outflow_model_name")).otherwise(F.col("prev.outflow_model_name")).alias("outflow_model_name"),
        F.when(use_latest, F.col("latest.outflow_best_params_json")).otherwise(F.col("prev.outflow_best_params_json")).alias("outflow_best_params_json"),
        F.when(use_latest, F.col("latest.param_signature")).otherwise(F.col("prev.param_signature")).alias("param_signature"),
        F.when(use_latest, F.col("latest.test_row_count")).otherwise(F.col("prev.test_row_count")).alias("test_row_count"),
        F.when(use_latest, F.col("latest.asymmetric_mae_mean")).otherwise(F.col("prev.asymmetric_mae_mean")).alias("asymmetric_mae_mean"),
        F.when(use_latest, F.col("latest.asymmetric_rmse_mean")).otherwise(F.col("prev.asymmetric_rmse_mean")).alias("asymmetric_rmse_mean"),
        F.when(use_latest, F.col("latest.inflow_model_path")).otherwise(F.col("prev.inflow_model_path")).alias("inflow_model_path"),
        F.when(use_latest, F.col("latest.outflow_model_path")).otherwise(F.col("prev.outflow_model_path")).alias("outflow_model_path"),
        F.col("prev.run_id").alias("previous_champion_run_id"),
        F.col("prev.asymmetric_rmse_mean").alias("previous_champion_rmse"),
        rmse_improvement.alias("rmse_improvement_vs_prev"),
        use_latest.alias("is_champion_replaced"),
        decision.alias("decision"),
    )


def save_current_champion_if_enabled(
    champion_current_df: DataFrame,
    champion_history_df: DataFrame,
    base_path: str,
    run_id: str,
) -> tuple[bool, str, str]:
    save_enabled = str(os.environ.get("CHAMP_SAVE_CURRENT", "1")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    champion_current_path = build_storage_path(base_path, "models", "summary", "champion", "current")
    champion_history_path = build_storage_path(base_path, "models", "summary", "champion", "history", run_id)

    if save_enabled:
        # Materialize snapshots without cache/persist so serverless compute is supported,
        # and break lineage with previous champion reads before overwrite.
        spark = champion_current_df.sparkSession
        current_rows = champion_current_df.collect()
        history_rows = champion_history_df.collect()
        current_df = spark.createDataFrame(current_rows, schema=champion_current_df.schema)
        history_df = spark.createDataFrame(history_rows, schema=champion_history_df.schema)
        current_df.write.mode("overwrite").parquet(champion_current_path)
        history_df.write.mode("overwrite").parquet(champion_history_path)

    return save_enabled, champion_current_path, champion_history_path


def main() -> None:

    spark = get_spark()
    apply_local_spark_defaults(spark)

    base_path = resolve_data_path()
    min_rmse_improvement = float(os.environ.get("CHAMP_MIN_RMSE_IMPROVEMENT", "0.05"))

    combi_all_df = load_combi_summary_history(spark, base_path)
    if combi_all_df is None:
        raise FileNotFoundError(
            "No combi summary runs found under models/summary/combi. "
            "Run stage 08 first."
        )

    run_id, run_ts = resolve_combi_run_id(combi_all_df)
    latest_run_df = combi_all_df.where(F.col("run_id") == run_id)
    latest_best_df = select_best_combo_per_station(latest_run_df)

    previous_champion_df = load_current_champion(spark, base_path)
    champion_df = build_champion_snapshot(
        latest_best_df=latest_best_df,
        previous_champion_df=previous_champion_df,
        min_rmse_improvement=min_rmse_improvement,
    ).orderBy(F.col("station_id").asc())

    champion_history_df = (
        champion_df
        .withColumn("history_run_id", F.lit(run_id))
        .withColumn("history_run_ts", F.lit(run_ts))
    )

    champion_saved, champion_current_path, champion_history_path = save_current_champion_if_enabled(
        champion_current_df=champion_df,
        champion_history_df=champion_history_df,
        base_path=base_path,
        run_id=run_id,
    )

    # If champion/current was overwritten, read it back to avoid stale file
    # references from pre-overwrite DataFrame lineage.
    if champion_saved:
        champion_display_df = spark.read.parquet(champion_current_path).orderBy(F.col("station_id").asc())
    else:
        champion_display_df = champion_df

    champion_history_table_name = None
    if should_write_summary_tables():
        catalog, schema, table, full_table_name = resolve_summary_table_target(
            default_table="champion_history",
            env_key="PIPELINE_TABLE_CHAMPION_HISTORY",
        )
        try:
            champion_history_table_name = append_run_df_to_delta_table(
                spark=spark,
                df=champion_history_df,
                catalog=catalog,
                schema=schema,
                table=table,
                run_id=run_id,
                run_id_col="history_run_id",
            )
        except ValueError as exc:
            if "Duplicate run id detected" not in str(exc):
                raise
            run_id_sql = str(run_id).replace("'", "''")
            spark.sql(f"DELETE FROM {full_table_name} WHERE history_run_id = '{run_id_sql}'")
            champion_history_df.write.format("delta").mode("append").saveAsTable(full_table_name)
            champion_history_table_name = full_table_name

    print("=== Stage 09 Champion Select ===")
    print(f"Base path: {base_path}")
    print(f"Combi run id: {run_id}")
    print(f"Combi run ts: {run_ts}")
    print(f"Min RMSE improvement threshold: {min_rmse_improvement}")
    print(f"Champion save enabled: {champion_saved}")
    if champion_saved:
        print(f"Saved champion current path: {champion_current_path}")
        print(f"Saved champion history path: {champion_history_path}")
    else:
        print("Champion snapshot not persisted (CHAMP_SAVE_CURRENT disabled).")
        print(f"Would write champion current path: {champion_current_path}")
        print(f"Would write champion history path: {champion_history_path}")
    if champion_history_table_name:
        print(f"Appended champion history table: {champion_history_table_name}")

    champion_display_df.show(200, truncate=False)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 09 champion netflow model selection")
    parser.add_argument(
        "--pipeline-run-id",
        default=os.environ.get("PIPELINE_RUN_ID"),
        help="Run identifier (maps to PIPELINE_RUN_ID)",
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
        "--pipeline-table-champion-history",
        default=os.environ.get("PIPELINE_TABLE_CHAMPION_HISTORY", "champion_history"),
        help="Stage 09 summary table name (maps to PIPELINE_TABLE_CHAMPION_HISTORY)",
    )
    parser.add_argument(
        "--champ-save-current",
        default=os.environ.get("CHAMP_SAVE_CURRENT", "1"),
        help="Persist champion current/history snapshot (maps to CHAMP_SAVE_CURRENT)",
    )
    parser.add_argument(
        "--champ-min-rmse-improvement",
        default=os.environ.get("CHAMP_MIN_RMSE_IMPROVEMENT", "0.05"),
        help="Minimum RMSE improvement to replace champion (maps to CHAMP_MIN_RMSE_IMPROVEMENT)",
    )
    args = parser.parse_args()

    _set_env_if_provided("PIPELINE_RUN_ID", args.pipeline_run_id)
    _set_env_if_provided("CHAMP_RUN_ID", args.champ_run_id)
    _set_env_if_provided("PIPELINE_JOB_RUN_ID", args.pipeline_job_run_id)
    _set_env_if_provided("PIPELINE_REPAIR_COUNT", args.pipeline_repair_count)
    _set_env_if_provided("PIPELINE_MODE", args.pipeline_mode)
    _set_env_if_provided("PIPELINE_ENABLE_TABLE_WRITES", args.pipeline_enable_table_writes)
    _set_env_if_provided("PIPELINE_TABLE_CATALOG", args.pipeline_table_catalog)
    _set_env_if_provided("PIPELINE_TABLE_SCHEMA", args.pipeline_table_schema)
    _set_env_if_provided("PIPELINE_TABLE_CHAMPION_HISTORY", args.pipeline_table_champion_history)
    _set_env_if_provided("CHAMP_SAVE_CURRENT", args.champ_save_current)
    _set_env_if_provided("CHAMP_MIN_RMSE_IMPROVEMENT", args.champ_min_rmse_improvement)

    main()
