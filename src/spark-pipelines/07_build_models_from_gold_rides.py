
import argparse

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.storagelevel import StorageLevel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GeneralizedLinearRegression
from pyspark.ml.feature import VectorAssembler, PCA, UnivariateFeatureSelector
from pyspark.ml.feature import OneHotEncoderModel
from pyspark.ml.tuning import ParamGridBuilder

import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import os
import numpy as np
import uuid
import gc

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import seaborn as sns

from sparkutils import (
    append_run_df_to_delta_table,
    apply_local_spark_defaults,
    asymmetric_loss_col,
    asymmetric_loss_mean,
    get_spark,
    resolve_data_path,
    resolve_pipeline_run_metadata,
    resolve_summary_table_target,
    should_write_summary_tables,
)
from sparkutils import tsCrossValidator

import datetime

HARD_CODED_STATION_IDS: list[str] = ['STN_0001', 'STN_0002', 'STN_0003', 'STN_0004', 'STN_0005', 'STN_0006']
DEFAULT_SPARKML_TEMP_DFS_PATH = "/Volumes/workspace/bixi-fs/tmp/sparkml"

def load_gold_inputs(spark, base_path: str) -> tuple[DataFrame, DataFrame]:
    flow_df = spark.read.parquet(f"{base_path}/gold/station_flow")
    stations_df = spark.read.parquet(f"{base_path}/silver/station_cleaning/station_canonical_summary")
    return flow_df, stations_df


def parse_station_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return sorted({token.strip() for token in raw_value.split(",") if token.strip()})


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

def get_columns(target_col: str = 'station_inflow') -> tuple[list[str], list[str], list[str], str, str]:
    ALL_COLUMNS = ['ts_hour', 'station_inflow', 'station_outflow', 'station_netflow', 'radius100m_inflow_lag1', 'radius100m_outflow_lag1', 'radius100m_inflow_lag12', 'radius100m_outflow_lag12', 'radius100m_inflow_rollmean6', 'radius100m_outflow_rollmean6', 'radius100m_inflow_rollmean12', 'radius100m_outflow_rollmean12', 'radius100m_inflow_rollsum6', 'radius100m_outflow_rollsum6', 'radius100m_inflow_rollsum12', 'radius100m_outflow_rollsum12', 'radius200m_inflow_lag1', 'radius200m_outflow_lag1', 'radius200m_inflow_lag12', 'radius200m_outflow_lag12', 'radius200m_inflow_rollmean6', 'radius200m_outflow_rollmean6', 'radius200m_inflow_rollmean12', 'radius200m_outflow_rollmean12', 'radius200m_inflow_rollsum6', 'radius200m_outflow_rollsum6', 'radius200m_inflow_rollsum12', 'radius200m_outflow_rollsum12', 'radius500m_inflow_lag1', 'radius500m_outflow_lag1', 'radius500m_inflow_lag12', 'radius500m_outflow_lag12', 'radius500m_inflow_rollmean6', 'radius500m_outflow_rollmean6', 'radius500m_inflow_rollmean12', 'radius500m_outflow_rollmean12', 'radius500m_inflow_rollsum6', 'radius500m_outflow_rollsum6', 'radius500m_inflow_rollsum12', 'radius500m_outflow_rollsum12', 'temp', 'precip', 'station_inflow_lag1', 'station_outflow_lag1', 'station_inflow_lag12', 'station_outflow_lag12', 'precip_rollmean3', 'station_inflow_rollmean6', 'station_outflow_rollmean6', 'station_inflow_rollmean12', 'station_outflow_rollmean12', 'precip_rollsum3', 'station_inflow_rollsum6', 'station_outflow_rollsum6', 'station_inflow_rollsum12', 'station_outflow_rollsum12', 'temp_bin', 'dow', 'is_weekday', 'hod', 'moy', 'dow_cos', 'hod_cos', 'moy_cos']

    TIME_COL = "ts_hour"
    TARGET_COL = target_col   # change to station_outflow or station_netflow as needed

    # features to exclude from predictors
    BASE_EXCLUDE = [TIME_COL, "station_inflow", "station_outflow", "station_netflow"]

    # categorical columns detected from your schema preview
    CATEGORICAL_COLS = ["temp_bin"]

    # boolean columns
    BOOLEAN_COLS = ["is_weekday"]

    # numeric columns inferred from your schema preview
    NUMERIC_COLS = [c for c in ALL_COLUMNS if c not in BASE_EXCLUDE + CATEGORICAL_COLS + BOOLEAN_COLS]

    print("Target:", TARGET_COL)
    print("Numeric feature count:", len(NUMERIC_COLS))
    print("Categorical feature count:", len(CATEGORICAL_COLS))
    print("Boolean feature count:", len(BOOLEAN_COLS))

    return NUMERIC_COLS, CATEGORICAL_COLS, BOOLEAN_COLS, TIME_COL, TARGET_COL

def fillna_numerics_and_booleans(df: DataFrame, num_cols: list[str], bool_cols: list[str]) -> DataFrame:
    fill_map = {c: 0.0 for c in num_cols}
    fill_map.update({c: 0 for c in bool_cols})

    screen_df = df.fillna(fill_map)
    return screen_df

def split_train_test_by_cutoff(df: DataFrame, cutoff_date = "2025-08-01"):
    # split data
    train_df = df.filter(F.col("ts_hour") < cutoff_date)
    test_df  = df.filter(F.col("ts_hour") >= cutoff_date)
    return train_df, test_df


def create_preprocessing_pipeline_stages(categorical_cols, numerical_cols, target_col):
    stages = []
    # 1. Index and Encode Categorical Features
    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec")
        stages += [indexer, encoder]
    
    # 2. Assemble all features
    assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="keep")
    
    # 3. Standardize features
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)
    # selector = UnivariateFeatureSelector(
    #     featuresCol="scaled_features", outputCol="features", labelCol=target_col).setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(20)
    # pca = PCA(k=20, inputCol="scaled_features", outputCol="features")
    stages += [assembler, scaler]
    return stages

def evaluate_predictions(pred_df, label_col, evaluator_specs = ["rmse", "mae"]):
    out = {}
    for metric in evaluator_specs:
        ev = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName=metric)
        out[metric] = ev.evaluate(pred_df)
    return out

def fit_and_evaluate(model_name, estimator, train_data, eval_data, target_col):
    fitted = estimator.fit(train_data)
    pred = fitted.transform(eval_data)
    metrics = evaluate_predictions(pred, target_col)
    return {
        "model_name": model_name,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
    }


def save_best_model(best_model, model_path: str) -> None:
    best_model.write().overwrite().save(model_path)


def clear_spark_caches_safe(spark) -> None:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        print("Info: skipping Spark cache clear on Databricks runtime")
        return

    spark.catalog.clearCache()
    spark.sql("CLEAR CACHE")


def persist_df_best_effort(
    df: DataFrame,
    storage_level: StorageLevel,
    label: str,
) -> tuple[DataFrame, bool]:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        print(f"Info: {label} persistence skipped on Databricks runtime")
        return df, False

    persisted_df = df.persist(storage_level)
    # Force materialization so downstream reuse benefits from persistence.
    persisted_df.count()
    return persisted_df, True


def write_summary_rows(spark, summary_rows: list[dict], summary_path: str, run_id: str) -> str | None:
    if not summary_rows:
        return None

    summary_df = spark.createDataFrame(summary_rows)
    summary_df, summary_df_cached = persist_df_best_effort(
        summary_df,
        StorageLevel.MEMORY_AND_DISK,
        label="stage07 summary",
    )
    summary_df.count()
    summary_df.write.mode("overwrite").parquet(summary_path)

    table_name = None
    if should_write_summary_tables():
        catalog, schema, table, _ = resolve_summary_table_target(
            default_table="solo",
            env_key="PIPELINE_TABLE_SOLO",
        )
        table_name = append_run_df_to_delta_table(
            spark=spark,
            df=summary_df,
            catalog=catalog,
            schema=schema,
            table=table,
            run_id=run_id,
            run_id_col="run_id",
        )

    if summary_df_cached:
        summary_df.unpersist(blocking=True)
    return table_name


def build_storage_path(base_path: str, *parts: str) -> str:
    base = base_path.rstrip("/")
    suffix = "/".join(str(p).strip("/") for p in parts)
    return f"{base}/{suffix}" if suffix else base


def generate_run_metadata() -> tuple[str, str]:
    return resolve_pipeline_run_metadata(
        fallback_envs=("STAGE7_RUN_ID",),
        require_run_id_in_production=False,
    )


def resolve_sparkml_temp_dfs_path(cli_value: str | None) -> str | None:
    if cli_value and cli_value.strip():
        return cli_value.strip()

    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return os.environ.get("SPARKML_TEMP_DFS_PATH", DEFAULT_SPARKML_TEMP_DFS_PATH)

    return os.environ.get("SPARKML_TEMP_DFS_PATH")


def _set_env_if_provided(env_key: str, value: str | None) -> None:
    if value is None:
        return
    os.environ[env_key] = str(value)

def get_baseline_models(TARGET_COL):
    return {
        "GBTRegressor": {
            "estimator": GBTRegressor(featuresCol="features", labelCol=TARGET_COL, seed=42),
            "paramGrid": ParamGridBuilder().addGrid(GBTRegressor.maxDepth, [5, 7]).addGrid(GBTRegressor.maxIter, [100, 120]).build()
        },
        "RandomForestRegressor": {
            "estimator": RandomForestRegressor(featuresCol="features", labelCol=TARGET_COL, seed=42),
            "paramGrid": ParamGridBuilder().addGrid(RandomForestRegressor.maxDepth, [8, 10]).addGrid(RandomForestRegressor.numTrees, [120, 160]).build()
        },
        "LinearRegression": {
            "estimator": LinearRegression(featuresCol="features", labelCol=TARGET_COL, maxIter=100, regParam=0.1),
            "paramGrid": ParamGridBuilder().addGrid(LinearRegression.regParam, [0.01, 0.1, 1.0]).addGrid(LinearRegression.elasticNetParam, [0.1, 8.0]).build()
        },
        "DecisionTreeRegressor": {
            "estimator": DecisionTreeRegressor(featuresCol="features", labelCol=TARGET_COL, seed=42),
            "paramGrid": ParamGridBuilder().addGrid(DecisionTreeRegressor.maxDepth, [3, 5, 7]).build()
        },
        "GeneralizedLinearRegression": {
            "estimator": GeneralizedLinearRegression(featuresCol="features", labelCol=TARGET_COL,family="poisson",link="log"),
            "paramGrid": ParamGridBuilder().addGrid(GeneralizedLinearRegression.regParam, [0.01, 0.1, 1.0]).build()
        }
    }


def build_model_for_station(
    spark,
    station_id: str,
    run_id: str,
    run_ts: str,
    train_df,
    test_df,
    TARGET_COL: str,
    CATEGORICAL_COLS: list[str],
    NUMERIC_COLS: list[str],
    model_root_path: str,
) -> list[dict]:

    # print("Schema:")
    # station_flow_df.printSchema()

    #check for nulls in columns
    # null_counts = station_flow_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in station_flow_df.columns])
    # print(f"Null counts for station {station_id}:")
    # null_counts.show()

    stages = create_preprocessing_pipeline_stages(CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL)
    baseline_models = get_baseline_models(TARGET_COL)
    results = []
    cv_num_folds = int(os.environ.get("STAGE7_CV_NUM_FOLDS", "3"))
    for estimator__name, model_dt in baseline_models.items():
        model, paramGrid  = model_dt["estimator"], model_dt["paramGrid"]

        print(f"Tuning for {estimator__name} - Target: {TARGET_COL}")
        estimator = Pipeline(stages=stages + [model])
        print(f"  Grid size: {len(paramGrid)} | CV folds: {cv_num_folds}")

        pred = None
        tsv_model = None
        best_param_map = None
        best_params = None

        tsv_est = tsCrossValidator(
            estimator=estimator,
            estimatorParamMaps=paramGrid,
            timeSplit=datetime.timedelta(days=30),
            evaluator=RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mae"),
            datetimeCol='ts_hour',
            numFolds=cv_num_folds,
            parallelism=1,
            collectSubModels=False,
        )
        try:
            tsv_model = tsv_est.fit(train_df)

            # Extract and print winning parameters from the fitted CV model.
            best_idx = int(np.argmin(tsv_model.avgMetrics))
            best_mae = float(tsv_model.avgMetrics[best_idx])
            best_param_map = tsv_model.getEstimatorParamMaps()[best_idx]
            best_params = {param.name: value for param, value in best_param_map.items()}
            print(f"Best params for {estimator__name} ({TARGET_COL}) [cv_mae={best_mae:.4f}]:")
            for param_name, value in best_params.items():
                print(f"  {param_name}: {value}")

            pred = tsv_model.bestModel.transform(test_df)
            metrics = evaluate_predictions(pred, TARGET_COL)
            model_path = build_storage_path(model_root_path, station_id, TARGET_COL, estimator__name)
            save_best_model(tsv_model.bestModel, model_path)
            res = {
                "run_id": run_id,
                "run_ts": run_ts,
                "station_id": station_id,
                "target_col": TARGET_COL,
                "model_name": estimator__name,
                "model_path": model_path,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "cv_mae": best_mae,
                "best_params": best_params,
            }
            results.append(res)
        finally:
            # Spark Connect keeps fitted models in session-side cache; release references
            # aggressively to reduce cache growth when fitting many models in one session.
            del pred
            del tsv_model
            del tsv_est
            del estimator
            del model
            del paramGrid
            del best_param_map
            del best_params
            clear_spark_caches_safe(spark)
            gc.collect()

    return results


def prune_feature_columns_with_random_forest(
    train_df: DataFrame,
    target_col: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    keep_fraction: float = 0.6,
    min_keep: int = 12,
    seed: int = 42,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Fit a RandomForest on training data and keep the most important original columns.
    Categorical feature importance is aggregated across one-hot encoded dimensions.
    """
    if not categorical_cols and not numeric_cols:
        raise ValueError("No feature columns provided for pruning.")

    keep_fraction = min(max(keep_fraction, 0.0), 1.0)

    stages = create_preprocessing_pipeline_stages(categorical_cols, numeric_cols, target_col)
    stages.append(RandomForestRegressor(featuresCol="features", labelCol=target_col, seed=seed))

    rf_pipeline = Pipeline(stages=stages)
    rf_model = rf_pipeline.fit(train_df)
    rf_regressor_model = rf_model.stages[-1]
    importances = list(rf_regressor_model.featureImportances.toArray())

    encoder_models = [stage for stage in rf_model.stages if isinstance(stage, OneHotEncoderModel)]
    importance_rows = []
    offset = 0

    for col, encoder_model in zip(categorical_cols, encoder_models):
        category_size = encoder_model.categorySizes[0] if encoder_model.categorySizes else 0
        encoded_size = category_size - 1 if encoder_model.getDropLast() else category_size
        encoded_size = max(encoded_size, 0)
        col_importance = float(sum(importances[offset:offset + encoded_size])) if encoded_size > 0 else 0.0
        importance_rows.append({"feature": col, "importance": col_importance, "kind": "categorical"})
        offset += encoded_size

    for col in numeric_cols:
        col_importance = float(importances[offset]) if offset < len(importances) else 0.0
        importance_rows.append({"feature": col, "importance": col_importance, "kind": "numeric"})
        offset += 1

    importance_df = pd.DataFrame(importance_rows).sort_values("importance", ascending=False).reset_index(drop=True)
    if importance_df.empty:
        return categorical_cols, numeric_cols, importance_df

    keep_count = max(min_keep, math.ceil(len(importance_df) * keep_fraction))
    keep_count = min(keep_count, len(importance_df))
    kept_features = set(importance_df.head(keep_count)["feature"].tolist())

    reduced_categorical_cols = [c for c in categorical_cols if c in kept_features]
    reduced_numeric_cols = [c for c in numeric_cols if c in kept_features]

    # Ensure at least one feature survives pruning.
    if not reduced_categorical_cols and not reduced_numeric_cols:
        top_feature = importance_df.iloc[0]["feature"]
        if top_feature in categorical_cols:
            reduced_categorical_cols = [top_feature]
        else:
            reduced_numeric_cols = [top_feature]

    del rf_regressor_model
    del rf_model
    del rf_pipeline
    gc.collect()

    return reduced_categorical_cols, reduced_numeric_cols, importance_df



def main(sparkml_temp_dfs_path: str | None = None):

    spark = get_spark()
    apply_local_spark_defaults(spark)

    resolved_sparkml_temp_dfs_path = resolve_sparkml_temp_dfs_path(sparkml_temp_dfs_path)
    if resolved_sparkml_temp_dfs_path:
        os.environ["SPARKML_TEMP_DFS_PATH"] = resolved_sparkml_temp_dfs_path

    base_path = resolve_data_path()
    run_id, run_ts = generate_run_metadata()
    model_root_path = build_storage_path(base_path, "models", run_id)
    flow_df, stations_df = load_gold_inputs(spark, base_path)
    station_ids = resolve_target_station_ids(stations_df)

    print(f"Run ID: {run_id}")
    print(f"Run TS (UTC): {run_ts}")
    print(f"Stage 07 CV folds: {os.environ.get('STAGE7_CV_NUM_FOLDS', '3')}")

    NUMERIC_COLS, CATEGORICAL_COLS, BOOLEAN_COLS, TIME_COL, _ = get_columns()
    summary_rows = []

    for idx, station_id in enumerate(station_ids, start=1):
        print(f"[{idx}/{len(station_ids)}] Building model for station: {station_id}")
        for TARGET_COL in ['station_inflow', 'station_outflow']:
            station_flow_df = None
            station_df = None
            train_df = None
            test_df = None
            fit_train_df = None
            fit_test_df = None
            fit_train_cached = False
            fit_test_cached = False
            importance_df = None
            results = None

            try:
                # Filter the flow DataFrame for the specific station.
                station_flow_df = flow_df.filter(F.col("station_id") == station_id)
                station_df = fillna_numerics_and_booleans(station_flow_df, NUMERIC_COLS, BOOLEAN_COLS)
                train_df, test_df = split_train_test_by_cutoff(station_df)

                reduced_categorical_cols, reduced_numeric_cols, importance_df = prune_feature_columns_with_random_forest(
                    train_df=train_df,
                    target_col=TARGET_COL,
                    categorical_cols=CATEGORICAL_COLS,
                    numeric_cols=NUMERIC_COLS,
                )

                print(f"Building model for station {station_id} with {station_flow_df.count()} records")
                print(
                    f"Pruned feature set for {TARGET_COL}: "
                    f"{len(reduced_numeric_cols)} numeric, {len(reduced_categorical_cols)} categorical"
                )
                print("Top 10 feature importances:")
                print(importance_df.head(10))

                # Narrow the fit/test inputs to only required columns to reduce executor-side footprint.
                fit_columns = [TIME_COL, TARGET_COL] + reduced_categorical_cols + reduced_numeric_cols
                fit_train_df, fit_train_cached = persist_df_best_effort(
                    train_df.select(*fit_columns),
                    StorageLevel.DISK_ONLY,
                    label=f"{station_id}/{TARGET_COL} train",
                )
                fit_test_df, fit_test_cached = persist_df_best_effort(
                    test_df.select(*fit_columns),
                    StorageLevel.DISK_ONLY,
                    label=f"{station_id}/{TARGET_COL} test",
                )

                results = build_model_for_station(
                    spark,
                    station_id,
                    run_id,
                    run_ts,
                    fit_train_df,
                    fit_test_df,
                    TARGET_COL=TARGET_COL,
                    CATEGORICAL_COLS=reduced_categorical_cols,
                    NUMERIC_COLS=reduced_numeric_cols,
                    model_root_path=model_root_path,
                )
                print(pd.DataFrame(results)[["model_name", "rmse", "mae"]].sort_values("rmse"))
                summary_rows.extend(
                    [
                        {
                            "run_id": row["run_id"],
                            "run_ts": row["run_ts"],
                            "station_id": row["station_id"],
                            "target_col": row["target_col"],
                            "model_name": row["model_name"],
                            "model_path": row["model_path"],
                            "rmse": row["rmse"],
                            "mae": row["mae"],
                            "cv_mae": row["cv_mae"],
                            "best_params_json": json.dumps(row["best_params"], default=str),
                        }
                        for row in results
                    ]
                )
            finally:
                if fit_train_cached and fit_train_df is not None:
                    try:
                        fit_train_df.unpersist(blocking=True)
                    except Exception:
                        pass
                if fit_test_cached and fit_test_df is not None:
                    try:
                        fit_test_df.unpersist(blocking=True)
                    except Exception:
                        pass
                clear_spark_caches_safe(spark)
                del station_flow_df
                del station_df
                del train_df
                del test_df
                del fit_train_df
                del fit_test_df
                del fit_train_cached
                del fit_test_cached
                del importance_df
                del results
                gc.collect()

        clear_spark_caches_safe(spark)
        gc.collect()

    solo_summary_path = build_storage_path(base_path, "models", "summary", "solo", run_id)
    solo_table_name = write_summary_rows(
        spark,
        summary_rows,
        solo_summary_path,
        run_id=run_id,
    )

    print(f"Saved solo summary path: {solo_summary_path}")
    if solo_table_name:
        print(f"Appended solo summary table: {solo_table_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 07 build models from gold rides")
    parser.add_argument(
        "--pipeline-station-id",
        default=os.environ.get("PIPELINE_STATION_ID"),
        help="Comma-separated station IDs to train (maps to PIPELINE_STATION_ID)",
    )
    parser.add_argument(
        "--stage7-cv-num-folds",
        default=os.environ.get("STAGE7_CV_NUM_FOLDS", "3"),
        help="Time-series CV folds (maps to STAGE7_CV_NUM_FOLDS)",
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
        "--pipeline-table-solo",
        default=os.environ.get("PIPELINE_TABLE_SOLO", "solo"),
        help="Stage 07 summary table name (maps to PIPELINE_TABLE_SOLO)",
    )
    parser.add_argument(
        "--sparkml-temp-dfs-path",
        default=os.environ.get(
            "SPARKML_TEMP_DFS_PATH",
            DEFAULT_SPARKML_TEMP_DFS_PATH if os.environ.get("DATABRICKS_RUNTIME_VERSION") else None,
        ),
        help="UC volume path used by Spark ML caching on shared/serverless Databricks clusters",
    )
    args = parser.parse_args()

    _set_env_if_provided("PIPELINE_STATION_ID", args.pipeline_station_id)
    _set_env_if_provided("STAGE7_CV_NUM_FOLDS", args.stage7_cv_num_folds)
    _set_env_if_provided("PIPELINE_RUN_ID", args.pipeline_run_id)
    _set_env_if_provided("PIPELINE_RUN_TS", args.pipeline_run_ts)
    _set_env_if_provided("PIPELINE_JOB_RUN_ID", args.pipeline_job_run_id)
    _set_env_if_provided("PIPELINE_REPAIR_COUNT", args.pipeline_repair_count)
    _set_env_if_provided("PIPELINE_MODE", args.pipeline_mode)
    _set_env_if_provided("PIPELINE_ENABLE_TABLE_WRITES", args.pipeline_enable_table_writes)
    _set_env_if_provided("PIPELINE_TABLE_CATALOG", args.pipeline_table_catalog)
    _set_env_if_provided("PIPELINE_TABLE_SCHEMA", args.pipeline_table_schema)
    _set_env_if_provided("PIPELINE_TABLE_SOLO", args.pipeline_table_solo)

    main(sparkml_temp_dfs_path=args.sparkml_temp_dfs_path)