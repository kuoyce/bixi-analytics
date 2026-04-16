

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GeneralizedLinearRegression
from pyspark.ml.feature import VectorAssembler, PCA, UnivariateFeatureSelector
from pyspark.ml.feature import OneHotEncoderModel

import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import os
import re

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import seaborn as sns

from sparkutils import get_spark, resolve_data_path, apply_local_spark_defaults, asymmetric_loss_col, asymmetric_loss_mean

HARD_CODED_STATION_IDS: list[str] = ['STN_0001', 'STN_0002', 'STN_0003', 'STN_0004', 'STN_0005', 'STN_0006']

def load_gold_inputs(spark, base_path: str) -> tuple[DataFrame, DataFrame]:
    flow_df = spark.read.parquet(f"{base_path}/gold/station_flow")
    stations_df = spark.read.parquet(f"{base_path}/silver/station_cleaning/station_canonical_summary")
    return flow_df, stations_df


def parse_station_ids(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return sorted({token.strip() for token in raw_value.split(",") if token.strip()})


def resolve_target_station_ids(stations_df: DataFrame) -> list[str]:
    env_station_ids = parse_station_ids(os.environ.get("STAGE5_STATION_IDS"))
    hardcoded_station_ids = sorted({sid.strip() for sid in HARD_CODED_STATION_IDS if sid and sid.strip()})

    if env_station_ids:
        requested_station_ids = env_station_ids
        source = "env:STAGE5_STATION_IDS"
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
        raise ValueError("No station ids resolved. Provide STAGE5_STATION_IDS or populate HARD_CODED_STATION_IDS.")

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
        "fitted_model": fitted,
        "predictions": pred,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
    }

def get_baseline_models(TARGET_COL):
    return {
        "GBTRegressor": GBTRegressor(featuresCol="features", labelCol=TARGET_COL, seed=42),
        "RandomForestRegressor": RandomForestRegressor(featuresCol="features", labelCol=TARGET_COL, seed=42),
        "LinearRegression": LinearRegression(featuresCol="features", labelCol=TARGET_COL, maxIter=100, regParam=0.1),
        "DecisionTreeRegressor": DecisionTreeRegressor(featuresCol="features", labelCol=TARGET_COL, seed=42),
        "GeneralizedLinearRegression": GeneralizedLinearRegression(featuresCol="features", labelCol=TARGET_COL,family="poisson",link="log"),
    }

def build_model_for_station(train_df, test_df, TARGET_COL: str, CATEGORICAL_COLS: list[str], NUMERIC_COLS: list[str]) -> None:

    # print("Schema:")
    # station_flow_df.printSchema()

    #check for nulls in columns
    # null_counts = station_flow_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in station_flow_df.columns])
    # print(f"Null counts for station {station_id}:")
    # null_counts.show()

    stages = create_preprocessing_pipeline_stages(CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL)
    baseline_models = get_baseline_models(TARGET_COL)
    results = []
    for estimator__name, model in baseline_models.items():
        print(f"Training {estimator__name} - Target: {TARGET_COL}")
        estimator = Pipeline(stages=stages + [model])
        res = fit_and_evaluate(estimator__name, estimator, train_df, test_df, TARGET_COL)
        results.append(res)

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

    return reduced_categorical_cols, reduced_numeric_cols, importance_df
    



def main():
    spark = get_spark()
    apply_local_spark_defaults(spark)
    
    base_path = resolve_data_path()
    flow_df, stations_df = load_gold_inputs(spark, base_path)
    station_ids = resolve_target_station_ids(stations_df)

    NUMERIC_COLS, CATEGORICAL_COLS, BOOLEAN_COLS, TIME_COL, _ = get_columns()

    for idx, station_id in enumerate(station_ids, start=1):
        print(f"[{idx}/{len(station_ids)}] Building model for station: {station_id}")
        for TARGET_COL in ['station_inflow', 'station_outflow']:

            # Filter the flow DataFrame for the specific station
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

            results = build_model_for_station(
                train_df,
                test_df,
                TARGET_COL=TARGET_COL,
                CATEGORICAL_COLS=reduced_categorical_cols,
                NUMERIC_COLS=reduced_numeric_cols,
            )
            print(pd.DataFrame(results)[["model_name", "rmse", "mae"]].sort_values("rmse"))
            exit(0)


if __name__ == "__main__":
    main()