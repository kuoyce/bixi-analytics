import math
import os
from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sparkutils import get_spark, resolve_data_path, apply_local_spark_defaults


# Optional hardcoded station ids. If empty, the script will run for all stations.
HARD_CODED_STATION_IDS: list[str] = ['STN_0001', 'STN_0002', 'STN_0003', 'STN_0004', 'STN_0005', 'STN_0006']  # Example station ids; replace with actual ids or leave empty


def load_silver_inputs(spark, base_path: str) -> tuple[DataFrame, DataFrame, DataFrame]:
    rides_df = spark.read.parquet(f"{base_path}/silver/rides")
    stations_df = spark.read.parquet(f"{base_path}/silver/station_cleaning/station_canonical_summary")
    weather_df = spark.read.parquet(f"{base_path}/silver/weather/hourly_by_station")
    return rides_df, stations_df, weather_df


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


def prepare_weather_hourly(weather_df: DataFrame) -> DataFrame:
    return weather_df.dropDuplicates(["canonical_station_id", "ts_hour"])


def augment_flow_with_temporal_features(
    flow_df: DataFrame,
    timestamp_col: str = "ts_hour",
    timezone: str = "America/Montreal",
    dow_offset: float = 0.0,
    hod_offset: float = 0.0,
    moy_offset: float = 0.0,
) -> DataFrame:
    ts_local = F.from_utc_timestamp(F.col(timestamp_col), timezone)
    two_pi = 2.0 * math.pi
    dow_expr = F.dayofweek(ts_local).cast("double")
    hod_expr = F.hour(ts_local).cast("double")
    moy_expr = F.month(ts_local).cast("double")

    return (
        flow_df
        .withColumn("dow", dow_expr.cast("int"))
        .withColumn("is_weekday", F.dayofweek(ts_local).between(2, 6))
        .withColumn("hod", hod_expr.cast("int"))
        .withColumn("moy", moy_expr.cast("int"))
        .withColumn("dow_cos", F.cos((dow_expr + F.lit(dow_offset)) * F.lit(two_pi / 7.0)))
        .withColumn("hod_cos", F.cos((hod_expr + F.lit(hod_offset)) * F.lit(two_pi / 24.0)))
        .withColumn("moy_cos", F.cos((moy_expr + F.lit(moy_offset)) * F.lit(two_pi / 12.0)))
    )


def join_flow_with_weather(flow_df: DataFrame, weather_df: DataFrame) -> DataFrame:
    weather_hourly_df = prepare_weather_hourly(weather_df)
    return (
        flow_df
        .join(
            weather_hourly_df.select(
                F.col("canonical_station_id"),
                F.col("ts_hour"),
                F.col("temp"),
                F.col("precip"),
            ),
            on=[
                flow_df["ts_hour"] == weather_hourly_df["ts_hour"],
                flow_df["station_id"] == weather_hourly_df["canonical_station_id"],
            ],
            how="left",
        )
        .drop(weather_hourly_df["ts_hour"])
        .drop(weather_hourly_df["canonical_station_id"])
    )


def build_station_flow_agg(
    rides_df: DataFrame,
    station_id: str,
    time_grain: str = "hour",
) -> DataFrame:
    if time_grain not in {"hour", "day"}:
        raise ValueError("time_grain must be 'hour' or 'day'")

    trunc_unit = "hour" if time_grain == "hour" else "day"

    outflow_df = (
        rides_df
        .filter(F.col("start_canonical_station_id") == station_id)
        .select(
            F.lit(station_id).alias("station_id"),
            F.date_trunc(trunc_unit, F.col("start_time_ms")).alias("ts_hour"),
            F.lit(1).alias("station_outflow"),
            F.lit(0).alias("station_inflow"),
        )
    )

    inflow_df = (
        rides_df
        .filter(F.col("end_canonical_station_id") == station_id)
        .select(
            F.lit(station_id).alias("station_id"),
            F.date_trunc(trunc_unit, F.col("end_time_ms")).alias("ts_hour"),
            F.lit(0).alias("station_outflow"),
            F.lit(1).alias("station_inflow"),
        )
    )

    flow_df = outflow_df.unionByName(inflow_df)

    return (
        flow_df
        .groupBy("station_id", "ts_hour")
        .agg(
            F.sum("station_inflow").cast("long").alias("station_inflow"),
            F.sum("station_outflow").cast("long").alias("station_outflow"),
            (F.sum("station_inflow").cast("long") - F.sum("station_outflow").cast("long")).alias("station_netflow"),
        )
        .orderBy("ts_hour")
    )


def _haversine_distance_m(lat1_col, lon1_col, lat2_col, lon2_col):
    earth_radius_m = 6371000.0
    dlat = F.radians(lat2_col - lat1_col)
    dlon = F.radians(lon2_col - lon1_col)
    a = (
        F.pow(F.sin(dlat / 2.0), 2)
        + F.cos(F.radians(lat1_col)) * F.cos(F.radians(lat2_col)) * F.pow(F.sin(dlon / 2.0), 2)
    )
    return F.lit(2.0 * earth_radius_m) * F.asin(F.sqrt(a))


def get_radius_station_ids(
    stations_df: DataFrame,
    station_id: str,
    radius_meters: int,
) -> DataFrame:
    if radius_meters <= 0:
        raise ValueError("radius_meters must be positive")

    target_row = (
        stations_df
        .filter(F.col("canonical_station_id") == station_id)
        .select(
            F.col("canonical_lat").cast("double").alias("target_lat"),
            F.col("canonical_lon").cast("double").alias("target_lon"),
        )
        .limit(1)
        .collect()
    )
    if not target_row:
        raise ValueError(f"station_id '{station_id}' not found in station_canonical_summary")

    target_lat = target_row[0]["target_lat"]
    target_lon = target_row[0]["target_lon"]
    if target_lat is None or target_lon is None:
        raise ValueError(f"station_id '{station_id}' has null canonical coordinates")

    candidates_df = stations_df.select(
        F.col("canonical_station_id"),
        F.col("canonical_lat").cast("double").alias("canonical_lat"),
        F.col("canonical_lon").cast("double").alias("canonical_lon"),
    ).dropDuplicates(["canonical_station_id"])

    return (
        candidates_df
        .filter(F.col("canonical_lat").isNotNull() & F.col("canonical_lon").isNotNull())
        .withColumn(
            "distance_m",
            _haversine_distance_m(
                F.col("canonical_lat"),
                F.col("canonical_lon"),
                F.lit(target_lat),
                F.lit(target_lon),
            ),
        )
        .filter(F.col("distance_m") <= F.lit(float(radius_meters)))
        .select(F.col("canonical_station_id"))
    )


def add_radius_flow_agg(
    rides_df: DataFrame,
    stations_df: DataFrame,
    station_id: str,
    radius_meters: int,
    time_grain: str = "hour",
) -> DataFrame:
    if time_grain not in {"hour", "day"}:
        raise ValueError("time_grain must be 'hour' or 'day'")

    trunc_unit = "hour" if time_grain == "hour" else "day"
    radius_inflow_col = f"radius{radius_meters}m_inflow"
    radius_outflow_col = f"radius{radius_meters}m_outflow"

    radius_station_ids_df = get_radius_station_ids(
        stations_df=stations_df,
        station_id=station_id,
        radius_meters=radius_meters,
    ).withColumnRenamed("canonical_station_id", "radius_station_id")

    radius_outflow_df = (
        rides_df
        .join(
            radius_station_ids_df,
            rides_df["start_canonical_station_id"] == radius_station_ids_df["radius_station_id"],
            how="inner",
        )
        .select(
            F.lit(station_id).alias("station_id"),
            F.date_trunc(trunc_unit, F.col("start_time_ms")).alias("ts_hour"),
            F.lit(0).alias(radius_inflow_col),
            F.lit(1).alias(radius_outflow_col),
        )
    )

    radius_inflow_df = (
        rides_df
        .join(
            radius_station_ids_df,
            rides_df["end_canonical_station_id"] == radius_station_ids_df["radius_station_id"],
            how="inner",
        )
        .select(
            F.lit(station_id).alias("station_id"),
            F.date_trunc(trunc_unit, F.col("end_time_ms")).alias("ts_hour"),
            F.lit(1).alias(radius_inflow_col),
            F.lit(0).alias(radius_outflow_col),
        )
    )

    return (
        radius_outflow_df
        .unionByName(radius_inflow_df)
        .groupBy("station_id", "ts_hour")
        .agg(
            F.sum(radius_inflow_col).cast("long").alias(radius_inflow_col),
            F.sum(radius_outflow_col).cast("long").alias(radius_outflow_col),
        )
    )


def fill_missing_hours_with_zero_flow(flow_df: DataFrame, timestamp_col: str = "ts_hour") -> DataFrame:
    bounds = flow_df.agg(
        F.min(timestamp_col).alias("min_ts"),
        F.max(timestamp_col).alias("max_ts"),
    ).collect()[0]

    min_ts = bounds["min_ts"]
    max_ts = bounds["max_ts"]
    if min_ts is None or max_ts is None:
        return flow_df

    spark = flow_df.sparkSession
    expected_hours_df = spark.sql(
        f"""
        SELECT explode(sequence(
            TIMESTAMP('{min_ts}'),
            TIMESTAMP('{max_ts}'),
            INTERVAL 1 HOUR
        )) AS {timestamp_col}
        """
    )

    station_df = flow_df.select("station_id").dropDuplicates()
    complete_grid_df = station_df.crossJoin(expected_hours_df)

    return (
        complete_grid_df
        .join(flow_df, on=["station_id", timestamp_col], how="left")
        .fillna({"station_inflow": 0, "station_outflow": 0})
        .withColumn("station_inflow", F.col("station_inflow").cast("long"))
        .withColumn("station_outflow", F.col("station_outflow").cast("long"))
        .orderBy(timestamp_col)
    )


def add_lag_feature(flow_df: DataFrame, source_col: str, lag_window: int, output_col: str) -> DataFrame:
    w = Window.partitionBy("station_id").orderBy("ts_hour")
    return flow_df.withColumn(output_col, F.lag(F.col(source_col), lag_window).over(w))


def add_hardcoded_lag_features(flow_df: DataFrame) -> DataFrame:
    lag_specs = {
        "station_inflow_lag1": ("station_inflow", 1),
        "station_outflow_lag1": ("station_outflow", 1),
        "station_inflow_lag12": ("station_inflow", 12),
        "station_outflow_lag12": ("station_outflow", 12),
    }
    for output_col, (source_col, lag_window) in lag_specs.items():
        flow_df = add_lag_feature(flow_df, source_col=source_col, lag_window=lag_window, output_col=output_col)
    return flow_df


def add_rolling_mean_feature(flow_df: DataFrame, source_col: str, rolling_window: int, output_col: str) -> DataFrame:
    w = Window.partitionBy("station_id").orderBy("ts_hour").rowsBetween(-(rolling_window - 1), 0)
    return flow_df.withColumn(output_col, F.avg(F.col(source_col)).over(w))


def add_rolling_sum_feature(flow_df: DataFrame, source_col: str, rolling_window: int, output_col: str) -> DataFrame:
    w = Window.partitionBy("station_id").orderBy("ts_hour").rowsBetween(-(rolling_window - 1), 0)
    return flow_df.withColumn(output_col, F.sum(F.col(source_col)).over(w))


def add_hardcoded_rolling_features(flow_df: DataFrame) -> DataFrame:
    mean_specs = {
        "precip_rollmean3": ("precip", 3),
        "station_inflow_rollmean6": ("station_inflow", 6),
        "station_outflow_rollmean6": ("station_outflow", 6),
        "station_inflow_rollmean12": ("station_inflow", 12),
        "station_outflow_rollmean12": ("station_outflow", 12),
    }
    sum_specs = {
        "precip_rollsum3": ("precip", 3),
        "station_inflow_rollsum6": ("station_inflow", 6),
        "station_outflow_rollsum6": ("station_outflow", 6),
        "station_inflow_rollsum12": ("station_inflow", 12),
        "station_outflow_rollsum12": ("station_outflow", 12),
    }

    for output_col, (source_col, rolling_window) in mean_specs.items():
        flow_df = add_rolling_mean_feature(
            flow_df,
            source_col=source_col,
            rolling_window=rolling_window,
            output_col=output_col,
        )

    for output_col, (source_col, rolling_window) in sum_specs.items():
        flow_df = add_rolling_sum_feature(
            flow_df,
            source_col=source_col,
            rolling_window=rolling_window,
            output_col=output_col,
        )

    return flow_df


def add_hardcoded_radius_features(flow_df: DataFrame, radius_meters: int) -> DataFrame:
    inflow_col = f"radius{radius_meters}m_inflow"
    outflow_col = f"radius{radius_meters}m_outflow"

    lag_specs = {
        f"{inflow_col}_lag1": (inflow_col, 1),
        f"{outflow_col}_lag1": (outflow_col, 1),
        f"{inflow_col}_lag12": (inflow_col, 12),
        f"{outflow_col}_lag12": (outflow_col, 12),
    }
    rollmean_specs = {
        f"{inflow_col}_rollmean6": (inflow_col, 6),
        f"{outflow_col}_rollmean6": (outflow_col, 6),
        f"{inflow_col}_rollmean12": (inflow_col, 12),
        f"{outflow_col}_rollmean12": (outflow_col, 12),
    }
    rollsum_specs = {
        f"{inflow_col}_rollsum6": (inflow_col, 6),
        f"{outflow_col}_rollsum6": (outflow_col, 6),
        f"{inflow_col}_rollsum12": (inflow_col, 12),
        f"{outflow_col}_rollsum12": (outflow_col, 12),
    }

    for output_col, (source_col, lag_window) in lag_specs.items():
        flow_df = add_lag_feature(flow_df, source_col=source_col, lag_window=lag_window, output_col=output_col)

    for output_col, (source_col, rolling_window) in rollmean_specs.items():
        flow_df = add_rolling_mean_feature(
            flow_df,
            source_col=source_col,
            rolling_window=rolling_window,
            output_col=output_col,
        )

    for output_col, (source_col, rolling_window) in rollsum_specs.items():
        flow_df = add_rolling_sum_feature(
            flow_df,
            source_col=source_col,
            rolling_window=rolling_window,
            output_col=output_col,
        )

    return flow_df.drop(inflow_col, outflow_col)


def add_temp_bin_feature(flow_df: DataFrame, source_col: str = "temp", output_col: str = "temp_bin") -> DataFrame:
    return flow_df.withColumn(
        output_col,
        F.when(F.col(source_col).isNull(), None)
        .when(F.col(source_col) < -20, "<-20")
        .when(F.col(source_col) < -10, "-20:-10")
        .when(F.col(source_col) < 0, "-10:0")
        .when(F.col(source_col) < 10, "0:10")
        .when(F.col(source_col) < 15, "10:15")
        .when(F.col(source_col) < 20, "15:20")
        .when(F.col(source_col) < 25, "20:25")
        .when(F.col(source_col) < 30, "25:30")
        .otherwise("30+"),
    )


def add_hardcoded_temp_bin_features(flow_df: DataFrame) -> DataFrame:
    return add_temp_bin_feature(flow_df, source_col="temp", output_col="temp_bin")


def build_features_for_station(
    rides_df: DataFrame,
    stations_df: DataFrame,
    weather_df: DataFrame,
    station_id: str,
    radius_meters_list: Iterable[int],
    time_grain: str,
) -> DataFrame:
    flow_df = build_station_flow_agg(rides_df, station_id=station_id, time_grain=time_grain)
    flow_df = fill_missing_hours_with_zero_flow(flow_df)

    for radius_meters in sorted({int(r) for r in radius_meters_list}):
        radius_agg_df = add_radius_flow_agg(
            rides_df=rides_df,
            stations_df=stations_df,
            station_id=station_id,
            radius_meters=radius_meters,
            time_grain=time_grain,
        )
        radius_inflow_col = f"radius{radius_meters}m_inflow"
        radius_outflow_col = f"radius{radius_meters}m_outflow"

        flow_df = (
            flow_df
            .join(radius_agg_df, on=["station_id", "ts_hour"], how="left")
            .fillna({radius_inflow_col: 0, radius_outflow_col: 0})
            .withColumn(radius_inflow_col, F.col(radius_inflow_col).cast("long"))
            .withColumn(radius_outflow_col, F.col(radius_outflow_col).cast("long"))
        )

        flow_df = add_hardcoded_radius_features(flow_df, radius_meters=radius_meters)

    flow_df = join_flow_with_weather(flow_df, weather_df)
    flow_df = add_hardcoded_lag_features(flow_df)
    flow_df = add_hardcoded_rolling_features(flow_df)
    flow_df = add_hardcoded_temp_bin_features(flow_df)
    flow_df = augment_flow_with_temporal_features(flow_df)

    return flow_df


def write_gold_station_flow_partition_overwrite(flow_df: DataFrame, gold_path: str) -> None:
    # Dynamic partition overwrite updates only station_id partitions present in flow_df.
    flow_df.sparkSession.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    (
        flow_df
        .repartition("station_id")
        .write
        .mode("overwrite")
        .partitionBy("station_id")
        .parquet(gold_path)
    )


def main() -> None:
    spark = get_spark()
    apply_local_spark_defaults(spark)

    base_path = resolve_data_path()
    output_path = f"{base_path}/gold/station_flow"
    radius_meters_list = [100, 200, 500]
    time_grain = os.environ.get("STAGE5_TIME_GRAIN", "hour").strip().lower()

    if time_grain not in {"hour", "day"}:
        raise ValueError("STAGE5_TIME_GRAIN must be 'hour' or 'day'")

    rides_df, stations_df, weather_df = load_silver_inputs(spark, base_path)
    station_ids = resolve_target_station_ids(stations_df)

    print("=== Stage 5 start: build gold station flow features ===")
    print(f"Base path: {base_path}")
    print(f"Output path: {output_path}")
    print(f"Time grain: {time_grain}")
    print(f"Radii (m): {radius_meters_list}")
    print(f"Target station count: {len(station_ids)}")

    all_station_features_df = None
    for idx, station_id in enumerate(station_ids, start=1):
        print(f"[{idx}/{len(station_ids)}] Building features for station: {station_id}")
        station_features_df = build_features_for_station(
            rides_df=rides_df,
            stations_df=stations_df,
            weather_df=weather_df,
            station_id=station_id,
            radius_meters_list=radius_meters_list,
            time_grain=time_grain,
        )

        if all_station_features_df is None:
            all_station_features_df = station_features_df
        else:
            all_station_features_df = all_station_features_df.unionByName(station_features_df)

    if all_station_features_df is None:
        raise RuntimeError("No station features were produced.")

    write_gold_station_flow_partition_overwrite(all_station_features_df, output_path)

    result_rows = all_station_features_df.count()
    result_station_count = all_station_features_df.select("station_id").dropDuplicates().count()

    print("=== Stage 5 complete: gold features written ===")
    print(f"Rows written: {result_rows:,}")
    print(f"Stations written: {result_station_count}")
    print(f"Output path: {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
