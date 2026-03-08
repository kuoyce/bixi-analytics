import os

from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    lower,
    trim,
    regexp_replace,
    format_string,
    round,
    year,
    concat_ws,
    row_number,
    count,
    min,
    max,
)

from sparkutils import get_spark, resolve_data_path, read_table, write_table


def normalize_station_name(column):
    return lower(
        trim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(column, r"\s+", " "),
                    r"\s*/\s*",
                    " / ",
                ),
                r"\s*-\s*",
                "-",
            )
        )
    )


def main():
    spark = get_spark()
    spark.conf.set("spark.sql.shuffle.partitions", "200")

    base_path = resolve_data_path()
    # if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    #     default_format = "delta"
    # else:
    #     default_format = "parquet"
    default_format = "parquet"

    storage_format = os.environ.get("SILVER_STORAGE_FORMAT", default_format)

    rides_stage_path = f"{base_path}/silver/rides_stage"
    station_cleaning_base = f"{base_path}/silver/station_cleaning"
    mapping_csv_path = f"{station_cleaning_base}/station_direct_match_mapping_csv"
    mapping_table_path = f"{station_cleaning_base}/station_direct_match_mapping"
    canonical_table_path = f"{station_cleaning_base}/station_canonical_summary"

    silver_rides_df = read_table(spark, rides_stage_path, storage_format)

    start_points_df = (
        silver_rides_df.filter(
            col("start_station_name").isNotNull()
            & col("start_station_latitude").isNotNull()
            & col("start_station_longitude").isNotNull()
            & col("start_time_ms").isNotNull()
        )
        .select(
            format_string(
                "%.6f,%.6f",
                round(col("start_station_latitude"), 6),
                round(col("start_station_longitude"), 6),
            ).alias("coord_key"),
            round(col("start_station_latitude"), 6).alias("lat"),
            round(col("start_station_longitude"), 6).alias("lon"),
            normalize_station_name(col("start_station_name")).alias("normalized_name"),
            year(col("start_time_ms")).alias("trip_year"),
        )
    )

    end_points_df = (
        silver_rides_df.filter(
            col("end_station_name").isNotNull()
            & col("end_station_latitude").isNotNull()
            & col("end_station_longitude").isNotNull()
            & col("end_time_ms").isNotNull()
        )
        .select(
            format_string(
                "%.6f,%.6f",
                round(col("end_station_latitude"), 6),
                round(col("end_station_longitude"), 6),
            ).alias("coord_key"),
            round(col("end_station_latitude"), 6).alias("lat"),
            round(col("end_station_longitude"), 6).alias("lon"),
            normalize_station_name(col("end_station_name")).alias("normalized_name"),
            year(col("end_time_ms")).alias("trip_year"),
        )
    )

    station_points_df = start_points_df.unionByName(end_points_df)

    historical_keys_df = (
        station_points_df.filter(col("normalized_name").isNotNull())
        .groupBy("coord_key", "lat", "lon", "normalized_name")
        .agg(
            count("normalized_name").alias("observed_trip_count"),
            min("trip_year").alias("first_year_seen"),
            max("trip_year").alias("last_year_seen"),
        )
    )

    coord_totals_df = (
        station_points_df.groupBy("coord_key", "lat", "lon")
        .count()
        .withColumnRenamed("count", "coord_total_trips")
    )

    coord_rank_window = Window.orderBy(col("coord_total_trips").desc(), col("coord_key").asc())

    canonical_coord_df = (
        coord_totals_df.withColumn("station_rank", row_number().over(coord_rank_window))
        .withColumn("canonical_station_id", format_string("STN_%04d", col("station_rank")))
        .select(
            "coord_key",
            "canonical_station_id",
            col("lat").alias("canonical_lat"),
            col("lon").alias("canonical_lon"),
            "coord_total_trips",
        )
    )

    direct_match_mapping_df = (
        historical_keys_df.join(canonical_coord_df, on="coord_key", how="left")
        .withColumn("station_key", concat_ws("|", col("coord_key"), col("normalized_name")))
        .select(
            "station_key",
            "coord_key",
            "normalized_name",
            "canonical_station_id",
            "canonical_lat",
            "canonical_lon",
            "observed_trip_count",
            "first_year_seen",
            "last_year_seen",
        )
        .dropDuplicates(["station_key"])
    )

    write_table(direct_match_mapping_df, mapping_table_path, storage_format)
    write_table(canonical_coord_df, canonical_table_path, storage_format)
    direct_match_mapping_df.coalesce(1).write.mode("overwrite").option("header", True).csv(mapping_csv_path)

    total_station_keys = direct_match_mapping_df.count()
    unique_canonical = direct_match_mapping_df.select("canonical_station_id").distinct().count()

    print("=== Stage 2 complete: build station direct mapping ===")
    print(f"Storage format: {storage_format}")
    print(f"Rides source path: {rides_stage_path}")
    print(f"Direct mapping table path: {mapping_table_path}")
    print(f"Direct mapping CSV path: {mapping_csv_path}")
    print(f"Canonical summary path: {canonical_table_path}")
    print(f"Station key rows: {total_station_keys:,}")
    print(f"Unique canonical stations: {unique_canonical:,}")

    spark.stop()


if __name__ == "__main__":
    main()
