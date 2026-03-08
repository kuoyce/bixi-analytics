import os

from pyspark.sql.functions import (
    col,
    lower,
    trim,
    regexp_replace,
    format_string,
    round,
    concat_ws,
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
    # spark.conf.set("spark.sql.files.maxRecordsPerFile", "500000")
    MAXRECORDSPERFILE = 500000
    spark.conf.set("spark.sql.shuffle.partitions", "200")

    base_path = resolve_data_path()
    # if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    #     default_format = "delta"
    # else:
    #     default_format = "parquet"

    default_format = "parquet"
    read_format = default_format
    write_format = default_format

    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        write_format = "delta"
        table_location = "workspace.`bixi-fs`"

    # storage_format = os.environ.get("SILVER_STORAGE_FORMAT", default_format)
    
    rides_stage_path = f"{base_path}/silver/rides_stage"
    rides_final_path = f"{base_path}/silver/rides"
    if write_format == "delta":
        rides_final_path = f"{table_location}.`silver-rides`"
    station_cleaning_base = f"{base_path}/silver/station_cleaning"
    mapping_csv_path = f"{station_cleaning_base}/station_direct_match_mapping_csv"
    mapping_table_path = f"{station_cleaning_base}/station_direct_match_mapping"

    silver_rides_df = read_table(spark, rides_stage_path, read_format)

    if os.path.exists(mapping_csv_path):
        mapping_df = (
            spark.read.option("header", True)
            .csv(mapping_csv_path)
            .select("station_key", "canonical_station_id")
            .dropDuplicates(["station_key"])
        )
        print(f"Using mapping source: {mapping_csv_path}")
    else:
        mapping_df = (
            read_table(spark, mapping_table_path, read_format)
            .select("station_key", "canonical_station_id")
            .dropDuplicates(["station_key"])
        )
        print(f"Using mapping source: {mapping_table_path}")

    keyed_rides_df = (
        silver_rides_df
        .withColumn("start_station_name_norm", normalize_station_name(col("start_station_name")))
        .withColumn("end_station_name_norm", normalize_station_name(col("end_station_name")))
        .withColumn(
            "start_coord_key",
            format_string(
                "%.6f,%.6f",
                round(col("start_station_latitude"), 6),
                round(col("start_station_longitude"), 6),
            ),
        )
        .withColumn(
            "end_coord_key",
            format_string(
                "%.6f,%.6f",
                round(col("end_station_latitude"), 6),
                round(col("end_station_longitude"), 6),
            ),
        )
        .withColumn("start_station_key", concat_ws("|", col("start_coord_key"), col("start_station_name_norm")))
        .withColumn("end_station_key", concat_ws("|", col("end_coord_key"), col("end_station_name_norm")))
    )

    start_map_df = mapping_df.select(
        col("station_key").alias("start_station_key"),
        col("canonical_station_id").alias("start_canonical_station_id"),
    )
    end_map_df = mapping_df.select(
        col("station_key").alias("end_station_key"),
        col("canonical_station_id").alias("end_canonical_station_id"),
    )

    augmented_df = keyed_rides_df.join(start_map_df, on="start_station_key", how="left").join(
        end_map_df, on="end_station_key", how="left"
    )

    augmented_df = augmented_df.repartition(200, col("ride_year"))
    ## TEMP
    # augmented_df = augmented_df.filter(col("ride_year") == 2026)

    write_table(augmented_df, rides_final_path, write_format, partition_cols=["ride_year"], maxRecordsPerFile=MAXRECORDSPERFILE)
    # if table_format:
    #     write_table(augmented_df, rides_final_table, table_format, partition_cols=["ride_year"], maxRecordsPerFile=MAXRECORDSPERFILE)
    # else:
    #     write_table(augmented_df, rides_final_path, storage_format, partition_cols=["ride_year"], maxRecordsPerFile=MAXRECORDSPERFILE)


    total_rows = augmented_df.count()
    start_unmapped_rows = augmented_df.filter(col("start_canonical_station_id").isNull()).count()
    end_unmapped_rows = augmented_df.filter(col("end_canonical_station_id").isNull()).count()

    start_mapped_rows = total_rows - start_unmapped_rows
    end_mapped_rows = total_rows - end_unmapped_rows

    start_mapping_rate = (start_mapped_rows / total_rows * 100.0) if total_rows else 0.0
    end_mapping_rate = (end_mapped_rows / total_rows * 100.0) if total_rows else 0.0

    print("=== Stage 3 complete: augment silver rides ===")
    print(f"Storage format: {write_format}")
    print(f"Rides source path: {rides_stage_path}")
    print(f"Rides final path: {rides_final_path}")
    print(f"Total rides checked: {total_rows:,}")
    print(f"Start mapping rate: {start_mapped_rows:,}/{total_rows:,} ({start_mapping_rate:.4f}%)")
    print(f"End mapping rate:   {end_mapped_rows:,}/{total_rows:,} ({end_mapping_rate:.4f}%)")

    spark.stop()


if __name__ == "__main__":
    main()
