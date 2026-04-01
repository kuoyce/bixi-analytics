import os

import h3
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from sparkutils import get_spark, resolve_data_path, read_table, write_table, apply_local_spark_defaults


def _safe_int_resolution(raw_value: str, default: int = 9) -> int:
    try:
        value = int(float(raw_value))
    except Exception:
        value = default
    return max(0, min(15, value))


def main():
    spark = get_spark()
    apply_local_spark_defaults(spark)

    base_path = resolve_data_path()
    storage_format = os.environ.get("SILVER_STORAGE_FORMAT", "parquet")
    h3_resolution = _safe_int_resolution(os.environ.get("H3_RESOLUTION", "9"), default=9)

    station_cleaning_base = f"{base_path}/silver/station_cleaning"
    canonical_table_path = f"{station_cleaning_base}/station_canonical_summary"

    h3_mapping_table_path = f"{station_cleaning_base}/station_canonical_h3_mapping"
    h3_mapping_csv_path = f"{station_cleaning_base}/station_canonical_h3_mapping_csv"

    canonical_df = read_table(spark, canonical_table_path, storage_format)

    required_cols = {"canonical_station_id", "canonical_lat", "canonical_lon"}
    missing_cols = sorted(list(required_cols - set(canonical_df.columns)))
    if missing_cols:
        raise ValueError(
            f"station_canonical_summary missing required columns: {missing_cols}"
        )

    # Keep one coordinate pair per canonical station and filter invalid locations.
    station_coords_df = (
        canonical_df.select("canonical_station_id", "canonical_lat", "canonical_lon")
        .dropDuplicates(["canonical_station_id"])
        .filter(F.col("canonical_station_id").isNotNull())
        .filter(F.col("canonical_lat").isNotNull() & F.col("canonical_lon").isNotNull())
        .filter((F.col("canonical_lat") != 0) & (F.col("canonical_lon") != 0))
        .filter((F.col("canonical_lat") != -1) & (F.col("canonical_lon") != -1))
    )

    @F.udf(returnType=StringType())
    def latlon_to_h3(lat, lon):
        return h3.geo_to_h3(float(lat), float(lon), h3_resolution)

    h3_mapping_df = (
        station_coords_df.withColumn("h3_cell", latlon_to_h3(F.col("canonical_lat"), F.col("canonical_lon")))
        .withColumn("h3_resolution", F.lit(h3_resolution))
        .select(
            "canonical_station_id",
            "canonical_lat",
            "canonical_lon",
            "h3_cell",
            "h3_resolution",
        )
    )

    write_table(h3_mapping_df, h3_mapping_table_path, storage_format)
    h3_mapping_df.coalesce(1).write.mode("overwrite").option("header", True).csv(h3_mapping_csv_path)

    station_count = h3_mapping_df.count()
    h3_cell_count = h3_mapping_df.select("h3_cell").distinct().count()

    print(f"Saved H3 mapping table: {h3_mapping_table_path}")
    print(f"Saved H3 mapping CSV:   {h3_mapping_csv_path}")
    print(f"H3 resolution used:     {h3_resolution}")
    print(f"Stations mapped:        {station_count}")
    print(f"Distinct H3 cells:      {h3_cell_count}")


if __name__ == "__main__":
    main()
