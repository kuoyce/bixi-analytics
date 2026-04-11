from sparkutils import get_spark, apply_local_spark_defaults


def main():
    spark = get_spark()
    apply_local_spark_defaults(spark)
    print("Stage 3 has been retired: H3 mapping is no longer part of the pipeline.")
    print("No output tables were written.")


if __name__ == "__main__":
    main()
