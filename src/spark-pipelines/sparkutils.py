
import os
from pyspark.sql import SparkSession

def get_spark():
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        spark = SparkSession.builder.appName("Bixi Analytics").getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(
            SparkSession.builder \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .appName("Bixi Analytics").master("local[*]")
        ).getOrCreate()

    return spark

def resolve_project_root():
    """
    resolve project root by searching for .git directory, recursively loop for parent directory until .git is found, if not found, return current directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, ".git")):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # reached root directory
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        current_dir = parent_dir
    return current_dir


def resolve_data_path():
    """
    search for data path in current directory, if not available, then search for data path in project directory. 
    else recursively search for data/ directory inside project directory.
    """
    
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return "/Volumes/workspace/bixi-fs/"
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data")
        if os.path.exists(data_path):
            return data_path
        
        project_root = resolve_project_root()
        data_path = os.path.join(project_root, "data")
        if os.path.exists(data_path):
            return data_path
        
        # recursively search for data/ directory inside project directory
        for root, dirs, files in os.walk(project_root):
            if "data" in dirs:
                return os.path.join(root, "data")
        
        raise FileNotFoundError("Data directory not found in current directory or project directory.")

def write_table(df, path: str, fmt: str = "parquet", mode: str = "overwrite", partition_cols=None, maxRecordsPerFile=None):
    writer = df.write.mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    if maxRecordsPerFile:
        writer = writer.option("maxRecordsPerFile", maxRecordsPerFile)
    if fmt == "delta":
        writer.format("delta").saveAsTable(path)
    else:
        writer.parquet(path)

def read_table(spark, path: str, fmt: str = "parquet"):
    if fmt == "delta":
        return spark.read.format("delta").load(path)
    return spark.read.parquet(path)

if __name__ == "__main__":
    print(resolve_data_path())