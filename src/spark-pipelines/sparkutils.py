
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column

def get_spark():
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        spark = SparkSession.builder.appName("Bixi Analytics").getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        local_driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "12g")
        spark = configure_spark_with_delta_pip(
            SparkSession.builder \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .config("spark.driver.memory", local_driver_memory) \
                .appName("Bixi Analytics").master("local[*]")
        ).getOrCreate()

    return spark


def apply_local_spark_defaults(spark):
    """
    Apply consistent, memory-safer Spark SQL defaults for local runs.
    Databricks runtime keeps cluster-level settings, so skip there.
    """
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return

    spark.conf.set("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "24"))
    spark.conf.set("spark.sql.adaptive.enabled", os.environ.get("SPARK_SQL_ADAPTIVE_ENABLED", "true"))
    spark.conf.set("spark.sql.files.maxPartitionBytes", os.environ.get("SPARK_MAX_PARTITION_BYTES", "33554432"))
    spark.conf.set(
        "spark.sql.parquet.enableVectorizedReader",
        os.environ.get("SPARK_ENABLE_VECTORIZED_READER", "false"),
    )


def set_log_level_safe(spark, level: str = "WARN") -> bool:
    """
    Best-effort Spark log level update.
    Returns True when log level is applied, False when runtime blocks access.
    """
    try:
        spark.sparkContext.setLogLevel(level)
        return True
    except Exception as exc:
        # Databricks serverless may block direct sparkContext/JVM access.
        print(f"Info: skipping Spark log level override ({exc})")
        return False

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

def asymmetric_loss_col(
    y_true: Column,
    y_pred: Column,
    alpha: float = 1.25,
    beta: float = 1.0,
    loss_type: str = "rmse",
) -> Column:
    """
    Build a Spark Column expression for asymmetric loss.

    Error definition follows: e = y_pred - y_true

    If loss_type == "rmse":
        L(e) = alpha * e^2  when e > 0
             = beta  * e^2  when e <= 0

    If loss_type == "mae":
        L(e) = alpha * |e|  when e > 0
             = beta  * |e|  when e <= 0
    """
    if loss_type not in {"rmse", "mae"}:
        raise ValueError("loss_type must be either 'rmse' or 'mae'.")

    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")

    e = y_pred - y_true
    base_loss = F.pow(e, 2) if loss_type == "rmse" else F.abs(e)
    return F.when(e > 0, F.lit(alpha) * base_loss).otherwise(F.lit(beta) * base_loss)


def asymmetric_loss_mean(
    df,
    y_true_col: str,
    y_pred_col: str,
    alpha: float = 1.25,
    beta: float = 1.0,
    loss_type: str = "rmse",
) -> float:
    """
    Compute mean asymmetric loss over a Spark DataFrame.
    """
    loss_expr = asymmetric_loss_col(
        y_true=F.col(y_true_col),
        y_pred=F.col(y_pred_col),
        alpha=alpha,
        beta=beta,
        loss_type=loss_type,
    )
    return df.select(F.avg(loss_expr).alias("asymmetric_loss")).first()["asymmetric_loss"]


def build_asymmetric_loss_test_df(spark):
    """
    Create deterministic sample data to validate asymmetric loss behavior.

    Includes a mix of overprediction (e > 0), underprediction (e < 0),
    and exact prediction (e = 0) cases.
    """
    rows = [
        (1, 10.0, 12.0),  # overprediction: e = +2
        (2, 10.0, 8.0),   # underprediction: e = -2
        (3, 20.0, 21.0),  # overprediction: e = +1
        (4, 20.0, 19.0),  # underprediction: e = -1
        (5, 15.0, 15.0),  # exact: e = 0
    ]
    return spark.createDataFrame(rows, ["sample_id", "y", "y_hat"])

if __name__ == "__main__":
    print(resolve_data_path())