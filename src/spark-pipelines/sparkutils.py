
import datetime
import importlib
import os
import uuid
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


def _str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def is_databricks_runtime() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def sync_databricks_widgets_to_env() -> dict[str, str]:
    """
    On Databricks, copy all widget parameters into process environment variables.
    Returns the applied key/value mapping.
    """
    if not is_databricks_runtime():
        return {}

    dbutils_obj = None
    import_error = None

    # Import dbutils only after confirming Databricks runtime.
    try:
        dbutils_module = importlib.import_module("pyspark.dbutils")
        DBUtils = getattr(dbutils_module, "DBUtils")
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        dbutils_obj = DBUtils(spark)
    except Exception as exc:
        import_error = exc

    if dbutils_obj is None:
        try:
            import builtins

            dbutils_obj = getattr(builtins, "dbutils", None)
        except Exception:
            dbutils_obj = None

    if dbutils_obj is None:
        if import_error is not None:
            print(f"Info: Databricks runtime detected but dbutils is unavailable ({import_error})")
        else:
            print("Info: Databricks runtime detected but dbutils is unavailable")
        return {}

    try:
        params = dbutils_obj.widgets.getAll()
    except Exception as exc:
        print(f"Info: unable to read Databricks widgets ({exc})")
        return {}

    applied: dict[str, str] = {}
    for key, value in params.items():
        env_key = str(key)
        env_value = "" if value is None else str(value)
        os.environ[env_key] = env_value
        applied[env_key] = env_value

    if applied:
        sample_key = next(iter(applied))
        print(f"Loaded {len(applied)} Databricks widget parameter(s) into environment")
        print(f"Verified: {os.getenv(sample_key)}")

    return applied


def is_production_mode() -> bool:
    mode = os.environ.get("PIPELINE_MODE")
    if mode and mode.strip():
        return mode.strip().lower() in {"prod", "production", "databricks", "job"}
    return is_databricks_runtime()


def should_write_summary_tables() -> bool:
    if "PIPELINE_ENABLE_TABLE_WRITES" in os.environ:
        return _str_to_bool(os.environ.get("PIPELINE_ENABLE_TABLE_WRITES"), default=False)
    return is_databricks_runtime()


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def build_qualified_table_name(catalog: str, schema: str, table: str) -> str:
    return f"{quote_identifier(catalog)}.{quote_identifier(schema)}.{quote_identifier(table)}"


def resolve_summary_table_target(default_table: str, env_key: str | None = None) -> tuple[str, str, str, str]:
    catalog = os.environ.get("PIPELINE_TABLE_CATALOG", "workspace").strip()
    schema = os.environ.get("PIPELINE_TABLE_SCHEMA", "bixi-fs").strip()
    table_name = default_table
    if env_key:
        table_name = os.environ.get(env_key, default_table)
    table_name = str(table_name).strip()
    full_name = build_qualified_table_name(catalog, schema, table_name)
    return catalog, schema, table_name, full_name


def resolve_pipeline_run_metadata(
    fallback_envs: tuple[str, ...] = (),
    require_run_id_in_production: bool = False,
) -> tuple[str, str]:
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if not run_id:
        for env_key in fallback_envs:
            value = os.environ.get(env_key)
            if value and value.strip():
                run_id = value.strip()
                break

    if not run_id:
        job_run_id = os.environ.get("PIPELINE_JOB_RUN_ID")
        if job_run_id and job_run_id.strip():
            repair_count = os.environ.get("PIPELINE_REPAIR_COUNT", "0").strip() or "0"
            run_id = f"job_{job_run_id.strip()}_repair_{repair_count}"

    if require_run_id_in_production and is_production_mode() and not run_id:
        raise ValueError(
            "PIPELINE_RUN_ID is required in production mode. "
            "For Databricks jobs, pass PIPELINE_RUN_ID from {{job.run_id}} plus {{job.repair_count}}."
        )

    now_utc = datetime.datetime.now(datetime.UTC)
    if not run_id:
        run_id = f"run_{now_utc.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    run_ts = os.environ.get("PIPELINE_RUN_TS")
    if not run_ts:
        run_ts = now_utc.isoformat().replace("+00:00", "Z")

    return run_id, run_ts


def _table_exists(spark, catalog: str, schema: str, table: str) -> bool:
    escaped_table = table.replace("'", "\\'")
    return (
        spark.sql(
            f"SHOW TABLES IN {quote_identifier(catalog)}.{quote_identifier(schema)} LIKE '{escaped_table}'"
        )
        .where(F.col("tableName") == table)
        .limit(1)
        .count()
        > 0
    )


def append_run_df_to_delta_table(
    spark,
    df,
    catalog: str,
    schema: str,
    table: str,
    run_id: str,
    run_id_col: str = "run_id",
    partition_cols: list[str] | None = None,
) -> str:
    if run_id_col not in df.columns:
        raise ValueError(f"Cannot append to table: missing required run id column '{run_id_col}'")

    full_table_name = build_qualified_table_name(catalog, schema, table)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(catalog)}.{quote_identifier(schema)}")

    table_exists = _table_exists(spark, catalog, schema, table)
    if table_exists:
        duplicate_exists = (
            spark.table(full_table_name)
            .where(F.col(run_id_col) == run_id)
            .limit(1)
            .count()
            > 0
        )
        if duplicate_exists:
            raise ValueError(
                f"Duplicate run id detected for {full_table_name}: "
                f"{run_id_col}='{run_id}'. Refusing append."
            )

    writer = df.write.format("delta").mode("append")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table_name)
    return full_table_name

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

from multiprocessing.pool import ThreadPool

from pyspark import keyword_only
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.tuning import CrossValidator

class tsCrossValidator(CrossValidator):
    """
    Custom validator for time-series cross-validation.

    This class extends the functionality of PySpark's CrossValidator to support
    walk-forward time-series cross-validation. It splits the dataset into
    consecutive periods with each fold using data from the past as training
    and the most recent period as validation.

    In particular, it overrides the _kFold method (which is used in the fit method)
    """
    datetimeCol = Param(
        Params._dummy(), 
        "datetimeCol", 
        "Column name for splitting the data",
        typeConverter=TypeConverters.toString)
    
    timeSplit = Param(
        Params._dummy(), 
        "timeSplit", 
        "Length of time to leave in validation set. Should be some sort of timedelta or relativedelta")
    
    gap = Param(
        Params._dummy(), 
        "gap", 
        "Length of time to leave bas gap between train and validation")
    
    disableExpandingWindow = Param(
        Params._dummy(),
        "disableExpandingWindow",
        "Boolean for disabling expanding window folds and taking rolling windows instead.",
        typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None,
                 numFolds=3, datetimeCol = 'date', timeSplit=None, 
                 gap=None, disableExpandingWindow=False, parallelism=1, collectSubModels=False):

        super(tsCrossValidator, self).__init__(
            estimator=estimator, 
            estimatorParamMaps=estimatorParamMaps, 
            evaluator=evaluator, 
            numFolds=numFolds,
            parallelism=parallelism, 
            collectSubModels=collectSubModels
        )
       
        self._setDefault(gap=None, datetimeCol='date', timeSplit=None, disableExpandingWindow=False)

        # Explicitly set the provided values
        self._set(gap=gap, datetimeCol=datetimeCol, timeSplit=timeSplit, disableExpandingWindow=disableExpandingWindow)

        kwargs = self._input_kwargs
        self._set(**kwargs)
    
    def getDatetimeCol(self):
        return self.getOrDefault(self.datetimeCol)
    
    def setDatetimeCol(self, datetimeCol):
        return self._set(datetimeCol=datetimeCol)
    
    def getTimeSplit(self):
        return self.getOrDefault(self.timeSplit)
    
    def setTimeSplit(self, timeSplit):
        return self._set(timeSplit=timeSplit)
    
    def getDisableExpandingWindow(self):
        return self.getOrDefault(self.disableExpandingWindow)
    
    def setDisableExpandingWindow(self, disableExpandingWindow):
        return self._set(disableExpandingWindow=disableExpandingWindow)
    
    def getGap(self):
        return self.getOrDefault(self.gap)

    def setGap(self, gap):
        return self._set(gap=gap)

    def _kFold(self, dataset):
        nFolds = self.getOrDefault(self.numFolds)
        datetimeCol = self.getOrDefault(self.datetimeCol)
        timeSplit = self.getOrDefault(self.timeSplit)
        gap = self.getOrDefault(self.gap)
        disableExpandingWindow = self.getOrDefault(self.disableExpandingWindow)

        datasets = []
        endDate = dataset.agg({datetimeCol : 'max'}).collect()[0][0]
        trainLB = dataset.agg({datetimeCol: 'min'}).collect()[0][0]
        for i in reversed(range(nFolds)):
            validateUB = endDate - i * timeSplit
            validateLB = endDate - (i + 1) * timeSplit
            trainUB = validateLB - gap if gap is not None else validateLB

            val_condition = (dataset[datetimeCol] > validateLB) & (dataset[datetimeCol] <= validateUB)
            train_condition = (dataset[datetimeCol] <= trainUB) & (dataset[datetimeCol] >= trainLB)

            validation = dataset.filter(val_condition)
            train = dataset.filter(train_condition)

            datasets.append((train, validation))

            if disableExpandingWindow:
                trainLB += timeSplit
        
        return datasets

if __name__ == "__main__":
    print(resolve_data_path())