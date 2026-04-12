from pyspark.sql.column import Column
from pyspark.sql import functions as F

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
