"""Store-level metrics with explicit data contracts."""

from __future__ import annotations

import logging
from typing import Final

import polars as pl

from ds_prep.errors import DataValidationError

log = logging.getLogger(__name__)

REQUIRED: Final[dict[str, type[pl.DataType]]] = {
    "order_id": pl.Int64,
    "store_id": pl.Int64,
    "gmv": pl.Float64,
}


def validate_orders(df: pl.DataFrame) -> None:
    """Raise DataValidationError if ``df`` breaks the orders contract."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError("missing columns", missing=missing)

    wrong = {c: str(df.schema[c]) for c, t in REQUIRED.items() if not isinstance(df.schema[c], t)}
    if wrong:
        raise DataValidationError("wrong dtypes", wrong=wrong)

    n_neg = df.filter(pl.col("gmv") < 0).height
    if n_neg:
        raise DataValidationError("negative gmv", rows=n_neg)

    n_dup = int(df["order_id"].is_duplicated().sum())
    if n_dup:
        raise DataValidationError("duplicate order_id rows", rows=n_dup)


def store_aov(df: pl.DataFrame, min_orders: int) -> pl.DataFrame:
    """AOV per store, dropping stores with fewer than ``min_orders`` orders."""
    validate_orders(df)
    out = (
        df.group_by(pl.col("store_id"))
        .agg(orders=pl.len(), gmv=pl.col("gmv").sum())
        .with_columns(aov=pl.col("gmv") / pl.col("orders"))
        .filter(pl.col("orders") >= min_orders)
        .sort("store_id")
    )
    log.info(
        "store_aov computed",
        extra={
            "stores_in": df["store_id"].n_unique(),
            "stores_out": out.height,
            "min_orders": min_orders,
        },
    )
    return out
