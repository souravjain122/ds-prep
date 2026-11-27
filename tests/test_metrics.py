import polars as pl
import pytest

from ds_prep.errors import DataValidationError
from ds_prep.metrics import store_aov


def _orders(**over: list[object]) -> pl.DataFrame:
    base: dict[str, list[object]] = {
        "order_id": [1, 2, 3, 4],
        "store_id": [10, 10, 20, 20],
        "gmv": [100.0, 300.0, 50.0, 150.0],
    }
    base.update(over)
    return pl.DataFrame(
        base, schema={"order_id": pl.Int64, "store_id": pl.Int64, "gmv": pl.Float64}
    )


def test_aov_is_gmv_over_orders() -> None:
    out = store_aov(_orders(), min_orders=1)
    assert out["aov"].to_list() == [200.0, 100.0]


def test_min_orders_filter() -> None:
    out = store_aov(_orders(store_id=[10, 10, 10, 20]), min_orders=2)
    assert out["store_id"].to_list() == [10]


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        ({"gmv": [1.0, -2.0, 3.0, 4.0]}, "negative gmv"),
        ({"order_id": [1, 1, 2, 3]}, "duplicate order_id rows"),
    ],
)
def test_contract_violations(over: dict[str, list[object]], msg: str) -> None:
    with pytest.raises(DataValidationError, match=msg):
        store_aov(_orders(**over), min_orders=1)


def test_missing_column() -> None:
    with pytest.raises(DataValidationError, match="missing columns") as ei:
        store_aov(_orders().drop("gmv"), min_orders=1)
    assert ei.value.context["missing"] == ["gmv"]
