"""Entry point: the only place that catches DsPrepError and picks an exit code."""

from __future__ import annotations

import logging
import sys
import uuid

import numpy as np
import polars as pl

from ds_prep.config import get_settings
from ds_prep.errors import ConfigError, DsPrepError
from ds_prep.logging_setup import configure_logging, run_id_var
from ds_prep.metrics import store_aov

log = logging.getLogger("ds_prep.cli")


def _fake_orders(n: int = 2_000, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "order_id": np.arange(n, dtype=np.int64),
            "store_id": rng.choice(np.arange(5, dtype=np.int64), n, p=[0.4, 0.3, 0.2, 0.09, 0.01]),
            "gmv": np.round(rng.lognormal(6.0, 0.5, n), 2),
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    run_id_var.set(uuid.uuid4().hex[:8])

    try:
        settings = get_settings()
    except ConfigError as exc:
        configure_logging("ERROR", as_json=True)
        log.error("config error", extra={"ctx": exc.context})  # noqa: TRY400 - no traceback for config
        return exc.exit_code

    configure_logging(settings.log_level, as_json=settings.log_json)
    log.info("start", extra={"env": settings.env, "argv": args})

    try:
        df = _fake_orders()
        if "--corrupt" in args:
            df = df.with_columns(
                pl.when(pl.col("order_id") < 3).then(-1.0).otherwise(pl.col("gmv")).alias("gmv")
            )
        out = store_aov(df, settings.min_orders_per_store)
        print(out)
    except DsPrepError as exc:
        log.exception("run failed", extra={"error": type(exc).__name__, "ctx": exc.context})
        return exc.exit_code

    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
