#!/usr/bin/env python3
"""Generate synthetic quick-commerce practice data.

python make_orders.py --out data/ [--users 20000 --seed 7]

Writes:
  users.csv       user_id, signup_date, city, acquisition_channel
  orders.csv      order_id, user_id, store_id, city, order_ts, items, gmv, delivery_min, category
  experiment.csv  user_id, variant, exposure_date, pre_gmv_28d, orders_14d, gmv_14d
experiment.csv contains deliberate quirks for practice (don't reveal them to the learner).
"""
import argparse, os
import numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--users", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    cities = ["Delhi", "Gurugram", "Bengaluru", "Mumbai", "Hyderabad", "Pune"]
    n = a.users
    users = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "signup_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D"),
        "city": rng.choice(cities, n, p=[.25, .1, .25, .2, .1, .1]),
        "acquisition_channel": rng.choice(["organic", "paid_social", "referral", "search"], n, p=[.45, .25, .15, .15]),
    })
    stores = {c: [f"{c[:3].upper()}-{i:02d}" for i in range(1, 9)] for c in cities}

    # orders: per-user propensity, heavy-tailed GMV
    lam = rng.gamma(1.2, 4, n)
    counts = rng.poisson(lam)
    uid = np.repeat(users.user_id.values, counts)
    m = len(uid)
    u = users.set_index("user_id").loc[uid]
    start = u.signup_date.values
    end = np.datetime64("2026-09-30")
    span = ((end - start).astype("timedelta64[D]").astype(int)).clip(1)
    ts = start + (rng.random(m) * span).astype("timedelta64[D]") + rng.integers(7 * 60, 24 * 60, m).astype("timedelta64[m]")
    city = u.city.values
    items = 1 + rng.poisson(4, m)
    gmv = np.round(items * rng.lognormal(4.3, 0.55, m), 2)
    orders = pd.DataFrame({
        "order_id": np.arange(1, m + 1), "user_id": uid,
        "store_id": [stores[c][k] for c, k in zip(city, rng.integers(0, 8, m))],
        "city": city, "order_ts": ts, "items": items, "gmv": gmv,
        "delivery_min": np.round(rng.gamma(6, 2.0, m) + 4, 1),
        "category": rng.choice(["grocery", "dairy", "snacks", "personal_care", "household", "pharmacy"], m),
    }).sort_values("order_ts").reset_index(drop=True)
    orders["order_id"] = np.arange(1, m + 1)

    # experiment with hidden SRM (~53/47) and decaying novelty effect
    exp_users = users.sample(frac=0.6, random_state=a.seed).user_id.values
    k = len(exp_users)
    variant = np.where(rng.random(k) < 0.53, "treatment", "control")
    exposure = pd.to_datetime("2026-09-01") + pd.to_timedelta(rng.integers(0, 14, k), unit="D")
    pre = np.round(rng.lognormal(6.2, 0.9, k) * (rng.random(k) > 0.25), 2)
    base_rate = 0.5 + pre / (pre.mean() * 1.5)
    days_in = (pd.Timestamp("2026-09-14") - exposure).days.values
    lift = np.where(variant == "treatment", 0.10 * np.exp(-days_in / 6) + 0.015, 0)
    orders14 = rng.poisson(base_rate * (1 + lift))
    gmv14 = np.round(orders14 * rng.lognormal(5.6, 0.6, k) * (1 + 0.5 * lift), 2)
    experiment = pd.DataFrame({"user_id": exp_users, "variant": variant, "exposure_date": exposure.date,
                               "pre_gmv_28d": pre, "orders_14d": orders14, "gmv_14d": gmv14})

    users.to_csv(f"{a.out}/users.csv", index=False)
    orders.to_csv(f"{a.out}/orders.csv", index=False)
    experiment.to_csv(f"{a.out}/experiment.csv", index=False)
    print(f"users {len(users):,} | orders {len(orders):,} | experiment {len(experiment):,} -> {a.out}/")

if __name__ == "__main__":
    main()
