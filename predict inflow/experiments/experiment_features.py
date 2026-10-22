"""Height -> caudal: does richer height information beat the two-number curve?

Same training hours for every model (Marcos's split), judged on his test days and
on October data no model saw. Inputs only use height up to the end of the hour
being estimated, and rain from before it.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "predict inflow"))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from dados_novos import chuva_open_meteo, leituras_novas
from pipeline_config import INTERIM_DIR, STATIONS
from previsao import TARGET, fit_model, predict, split_train_test, training_table, tune

OUTUBRO = pd.Timestamp("2026-10-01")


def all_readings(st):
    """2-min readings from the stage-01 merge and the new October folder, from late June 2026."""
    columns = ["datetime", "caudal_l_s", "caudal_m3_h", "altura_m", "temperatura_agua_c"]
    old = pd.read_csv(INTERIM_DIR / f"{st}_merged.csv", parse_dates=["datetime"], usecols=columns)
    out = pd.concat([old, leituras_novas(st)[columns]]).drop_duplicates("datetime", keep="last")
    return out[out["datetime"] >= pd.Timestamp("2026-06-24")].sort_values("datetime").set_index("datetime")


def rain(st):
    return chuva_open_meteo(st, pd.Timestamp("2026-06-20"), pd.Timestamp("2026-10-15"))


def height_features(readings, diameter, precip):
    """One row per hour. Index = start of the hour; every value uses readings up to the end of that hour."""
    h = readings["altura_m"]
    h = h[(h > 0) & (h < diameter)]
    g = h.groupby(h.index.floor("h"))
    f = pd.DataFrame({
        "h_mean": g.mean(), "h_max": g.max(), "h_min": g.min(), "h_std": g.std(),
        "h_first": g.first(), "h_last": g.last(), "h_n": g.size(),
    })
    f = f[f["h_n"] >= 15]
    f = f.reindex(pd.date_range(f.index.min(), f.index.max(), freq="h"))
    f["h_change_in_hour"] = f["h_last"] - f["h_first"]
    for k in (1, 2, 3):
        f[f"h_lag{k}"] = f["h_mean"].shift(k)
    f["h_change_1h"] = f["h_mean"] - f["h_lag1"]
    for w in (6, 24):
        roll = f["h_mean"].rolling(w, min_periods=w // 2)
        f[f"h_mean_{w}h"] = roll.mean()
        f[f"h_min_{w}h"] = roll.min()
        f[f"h_max_{w}h"] = roll.max()
    f["hour"] = f.index.hour
    # Open-Meteo labels rain by the end of its hour: label 14:00 = rain 13:00-14:00.
    # For the hour starting at t, use labels up to t (rain that fell before t).
    p = precip.reindex(pd.date_range(precip.index.min(), precip.index.max(), freq="h")).fillna(0.0)
    for w in (3, 6, 24, 72):
        f[f"rain_{w}h"] = p.rolling(w, min_periods=1).sum().reindex(f.index)
    return f


def hourly_target(readings, diameter):
    ok = readings[(readings["caudal_l_s"] > 0) & (readings["altura_m"] > 0) & (readings["altura_m"] < diameter)]
    g = ok.groupby(ok.index.floor("h"))["caudal_m3_h"].agg(["mean", "size"])
    return g.loc[g["size"] >= 15, "mean"].rename(TARGET)


BASE = ["h_mean", "h_max"]
MODELS = {
    "2. today's model (height)": BASE,
    "3. + rain, hour of day": [*BASE, "rain_3h", "rain_6h", "hour"],
    "4. + within-hour detail": [*BASE, "h_min", "h_std", "h_change_in_hour"],
    "5. + history": [*BASE, "h_min", "h_std", "h_change_in_hour", "h_lag1", "h_lag2", "h_lag3",
                     "h_change_1h", "h_mean_6h", "h_min_6h", "h_max_6h", "h_mean_24h", "h_min_24h", "h_max_24h"],
}


def curve(train):
    b, log_a = np.polyfit(np.log(train["h_mean"]), np.log(train[TARGET]), 1)
    return lambda frame: np.exp(log_a) * frame["h_mean"] ** b


def metrics(y, p):
    daily = pd.DataFrame({"y": y, "p": p}).groupby(y.index.date).agg(["sum", "size"])
    daily = daily[daily[("y", "size")] >= 20]
    daily_err = (daily[("p", "sum")] / daily[("y", "sum")] - 1).abs().mean() * 100
    return {"R²": r2_score(y, p), "MAE m³/h": np.mean(np.abs(y - p)),
            "total vol %": 100 * (p.sum() / y.sum() - 1), "daily vol err %": daily_err}


def run(st):
    D = STATIONS[st]["collector_diameter_m"]
    readings = all_readings(st)
    feats = height_features(readings, D, rain(st))
    target = hourly_target(readings, D)

    marcos = training_table(st)
    train_idx, test_idx = (part.index for part in split_train_test(marcos))
    data = feats.join(target, how="inner")
    data["dia"] = data.index.date
    train, test = data.loc[data.index.intersection(train_idx)], data.loc[data.index.intersection(test_idx)]
    full = data[data.index < OUTUBRO]
    october = data[data.index >= OUTUBRO]

    rows = []
    for period, tr, te in (("Marcos's test", train, test), ("October (never seen)", full, october)):
        fc = curve(tr)
        rows.append({"period": period, "model": "1. two-number curve", **metrics(te[TARGET], fc(te))})
        for name, cols in MODELS.items():
            params = tune(tr, cols).params
            model = fit_model(tr, cols, params)
            rows.append({"period": period, "model": name, **metrics(te[TARGET], predict(model, te, cols))})
    out = pd.DataFrame(rows)
    print(f"\n===== {STATIONS[st]['folder']}: train {len(train)} h, test {len(test)} h | "
          f"all before October {len(full)} h, October {len(october)} h")
    for period, g in out.groupby("period", sort=False):
        print(f"  -- {period}")
        print(g.drop(columns="period").set_index("model").round(3).to_string())
    return out


if __name__ == "__main__":
    results = pd.concat({st: run(st) for st in STATIONS})
    results.to_csv(Path(__file__).with_name("experiment_features_results.csv"))
