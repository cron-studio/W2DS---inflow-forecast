"""Round 2: combine the curve with the model, add calendar, add water temperature.

Same rules as round 1: same training hours for every model, judged on Marcos's
test days and on October; inputs only use data up to the end of the hour.
"""
import pandas as pd

import experiment_features as e
from previsao import TARGET, fit_model, predict, split_train_test, training_table, tune

# Coimbra municipal holiday (4 Jul) and national holidays in the period.
FERIADOS = {pd.Timestamp(d).date() for d in ("2026-07-04", "2026-08-15", "2026-10-05")}


def extra_features(feats, readings):
    t = readings["temperatura_agua_c"]
    t = t[(t > 0) & (t < 40)].groupby(t[(t > 0) & (t < 40)].index.floor("h")).mean().reindex(feats.index)
    feats["temp_mean"] = t
    feats["temp_change_1h"] = t - t.shift(1)
    feats["temp_vs_24h"] = t - t.rolling(24, min_periods=12).mean()
    feats["weekend"] = (feats.index.dayofweek >= 5).astype(int)
    feats["holiday"] = pd.Series(feats.index.date, index=feats.index).isin(FERIADOS).astype(int)
    return feats


HISTORY = e.MODELS["5. + history"]
ML_MODELS = {
    "2. today's model (height)": e.BASE,
    "5. + history": HISTORY,
    "6. curve as an input + history": ["curve_q", *HISTORY],
    "8. + history + calendar": [*HISTORY, "hour", "weekend", "holiday"],
    "9. + history + water temperature": [*HISTORY, "temp_mean", "temp_change_1h", "temp_vs_24h"],
}


def run(st):
    D = e.STATIONS[st]["collector_diameter_m"]
    readings = e.all_readings(st)
    feats = extra_features(e.height_features(readings, D, e.rain(st)), readings)
    data = feats.join(e.hourly_target(readings, D), how="inner")
    data["dia"] = data.index.date
    train_idx, test_idx = (part.index for part in split_train_test(training_table(st)))
    train, test = data.loc[data.index.intersection(train_idx)], data.loc[data.index.intersection(test_idx)]
    full, october = data[data.index < e.OUTUBRO], data[data.index >= e.OUTUBRO]

    rows = []
    for period, tr, te in (("Marcos's test", train, test), ("October (never seen)", full, october)):
        tr, te = tr.copy(), te.copy()
        fc = e.curve(tr)  # fitted on the training hours only
        tr["curve_q"], te["curve_q"] = fc(tr), fc(te)
        rows.append({"period": period, "model": "1. two-number curve", **e.metrics(te[TARGET], te["curve_q"])})
        for name, cols in ML_MODELS.items():
            model = fit_model(tr, cols, tune(tr, cols).params)
            rows.append({"period": period, "model": name, **e.metrics(te[TARGET], predict(model, te, cols))})
        # 7: the model only learns how far off the curve is (caudal / curve), then multiplies the curve.
        ratio_tr = tr.assign(**{TARGET: tr[TARGET] / tr["curve_q"]})
        model = fit_model(ratio_tr, HISTORY, tune(ratio_tr, HISTORY).params)
        rows.append({"period": period, "model": "7. corrects the curve + history",
                     **e.metrics(te[TARGET], predict(model, te, HISTORY) * te["curve_q"])})
    out = pd.DataFrame(rows)
    order = sorted(out["model"].unique(), key=lambda m: int(m.split(".")[0]))
    print(f"\n===== {e.STATIONS[st]['folder']}: train {len(train)} h, test {len(test)} h | "
          f"before October {len(full)} h, October {len(october)} h")
    for period, g in out.groupby("period", sort=False):
        print(f"  -- {period}")
        print(g.drop(columns="period").set_index("model").loc[order].round(3).to_string())
    return out


if __name__ == "__main__":
    pd.concat({st: run(st) for st in e.STATIONS}).to_csv(
        e.Path(__file__).with_name("experiment_features_2_results.csv"))
