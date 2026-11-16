#!/usr/bin/env python3
"""Train and evaluate the fatigue-point model.

Training data: SIMULATED push-up sets (random tempo, fatigue speed, rep count),
passed through the EMG-only rep tracker. Evaluation:
  1. held-out simulated sets (same generator, so optimistic)
  2. the set whose rep timing was MEASURED from real video (EMG simulated from it)
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import make_sets as ms
import simulate_emg as sim
from emg_features import RepTracker
from fatigue_model import MODEL_PATH, K, early_features, stop_zone_rep

N_SETS = 400
HERE = Path(__file__).parent


def reps_from_emg(t, v):
    tr = RepTracker()
    for ti, vi in zip(t, v):
        tr.update(ti, vi)
    return tr.reps


def one_set(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(18, 43))
    t0 = rng.uniform(1.0, 1.8)
    slow = rng.uniform(2.0, 4.5)
    p = rng.uniform(1.5, 4.5)
    n0 = rng.uniform(10, 25)           # reps over which EMG fatigue builds (absolute)
    gain = rng.uniform(0.5, 1.2)
    bottoms = ms.gen_bottoms(n, t0, slow, p, rng)
    tg, drive = ms.drive_from_bottoms(bottoms, gain, rng, n0=n0)
    t, adc = sim.sensor_model(tg, drive, rng.uniform(300, 600), rng)
    reps = reps_from_emg(t, adc)
    y = stop_zone_rep(reps)
    if len(reps) < K + 2 or y is None or y <= K:
        return None
    return early_features(reps), y


def main():
    with ProcessPoolExecutor() as ex:
        out = [o for o in ex.map(one_set, range(N_SETS), chunksize=10) if o]
    X = np.array([o[0] for o in out])
    y = np.array([o[1] for o in out], float)
    print(f"{len(y)} usable simulated sets; stop-zone rep: mean {y.mean():.1f}, "
          f"range {y.min():.0f}-{y.max():.0f}")

    models = {
        "baseline (mean)": DummyRegressor(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gradient boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    res = {}
    for name, m in models.items():
        cv = cross_val_predict(m, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0))
        m.fit(Xtr, ytr)
        res[name] = {"cv_mae_reps": float(np.abs(cv - ytr).mean()),
                     "heldout_mae_reps": float(np.abs(m.predict(Xte) - yte).mean())}
        print(f"{name:18s} CV MAE {res[name]['cv_mae_reps']:.2f} | held-out MAE "
              f"{res[name]['heldout_mae_reps']:.2f} reps")

    best = min((k for k in res if k != "baseline (mean)"), key=lambda k: res[k]["cv_mae_reps"])
    final = models[best]
    final.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    # real-timing check: rep timing measured from video, EMG simulated from it
    d = np.genfromtxt(HERE / "data/sim_sets/set1_real_timing.csv", delimiter=",", skip_header=1)
    reps = reps_from_emg(d[:, 0], d[:, 1])
    actual = stop_zone_rep(reps)
    pred = float(final.predict([early_features(reps)])[0])
    base_pred = float(models["baseline (mean)"].predict([early_features(reps)])[0])
    real = {"actual_stop_rep": actual, "model_pred": round(pred, 1),
            "baseline_pred": round(base_pred, 1)}
    print(f"best model: {best}; real-timing set: actual stop-zone rep {actual}, "
          f"predicted {pred:.1f} (baseline {base_pred:.1f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pte = final.predict(Xte)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(yte, pte, s=12, alpha=0.6, label="held-out simulated sets")
    ax.scatter([actual], [pred], s=70, color="tab:red", marker="*", label="real-timing set")
    lim = [y.min() - 1, y.max() + 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("actual stop-zone rep"); ax.set_ylabel("predicted from first 8 reps")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(HERE / "data" / "fatigue_model_scatter.png", dpi=150)

    (HERE / "data" / "fatigue_model_results.json").write_text(json.dumps(
        {"n_sets": int(len(y)), "k_reps": K, "models": res, "best": best,
         "real_timing_set": real,
         "note": "trained and evaluated on SIMULATED sets; real-timing set uses simulated EMG"},
        indent=2))


if __name__ == "__main__":
    main()
