"""
Mini Project 2 - Rolls-Royce: Remaining Useful Life (RUL) prediction
Dataset : NASA C-MAPSS, subset FD001 (100 train engines, 100 test engines)
Model   : LSTM (sequence of last 30 cycles -> RUL in cycles)
Baselines: (1) Mean predictor  (2) Random Forest on single-cycle features

Pipeline (5 steps from the course):
  1. Define   - Input: multi-sensor time series  ->  Output: RUL (cycles)
  2. Data     - NASA C-MAPSS FD001
  3. Process  - RUL labels, drop flat sensors, split by engine, scale, make windows
  4. Build    - LSTM model + Random Forest baseline
  5. Evaluate - RMSE, MAE, NASA score, error distribution, maintenance alerts

Run:  python rul_lstm.py      (all plots + results are saved in outputs/)
"""
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ---------------- settings (hyperparameters) ----------------
SEED = 42
WINDOW = 30          # LSTM looks at the last 30 cycles of sensor data
RUL_CAP = 125        # piecewise-linear RUL: early life is treated as "healthy" (max 125)
HIDDEN = 64          # LSTM hidden units
EPOCHS = 60
BATCH = 256
LR = 1e-3
PATIENCE = 8         # early stopping
ALERT_LIMIT = 30     # maintenance alert if predicted RUL <= 30 cycles

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


# =========================== 1. LOAD DATA ===========================
def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), sep=r"\s+", header=None, names=COLS)


train = load("train_FD001.txt")
test = load("test_FD001.txt")
true_test_rul = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"), header=None).iloc[:, 0].values

print(f"Train rows: {len(train)}, engines: {train.unit.nunique()}")
print(f"Test rows : {len(test)}, engines: {test.unit.nunique()}")

# =========================== 2. RUL LABEL ===========================
# Training engines run until failure, so:  RUL = (last cycle of that engine) - (current cycle)
max_cycle = train.groupby("unit")["cycle"].transform("max")
train["RUL_raw"] = max_cycle - train["cycle"]
# Cap at 125: in early life the engine is healthy and sensors don't change,
# so the model cannot tell RUL 300 from RUL 200. Capping makes the target learnable.
train["RUL"] = train["RUL_raw"].clip(upper=RUL_CAP)

# =========================== 3. EDA PLOTS ===========================
plt.style.use("seaborn-v0_8-whitegrid")

# (a) engine life distribution
life = train.groupby("unit")["cycle"].max()
plt.figure(figsize=(7, 4))
plt.hist(life, bins=20, color="#1f5aa6", edgecolor="white")
plt.axvline(life.mean(), color="#c0392b", ls="--", label=f"mean = {life.mean():.0f} cycles")
plt.title("EDA 1: How long do engines last? (train, FD001)")
plt.xlabel("Total cycles until failure")
plt.ylabel("Number of engines")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda1_engine_life.png", dpi=150)
plt.close()

# (b) which sensors are flat (no information)?
sensor_cols = [f"s{i}" for i in range(1, 22)]
stds = train[sensor_cols + ["op1", "op2", "op3"]].std()
flat = stds[stds < 1e-3].index.tolist()
plt.figure(figsize=(9, 4))
colors = ["#c0392b" if c in flat else "#1f5aa6" for c in stds.index]
plt.bar(stds.index, np.log10(stds.values + 1e-6), color=colors)
plt.title("EDA 2: Sensor variation (red = constant, dropped)")
plt.ylabel("log10(standard deviation)")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda2_sensor_variation.png", dpi=150)
plt.close()

# Drop flat sensors + operating settings (FD001 has one operating condition)
features = [c for c in sensor_cols if c not in flat]
# s6 is almost constant too (only 2 values) - drop it as well
if "s6" in features and train["s6"].nunique() <= 2:
    features.remove("s6")
dropped = [c for c in sensor_cols if c not in features]
print("Dropped sensors:", dropped)
print("Used sensors   :", features)

# (c) sensor trends over life for 4 engines
plt.figure(figsize=(11, 6))
for k, s in enumerate(["s2", "s4", "s11", "s12"]):
    plt.subplot(2, 2, k + 1)
    for u in [1, 2, 3, 4]:
        d = train[train.unit == u]
        plt.plot(d["RUL_raw"], d[s], lw=1, label=f"engine {u}")
    plt.gca().invert_xaxis()
    plt.title(f"Sensor {s} vs remaining life")
    plt.xlabel("True RUL (cycles)  -> failure at 0")
    if k == 0:
        plt.legend(fontsize=7)
plt.suptitle("EDA 3: Sensors drift as the engine approaches failure")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda3_sensor_trends.png", dpi=150)
plt.close()

# (d) correlation of each sensor with RUL
corr = train[features + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
plt.figure(figsize=(7, 4.5))
plt.barh(corr.index, corr.values, color=["#c0392b" if v < 0 else "#1f5aa6" for v in corr.values])
plt.title("EDA 4: Correlation of each sensor with RUL")
plt.xlabel("Pearson correlation")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda4_correlation.png", dpi=150)
plt.close()

# (e) RUL label: raw vs capped
d = train[train.unit == 1]
plt.figure(figsize=(7, 4))
plt.plot(d["cycle"], d["RUL_raw"], label="Raw RUL (linear)", color="#7f8c8d", ls="--")
plt.plot(d["cycle"], d["RUL"], label=f"Capped RUL (max {RUL_CAP})", color="#1f5aa6", lw=2)
plt.title("EDA 5: How we label RUL (engine 1)")
plt.xlabel("Cycle")
plt.ylabel("RUL (cycles)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda5_rul_label.png", dpi=150)
plt.close()

# =========================== 4. SPLIT + SCALE ===========================
# Split BY ENGINE (not by row) -> no engine appears in both train and validation.
units = train.unit.unique()
rng = np.random.RandomState(SEED)
val_units = rng.choice(units, size=20, replace=False)
tr_df = train[~train.unit.isin(val_units)].copy()
va_df = train[train.unit.isin(val_units)].copy()
te_df = test.copy()

# Scaler is FIT ONLY on training engines, then applied to val and test (no leakage).
scaler = MinMaxScaler()
tr_df[features] = scaler.fit_transform(tr_df[features])
va_df[features] = scaler.transform(va_df[features])
te_df[features] = scaler.transform(te_df[features])
print(f"Train engines: {tr_df.unit.nunique()}, Val engines: {va_df.unit.nunique()}, Test engines: {te_df.unit.nunique()}")


# =========================== 5. MAKE SEQUENCES ===========================
def make_windows(df):
    """Every possible 30-cycle window -> label = RUL at the last cycle of the window."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        a = g[features].values
        r = g["RUL"].values
        for end in range(WINDOW, len(g) + 1):
            X.append(a[end - WINDOW:end])
            y.append(r[end - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def last_window(df):
    """For test engines we only predict at the LAST recorded cycle."""
    X = []
    for _, g in df.groupby("unit"):
        a = g[features].values
        if len(a) < WINDOW:  # pad short engines by repeating the first row
            a = np.vstack([np.repeat(a[:1], WINDOW - len(a), axis=0), a])
        X.append(a[-WINDOW:])
    return np.array(X, dtype=np.float32)


X_tr, y_tr = make_windows(tr_df)
X_va, y_va = make_windows(va_df)
X_te = last_window(te_df)
y_te = np.minimum(true_test_rul, RUL_CAP).astype(np.float32)
print(f"Windows -> train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")


# =========================== 6. METRICS ===========================
def nasa_score(y_true, y_pred):
    """Official C-MAPSS score: late predictions (d>0) are punished more than early ones."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def report(name, y_true, y_pred):
    return {
        "model": name,
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "NASA_score": round(nasa_score(y_true, y_pred), 1),
    }


results = []

# =========================== 7. BASELINES ===========================
# Baseline 1: always predict the average training RUL
mean_pred = np.full_like(y_te, y_tr.mean())
results.append(report("Baseline 1: Mean RUL", y_te, mean_pred))

# Baseline 2: Random Forest on the CURRENT cycle only (no memory of history)
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
rf.fit(X_tr[:, -1, :], y_tr)
rf_pred = rf.predict(X_te[:, -1, :])
results.append(report("Baseline 2: Random Forest", y_te, rf_pred))


# =========================== 8. LSTM MODEL ===========================
class LSTMRUL(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, 30, HIDDEN)
        return self.head(out[:, -1]).squeeze(-1)  # use the last time step's memory


model = LSTMRUL(len(features))
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Target is divided by RUL_CAP so the network learns values in 0..1 (much easier to train)
Xtr_t, ytr_t = torch.tensor(X_tr), torch.tensor(y_tr / RUL_CAP)
Xva_t, yva_t = torch.tensor(X_va), torch.tensor(y_va / RUL_CAP)

hist = {"train": [], "val": []}
best_val, best_state, wait = 1e9, None, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    tot = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    tr_rmse = (tot / len(perm)) ** 0.5 * RUL_CAP          # back to cycles
    model.eval()
    with torch.no_grad():
        va_rmse = loss_fn(model(Xva_t), yva_t).item() ** 0.5 * RUL_CAP
    hist["train"].append(tr_rmse)
    hist["val"].append(va_rmse)
    print(f"epoch {epoch:2d}  train RMSE {tr_rmse:6.2f}  val RMSE {va_rmse:6.2f}")
    if va_rmse < best_val - 0.01:
        best_val, wait = va_rmse, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

model.load_state_dict(best_state)
torch.save(best_state, f"{OUT_DIR}/lstm_rul_model.pt")
model.eval()
with torch.no_grad():
    lstm_pred = (model(torch.tensor(X_te)).numpy() * RUL_CAP).clip(0, RUL_CAP)
results.append(report("LSTM (ours)", y_te, lstm_pred))

# =========================== 9. EVALUATION PLOTS ===========================
# training curve
plt.figure(figsize=(7, 4))
plt.plot(hist["train"], label="train RMSE")
plt.plot(hist["val"], label="validation RMSE")
plt.title("LSTM training curve")
plt.xlabel("Epoch")
plt.ylabel("RMSE (cycles)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eval1_training_curve.png", dpi=150)
plt.close()

# predicted vs actual (sorted by true RUL)
order = np.argsort(y_te)
plt.figure(figsize=(9, 4.5))
plt.plot(y_te[order], "k-", lw=2, label="True RUL")
plt.plot(lstm_pred[order], "o", ms=4, color="#1f5aa6", label="LSTM prediction")
plt.plot(rf_pred[order], "x", ms=4, color="#e67e22", alpha=0.7, label="Random Forest")
plt.title("Test engines: predicted vs true RUL (sorted by true RUL)")
plt.xlabel("Test engine (sorted)")
plt.ylabel("RUL (cycles)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eval2_pred_vs_true.png", dpi=150)
plt.close()

# error distribution
err_lstm = lstm_pred - y_te
err_rf = rf_pred - y_te
plt.figure(figsize=(7, 4))
bins = np.arange(-60, 61, 5)
plt.hist(err_rf, bins=bins, alpha=0.5, color="#e67e22", label="Random Forest")
plt.hist(err_lstm, bins=bins, alpha=0.7, color="#1f5aa6", label="LSTM")
plt.axvline(0, color="k", lw=1)
plt.title("Error distribution (pred - true)\n<0 = early (safe), >0 = late (risky)")
plt.xlabel("Error (cycles)")
plt.ylabel("Number of test engines")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eval3_error_distribution.png", dpi=150)
plt.close()

# model comparison bar chart
res_df = pd.DataFrame(results)
plt.figure(figsize=(7, 4))
bars = plt.bar(res_df["model"].str.replace(": ", ":\n"), res_df["RMSE"], color=["#95a5a6", "#e67e22", "#1f5aa6"])
for b, v in zip(bars, res_df["RMSE"]):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v}", ha="center", fontweight="bold")
plt.title("Model comparison on 100 test engines (lower is better)")
plt.ylabel("RMSE (cycles)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eval4_model_comparison.png", dpi=150)
plt.close()

# full-life prediction for one validation engine (how the alert would trigger)
u = sorted(val_units)[0]
g = va_df[va_df.unit == u]
Xg, yg = make_windows(g)
with torch.no_grad():
    pg = (model(torch.tensor(Xg)).numpy() * RUL_CAP).clip(0, RUL_CAP)
cyc = g["cycle"].values[WINDOW - 1:]
plt.figure(figsize=(8, 4))
plt.plot(cyc, yg, "k-", lw=2, label="True RUL (capped)")
plt.plot(cyc, pg, color="#1f5aa6", label="LSTM prediction")
plt.axhline(ALERT_LIMIT, color="#c0392b", ls="--", label=f"Maintenance alert ({ALERT_LIMIT} cycles)")
plt.title(f"Validation engine {u}: RUL tracked over its whole life")
plt.xlabel("Cycle")
plt.ylabel("RUL (cycles)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eval5_engine_tracking.png", dpi=150)
plt.close()

# =========================== 10. BUSINESS VIEW ===========================
# Maintenance alert: an engine is "due" if true RUL <= 30 cycles.
due_true = true_test_rul <= ALERT_LIMIT
due_pred = lstm_pred <= ALERT_LIMIT
alert = {
    "engines_truly_due": int(due_true.sum()),
    "correctly_flagged": int((due_true & due_pred).sum()),
    "missed (dangerous)": int((due_true & ~due_pred).sum()),
    "false_alarms": int((~due_true & due_pred).sum()),
    "within_+-20_cycles_%": round(float(np.mean(np.abs(err_lstm) <= 20) * 100), 1),
    "late_predictions_%": round(float(np.mean(err_lstm > 0) * 100), 1),
}

print("\n================ RESULTS (test set, 100 engines) ================")
print(res_df.to_string(index=False))
print("\nMaintenance alert view (RUL <= 30 cycles):", alert)

with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump({"metrics": results, "alert": alert, "features_used": features,
               "dropped": dropped, "epochs_trained": len(hist["train"]),
               "best_val_rmse": round(best_val, 2),
               "settings": {"WINDOW": WINDOW, "RUL_CAP": RUL_CAP, "HIDDEN": HIDDEN,
                            "LR": LR, "BATCH": BATCH}}, f, indent=2)
pd.DataFrame({"engine": np.arange(1, 101), "true_RUL": true_test_rul,
              "LSTM_pred": lstm_pred.round(1), "RF_pred": rf_pred.round(1)}).to_csv(
    f"{OUT_DIR}/test_predictions.csv", index=False)
print(f"\nAll plots and results saved in '{OUT_DIR}/'")
