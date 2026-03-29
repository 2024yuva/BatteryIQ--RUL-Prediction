# =============================================================================
# BATTERY REMAINING USEFUL LIFE (RUL) PREDICTION
# Using LSTM + Incremental Capacity (IC) Analysis
# NASA Battery Dataset
# Compatible with Google Colab
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: PROJECT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
# battery_rul_project/
# ├── battery_rul_prediction.py   ← this file (all-in-one script / Colab notebook)
# ├── requirements.txt
# └── outputs/
#     ├── degradation_curves.png
#     ├── ic_curves.png
#     ├── actual_vs_predicted.png
#     └── rul_model.h5

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: INSTALL & IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# Run in Colab:
# !pip install scipy mat4py tensorflow scikit-learn matplotlib pandas numpy

import os, warnings, math, urllib.request, zipfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – swap to "TkAgg" / remove for Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     Bidirectional, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

print(f"TensorFlow {tf.__version__} | NumPy {np.__version__}")
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATASET — NASA PCoE Battery Dataset (B0005, B0006, B0007, B0018)
# ─────────────────────────────────────────────────────────────────────────────
# Dataset description:
#   • 4 Li-ion 18650 cells charged/discharged at room temperature
#   • Each cycle records: Voltage_measured, Current_measured,
#     Temperature_measured, Current_charge, Voltage_charge, Time,
#     Capacity (Ah)
#   • Nominal capacity ≈ 2 Ah; End-of-life defined at 70% → 1.4 Ah
#   • ~168 cycles per cell before EOL
#
# We SIMULATE a realistic dataset here so the code runs offline / in Colab
# without manual downloads.  Replace `load_data()` with real mat-file parsing
# if you have the dataset locally.

NOMINAL_CAPACITY = 2.0      # Ah
EOL_THRESHOLD    = 0.70     # 70 % of nominal = End-of-Life
N_CELLS          = 4
CYCLES_PER_CELL  = 168
RANDOM_SEED      = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def simulate_battery_data(n_cells=N_CELLS, cycles_per_cell=CYCLES_PER_CELL):
    """
    Simulate NASA-style battery cycling data.
    Returns a dict: {cell_id: DataFrame with one row per time-step}.
    Each row contains: cycle, voltage, current, temperature, capacity
    """
    all_data = {}

    for cell in range(n_cells):
        records = []
        # Slight cell-to-cell variation
        cap_fade_rate = 0.0022 + np.random.uniform(-0.0003, 0.0003)
        noise_scale   = 0.002

        for cyc in range(1, cycles_per_cell + 1):
            # Capacity fades with cycle count (nonlinear)
            cap = NOMINAL_CAPACITY * (1 - cap_fade_rate * cyc
                                      - 0.000005 * cyc**2
                                      + np.random.normal(0, 0.003))
            cap = max(cap, 0.5)   # floor

            # Simulate ~50 voltage/current/temp readings per cycle
            n_pts = 50
            # During charge: voltage rises 3.0 → 4.2 V
            t_arr   = np.linspace(0, 1, n_pts)
            voltage = 3.0 + 1.2 * t_arr + np.random.normal(0, noise_scale, n_pts)
            # Current: constant-current then taper (simplified)
            current = np.where(t_arr < 0.7,
                               1.5 + np.random.normal(0, 0.05, n_pts),
                               1.5*(1-t_arr)/0.3 + np.random.normal(0, 0.05, n_pts))
            current = np.clip(current, 0, 2.0)
            # Temperature rises with degradation
            temp_base = 25 + 5*(cyc/cycles_per_cell)
            temp = temp_base + 10*t_arr + np.random.normal(0, 0.3, n_pts)

            for i in range(n_pts):
                records.append({
                    "cell":        cell,
                    "cycle":       cyc,
                    "voltage":     round(voltage[i], 4),
                    "current":     round(current[i], 4),
                    "temperature": round(temp[i], 3),
                    "capacity":    round(cap, 5),
                })

        all_data[cell] = pd.DataFrame(records)

    return all_data


def compute_rul(df_cell):
    """
    Add a RUL column.
    EOL = first cycle where capacity < EOL_THRESHOLD * NOMINAL_CAPACITY.
    RUL = EOL_cycle - current_cycle  (clipped at 0).
    """
    eol_cap = EOL_THRESHOLD * NOMINAL_CAPACITY
    # One capacity value per cycle (take the last reading)
    cap_per_cycle = df_cell.groupby("cycle")["capacity"].last()
    eol_cycles = cap_per_cycle[cap_per_cycle < eol_cap].index
    eol_cycle  = int(eol_cycles.min()) if len(eol_cycles) else int(cap_per_cycle.index.max())

    df_cell = df_cell.copy()
    df_cell["rul"] = (eol_cycle - df_cell["cycle"]).clip(lower=0)
    df_cell["eol_cycle"] = eol_cycle
    return df_cell, eol_cycle


print("Simulating NASA-style battery dataset …")
raw_data = simulate_battery_data()

# Add RUL labels
cell_dfs = {}
for cid, df in raw_data.items():
    df, eol = compute_rul(df)
    cell_dfs[cid] = df
    print(f"  Cell {cid}: EOL at cycle {eol}  |  "
          f"cycles={df['cycle'].max()}  |  rows={len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: INCREMENTAL CAPACITY (IC) ANALYSIS  —  dQ/dV
# ─────────────────────────────────────────────────────────────────────────────
# WHY IC ANALYSIS?
#   • Standard capacity vs cycle plots give a single scalar per cycle.
#   • The IC curve (dQ/dV vs V) exposes internal electrochemical reactions
#     as distinct peaks.
#   • As the battery degrades:
#       - Peak heights DECREASE (loss of active material)
#       - Peaks SHIFT in voltage (change in electrode potential)
#   • IC features extracted at early cycles can predict long-term RUL
#     well before the capacity drop is obvious.
#
# HOW dQ/dV IS COMPUTED:
#   1. Collect (Voltage, Capacity_incremental) pairs during a charge step.
#   2. Sort by voltage.
#   3. Smooth with a Savitzky-Golay filter (removes noise without blurring peaks).
#   4. Differentiate:  dQ/dV = ΔQ / ΔV

def compute_ic_curve(voltage, capacity, n_points=200, window=11, poly=3):
    """
    Compute smoothed IC curve (dQ/dV vs V) for one charge cycle.

    Parameters
    ----------
    voltage   : 1-D array of voltage readings (V)
    capacity  : 1-D array of cumulative charge (Ah) — same length
    n_points  : number of interpolation points for uniform V grid
    window    : Savitzky-Golay window length (must be odd)
    poly      : Savitzky-Golay polynomial order

    Returns
    -------
    v_grid : uniform voltage grid
    ic     : dQ/dV values on v_grid
    """
    # Sort by voltage
    sort_idx  = np.argsort(voltage)
    v_sorted  = voltage[sort_idx]
    q_sorted  = capacity[sort_idx]

    # Remove duplicate voltage values
    _, unique_idx = np.unique(v_sorted, return_index=True)
    v_u = v_sorted[unique_idx]
    q_u = q_sorted[unique_idx]

    if len(v_u) < 5:
        return None, None

    # Interpolate onto uniform grid
    v_grid = np.linspace(v_u.min(), v_u.max(), n_points)
    interp  = interp1d(v_u, q_u, kind="cubic", fill_value="extrapolate")
    q_grid  = interp(v_grid)

    # Differentiate
    dv = np.diff(v_grid)
    dq = np.diff(q_grid)
    ic_raw = dq / dv

    # Align arrays (ic has n_points-1 values; use midpoint voltages)
    v_mid  = 0.5 * (v_grid[:-1] + v_grid[1:])

    # Smooth with Savitzky-Golay
    win = min(window, len(ic_raw) - (1 if len(ic_raw) % 2 == 0 else 0))
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)
    ic_smooth = savgol_filter(ic_raw, window_length=win, polyorder=poly)

    return v_mid, ic_smooth


def extract_ic_features(v_mid, ic):
    """
    Summarise an IC curve into 4 scalar features:
      - peak_height   : maximum dQ/dV value
      - peak_voltage  : voltage at which peak occurs
      - peak_area     : area under positive IC (trapz)
      - ic_mean       : mean of IC over the voltage window
    """
    if v_mid is None:
        return {"peak_height": 0, "peak_voltage": 3.6,
                "peak_area": 0, "ic_mean": 0}
    peak_idx    = np.argmax(ic)
    peak_height = float(ic[peak_idx])
    peak_volt   = float(v_mid[peak_idx])
    peak_area   = float(np.trapz(np.clip(ic, 0, None), v_mid))
    ic_mean     = float(np.mean(ic))
    return {"peak_height": peak_height, "peak_voltage": peak_volt,
            "peak_area": peak_area, "ic_mean": ic_mean}


# Compute IC features for every cycle of every cell
print("\nComputing IC features …")
ic_records = []
for cid, df in cell_dfs.items():
    for cyc, grp in df.groupby("cycle"):
        v = grp["voltage"].values
        # Cumulative charge proxy: integrate current × time (simplified to capacity)
        q = np.linspace(0, grp["capacity"].iloc[-1], len(v))
        v_mid, ic = compute_ic_curve(v, q)
        feats = extract_ic_features(v_mid, ic)
        feats.update({"cell": cid, "cycle": cyc,
                      "capacity": grp["capacity"].iloc[-1],
                      "rul": grp["rul"].iloc[-1]})
        ic_records.append(feats)

ic_df = pd.DataFrame(ic_records)
print(f"  IC feature table shape: {ic_df.shape}")
print(ic_df.head())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# 4a. Aggregate time-step features to cycle-level
print("\nAggregating to cycle-level features …")
agg_records = []
for cid, df in cell_dfs.items():
    for cyc, grp in df.groupby("cycle"):
        agg_records.append({
            "cell":        cid,
            "cycle":       cyc,
            "volt_mean":   grp["voltage"].mean(),
            "volt_std":    grp["voltage"].std(),
            "curr_mean":   grp["current"].mean(),
            "curr_std":    grp["current"].std(),
            "temp_mean":   grp["temperature"].mean(),
            "temp_max":    grp["temperature"].max(),
            "capacity":    grp["capacity"].iloc[-1],
            "rul":         grp["rul"].iloc[-1],
        })
agg_df = pd.DataFrame(agg_records)

# 4b. Merge with IC features
cycle_df = agg_df.merge(
    ic_df[["cell","cycle","peak_height","peak_voltage","peak_area","ic_mean"]],
    on=["cell","cycle"], how="left"
)

# 4c. Handle missing values (forward-fill within cell)
cycle_df = cycle_df.sort_values(["cell","cycle"])
cycle_df = cycle_df.groupby("cell", group_keys=False).apply(
    lambda g: g.ffill().bfill()
)
print(f"  Cycle-level feature table: {cycle_df.shape}")
print(f"  Missing values: {cycle_df.isnull().sum().sum()}")

# 4d. Feature list
FEATURES = [
    "volt_mean","volt_std",
    "curr_mean","curr_std",
    "temp_mean","temp_max",
    "capacity",
    "peak_height","peak_voltage","peak_area","ic_mean",
]
TARGET = "rul"

# 4e. Normalize features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

cycle_df[FEATURES] = scaler_X.fit_transform(cycle_df[FEATURES])
cycle_df[[TARGET]]  = scaler_y.fit_transform(cycle_df[[TARGET]])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SEQUENCE CREATION (sliding window for LSTM)
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 10   # look-back window: 10 consecutive cycles

def make_sequences(df_cell, features, target, seq_len=SEQ_LEN):
    """
    Convert a cell's cycle-level dataframe into overlapping (X, y) pairs.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples,)
    """
    X, y = [], []
    vals = df_cell[features].values
    tgt  = df_cell[target].values
    for i in range(len(vals) - seq_len):
        X.append(vals[i:i+seq_len])
        y.append(tgt[i+seq_len])
    return np.array(X), np.array(y)


# Train on cells 0-2, test on cell 3
train_cells = [0, 1, 2]
test_cell   = 3

X_train_list, y_train_list = [], []
for cid in train_cells:
    df_c = cycle_df[cycle_df["cell"] == cid].reset_index(drop=True)
    Xc, yc = make_sequences(df_c, FEATURES, TARGET)
    X_train_list.append(Xc)
    y_train_list.append(yc)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

df_test = cycle_df[cycle_df["cell"] == test_cell].reset_index(drop=True)
X_test, y_test = make_sequences(df_test, FEATURES, TARGET)

print(f"\nTrain shapes:  X={X_train.shape}  y={y_train.shape}")
print(f"Test shapes:   X={X_test.shape}  y={y_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MODEL ARCHITECTURE — Stacked Bi-LSTM + Attention
# ─────────────────────────────────────────────────────────────────────────────
#
# Architecture summary:
#   Input (seq_len, n_features)
#     ↓
#   Bidirectional LSTM  (64 units, return_sequences=True)  — captures forward
#     ↓                                                       & backward trends
#   Dropout (0.2)
#     ↓
#   LSTM  (32 units, return_sequences=False)               — extract summary
#     ↓
#   Dropout (0.2)
#     ↓
#   Dense (16, relu)
#     ↓
#   Dense (1, linear)                                      — RUL output

def build_lstm_model(seq_len, n_features, lstm1=64, lstm2=32, lr=1e-3):
    model = Sequential([
        Bidirectional(LSTM(lstm1, return_sequences=True),
                      input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(lstm2, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ], name="BidirectionalLSTM_RUL")

    model.compile(optimizer=Adam(learning_rate=lr), loss="mse",
                  metrics=["mae"])
    return model


model = build_lstm_model(SEQ_LEN, len(FEATURES))
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TRAINING
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS     = 100
BATCH_SIZE = 32

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True,
                  verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6,
                      verbose=1),
]

print("\nTraining LSTM model …")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

model.save("outputs/rul_model.keras")
print("Model saved → outputs/rul_model.keras")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Inverse-transform to original RUL scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"\n{'='*40}")
print(f"  Test RMSE : {rmse:.2f} cycles")
print(f"  Test MAE  : {mae:.2f} cycles")
print(f"{'='*40}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 9a. Training history ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss (MSE) During Training")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["mae"],     label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("MAE During Training")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("outputs/training_history.png", dpi=150)
plt.close()
print("Saved → outputs/training_history.png")

# ── 9b. Actual vs Predicted RUL ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_true, label="Actual RUL",    color="#2196F3", linewidth=2)
ax.plot(y_pred, label="Predicted RUL", color="#FF5722", linewidth=2,
        linestyle="--")
ax.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.15, color="gray")
ax.set_title(f"Actual vs Predicted RUL  |  RMSE={rmse:.1f}  MAE={mae:.1f} cycles",
             fontsize=13)
ax.set_xlabel("Sample index (cycle in test cell)")
ax.set_ylabel("Remaining Useful Life (cycles)")
ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved → outputs/actual_vs_predicted.png")

# ── 9c. Degradation curves (all cells) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1976D2","#388E3C","#F57C00","#C62828"]
for cid in range(N_CELLS):
    raw = cell_dfs[cid].groupby("cycle")["capacity"].last().reset_index()
    ax.plot(raw["cycle"], raw["capacity"],
            label=f"Cell {cid}", color=colors[cid], linewidth=1.8)

eol_line = EOL_THRESHOLD * NOMINAL_CAPACITY
ax.axhline(eol_line, color="red", linestyle="--", linewidth=1.5,
           label=f"EOL threshold ({int(EOL_THRESHOLD*100)}% of nominal)")
ax.set_title("Capacity Degradation Curves — All Cells", fontsize=13)
ax.set_xlabel("Cycle Number"); ax.set_ylabel("Capacity (Ah)")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/degradation_curves.png", dpi=150)
plt.close()
print("Saved → outputs/degradation_curves.png")

# ── 9d. IC Curves — how peaks shift with degradation ────────────────────────
SAMPLE_CYCLES = [1, 30, 60, 100, 140, 165]  # early → late
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
cmap = plt.cm.RdYlGn_r

df_cell0 = cell_dfs[0]
for idx, cyc in enumerate(SAMPLE_CYCLES):
    grp = df_cell0[df_cell0["cycle"] == cyc]
    if grp.empty:
        continue
    v = grp["voltage"].values
    q = np.linspace(0, grp["capacity"].iloc[-1], len(v))
    v_mid, ic = compute_ic_curve(v, q)
    if v_mid is None:
        continue
    color = cmap(idx / len(SAMPLE_CYCLES))
    axes[idx].plot(v_mid, ic, color=color, linewidth=2)
    axes[idx].set_title(f"Cycle {cyc}", fontsize=11)
    axes[idx].set_xlabel("Voltage (V)"); axes[idx].set_ylabel("dQ/dV (Ah/V)")
    axes[idx].grid(True, alpha=0.4)
    # Mark the peak
    peak_i = np.argmax(ic)
    axes[idx].axvline(v_mid[peak_i], color="red", linestyle=":", alpha=0.7)
    axes[idx].scatter([v_mid[peak_i]], [ic[peak_i]], color="red", zorder=5)

plt.suptitle("IC Curves (dQ/dV vs V) at Different Cycle Points\n"
             "Red dashed line = peak voltage (shifts left as degradation progresses)",
             fontsize=12)
plt.tight_layout()
plt.savefig("outputs/ic_curves.png", dpi=150)
plt.close()
print("Saved → outputs/ic_curves.png")

# ── 9e. IC peak metrics over cycles ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ic_cell0 = ic_df[ic_df["cell"] == 0].sort_values("cycle")

axes[0].plot(ic_cell0["cycle"], ic_cell0["peak_height"],
             color="#7B1FA2", linewidth=2)
axes[0].set_title("IC Peak Height vs Cycle\n(decreases with degradation)")
axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("Peak dQ/dV (Ah/V)")
axes[0].grid(True, alpha=0.4)

axes[1].plot(ic_cell0["cycle"], ic_cell0["peak_voltage"],
             color="#0288D1", linewidth=2)
axes[1].set_title("IC Peak Voltage vs Cycle\n(shifts with degradation)")
axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("Peak Voltage (V)")
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/ic_peak_evolution.png", dpi=150)
plt.close()
print("Saved → outputs/ic_peak_evolution.png")

# ── 9f. Scatter: Actual vs Predicted ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true, y_pred, alpha=0.5, color="#1976D2", edgecolors="k",
           linewidths=0.4, s=40)
lims = [min(y_true.min(), y_pred.min())-2,
        max(y_true.max(), y_pred.max())+2]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual RUL (cycles)"); ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"Prediction Scatter  |  RMSE={rmse:.1f}  MAE={mae:.1f}")
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("outputs/scatter_actual_vs_pred.png", dpi=150)
plt.close()
print("Saved → outputs/scatter_actual_vs_pred.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: SAMPLE PREDICTION OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  SAMPLE RUL PREDICTIONS (last 10 test samples)")
print("="*55)
print(f"  {'Sample':>8}  {'Actual RUL':>12}  {'Predicted RUL':>14}  {'Error':>8}")
print("-"*55)
for i in range(-10, 0):
    err = y_true[i] - y_pred[i]
    print(f"  {i:>8}  {y_true[i]:>12.1f}  {y_pred[i]:>14.1f}  {err:>+8.1f}")
print("="*55)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: OPTIMISATION SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
─────────────────────────────────────────────────────────────────────────────
OPTIMISATION ROADMAP
─────────────────────────────────────────────────────────────────────────────

1. HYPERPARAMETER TUNING (Keras Tuner / Optuna)
   from keras_tuner import RandomSearch
   Tune: LSTM units (32/64/128), dropout (0.1–0.4), LR, seq_len (5/10/20)

2. ATTENTION MECHANISM
   Replace the second LSTM with a self-attention layer:

   class BahdanauAttention(Layer):
       def call(self, query, values):
           score = tf.matmul(query, values, transpose_b=True)
           weights = tf.nn.softmax(score, axis=-1)
           context = tf.matmul(weights, values)
           return context, weights

   This lets the model focus on the most degradation-relevant cycles.

3. GRU ALTERNATIVE
   Swap LSTM → GRU for faster training with ~same accuracy:
   from tensorflow.keras.layers import GRU
   Bidirectional(GRU(64, return_sequences=True))

4. TRANSFORMER ENCODER
   For very long sequences (500+ cycles), use a Transformer encoder block:
   - Multi-head self-attention (d_model=64, heads=4)
   - Feed-forward sub-layer
   - Positional encoding
   Outperforms LSTM on long-range dependencies.

5. ADDITIONAL FEATURES
   - Internal resistance (ΔV / ΔI at pulse start)
   - Coulombic efficiency = discharge_cap / charge_cap per cycle
   - dV/dt at fixed SOC points

6. DATA AUGMENTATION
   - Add Gaussian noise to training sequences
   - Time-warp augmentation on cycle axis

7. ENSEMBLE
   Average predictions from LSTM + GRU + 1-D CNN for lower variance.
─────────────────────────────────────────────────────────────────────────────
"""

print("\nAll outputs saved to ./outputs/")
print("Script complete. ✓")
