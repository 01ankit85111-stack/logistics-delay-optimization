"""
Week 4 Task - Predictive Modeling
------------------------------------
Trains and compares 4 regression models to forecast shipment delay (hours):
    1. Linear Regression        (interpretable baseline)
    2. Decision Tree Regressor  (captures non-linearity, single tree)
    3. Random Forest Regressor  (bagging ensemble)
    4. Gradient Boosting Reg.   (boosting ensemble)

Evaluation: 80/20 train-test split + 5-fold cross-validation on the training
set, using MAE, RMSE, and R^2. The best model is then hyperparameter-tuned
with GridSearchCV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#2E5EAA", "#E8871E", "#4CA64C", "#C0392B"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUT = BASE_DIR / "model_charts"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Load data & split
# ---------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "model_dataset.csv")
X = df.drop(columns=["delay_hrs"])
y = df["delay_hrs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 2. Define candidate models
#    Linear Regression is wrapped with scaling since it is scale-sensitive;
#    tree-based models are scale-invariant so are used directly.
# ---------------------------------------------------------------
models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

results = []
test_predictions = {}

for name, model in models.items():
    # 5-fold cross-validation on the training set
    cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_mae = -cv_res["test_MAE"].mean()
    cv_rmse = -cv_res["test_RMSE"].mean()
    cv_r2 = cv_res["test_R2"].mean()

    # Fit on full training set, evaluate on held-out test set
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_predictions[name] = preds

    test_mae = mean_absolute_error(y_test, preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    test_r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "CV_MAE": round(cv_mae, 3), "CV_RMSE": round(cv_rmse, 3), "CV_R2": round(cv_r2, 3),
        "Test_MAE": round(test_mae, 3), "Test_RMSE": round(test_rmse, 3), "Test_R2": round(test_r2, 3),
    })

results_df = pd.DataFrame(results).sort_values("Test_RMSE")
print("\n" + "="*80)
print("MODEL COMPARISON (5-fold CV on train set, final eval on held-out test set)")
print("="*80)
print(results_df.to_string(index=False))
results_df.to_csv(DATA_DIR / "model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model by Test RMSE: {best_model_name}")

# ---------------------------------------------------------------
# 3. Hyperparameter tuning (GridSearchCV) on the best ensemble model
# ---------------------------------------------------------------
print("\n" + "="*80)
print("HYPERPARAMETER TUNING - Random Forest (GridSearchCV, 5-fold)")
print("="*80)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 3, 5],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print(f"Best CV RMSE: {-grid.best_score_:.3f}")

tuned_rf = grid.best_estimator_
tuned_preds = tuned_rf.predict(X_test)
tuned_mae = mean_absolute_error(y_test, tuned_preds)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_preds))
tuned_r2 = r2_score(y_test, tuned_preds)
print(f"Tuned Random Forest -> Test MAE: {tuned_mae:.3f} | Test RMSE: {tuned_rmse:.3f} | Test R2: {tuned_r2:.3f}")

tuning_summary = {
    "best_params": grid.best_params_,
    "best_cv_rmse": round(-grid.best_score_, 3),
    "tuned_test_mae": round(tuned_mae, 3),
    "tuned_test_rmse": round(tuned_rmse, 3),
    "tuned_test_r2": round(tuned_r2, 3),
}
with open(DATA_DIR / "tuning_summary.json", "w") as f:
    json.dump(tuning_summary, f, indent=2)

# ---------------------------------------------------------------
# 4. CHARTS
# ---------------------------------------------------------------

# Chart A: Model comparison bar chart (Test RMSE & MAE)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df["Test_MAE"], width, label="Test MAE", color=PALETTE[0])
ax.bar(x + width/2, results_df["Test_RMSE"], width, label="Test RMSE", color=PALETTE[3])
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=15)
ax.set_ylabel("Error (hours)")
ax.set_title("Model Comparison — Test MAE & RMSE (lower is better)", fontsize=13, weight="bold")
ax.legend()
for i, (mae, rmse) in enumerate(zip(results_df["Test_MAE"], results_df["Test_RMSE"])):
    ax.text(i - width/2, mae + 0.05, f"{mae:.2f}", ha="center", fontsize=9)
    ax.text(i + width/2, rmse + 0.05, f"{rmse:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/01_model_comparison_bar.png", dpi=150)
plt.show()

# Chart B: Actual vs Predicted scatter for the tuned Random Forest
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test, tuned_preds, alpha=0.5, color=PALETTE[0], s=35)
lims = [min(y_test.min(), tuned_preds.min()), max(y_test.max(), tuned_preds.max())]
ax.plot(lims, lims, color=PALETTE[3], linestyle="--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Delay (hours)")
ax.set_ylabel("Predicted Delay (hours)")
ax.set_title("Actual vs. Predicted Delay — Tuned Random Forest", fontsize=13, weight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_actual_vs_predicted.png", dpi=150)
plt.show()

# Chart C: Residual plot
residuals = y_test.values - tuned_preds
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tuned_preds, residuals, alpha=0.5, color=PALETTE[0], s=35)
ax.axhline(0, color=PALETTE[3], linestyle="--", linewidth=2)
ax.set_xlabel("Predicted Delay (hours)")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title("Residual Plot — Tuned Random Forest", fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/03_residual_plot.png", dpi=150)
plt.show()

# Chart D: Feature importance (tuned Random Forest)
importances = pd.Series(tuned_rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(importances.index, importances.values, color=PALETTE[0])
for b in bars:
    ax.text(b.get_width() + 0.003, b.get_y() + b.get_height()/2, f"{b.get_width():.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance — Tuned Random Forest", fontsize=13, weight="bold")
ax.set_xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150)
plt.show()

# Chart E: CV RMSE distribution across folds per model (boxplot)
cv_records = []
for name, model in models.items():
    cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    for v in -cv_res["test_RMSE"]:
        cv_records.append({"Model": name, "Fold RMSE": v})
cv_df = pd.DataFrame(cv_records)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=cv_df, x="Model", y="Fold RMSE", hue="Model", palette=PALETTE, legend=False, ax=ax)
sns.stripplot(data=cv_df, x="Model", y="Fold RMSE", color="black", size=5, alpha=0.6, ax=ax)
ax.set_title("5-Fold Cross-Validation RMSE by Model", fontsize=13, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("RMSE (hours)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT}/05_cv_rmse_boxplot.png", dpi=150)
plt.show()

print("\nAll model evaluation charts saved to", OUT)
