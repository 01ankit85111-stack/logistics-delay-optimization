# Logistics Delay Optimization

Predictive logistics analytics project for forecasting shipment delays and optimizing transport-mode allocation. The workflow engineers operational features, compares regression models, tunes a Random Forest, and solves a linear program to reduce monthly transportation cost while meeting a delay SLA.

## Project Workflow

1. `04_feature_engineering.py` creates time-based and cyclical features, builds a feature-driven `delay_hrs` target, and writes `model_dataset.csv`.
2. `05_model_training.py` trains Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models. It evaluates them with five-fold cross-validation and a held-out test set, then tunes the Random Forest.
3. `06_optimization.py` compares the current transport-mode mix with a cost-minimizing allocation subject to mode capacities and an average-delay SLA.

## Files

- `model_dataset.csv`: engineered model features and delay target.
- `model_comparison.csv`: cross-validation and test metrics for each model.
- `tuning_summary.json`: tuned Random Forest parameters and metrics.
- `optimization_result.csv`: current and optimized mode allocations.
- `model_charts/`: model evaluation and optimization charts.

## Requirements

Python 3.10 or newer is recommended. Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Run

Run the scripts from the repository directory:

```bash
python 04_feature_engineering.py
python 05_model_training.py
python 06_optimization.py
```

The scripts save CSV, JSON, and PNG outputs in the repository directory and `model_charts/`.

## Current Results

Using the included engineered dataset, Linear Regression has the lowest test RMSE among the four comparison models at approximately 3.125 hours. The optimization scenario reduces estimated monthly cost from about Rs 441,755 to Rs 334,202 while keeping average delay at the current SLA target of approximately 8.59 hours.

## Data Note

The original raw `logistics_dataset.csv` is not included in this workspace. When it is unavailable, `06_optimization.py` uses `model_dataset.csv` and estimates mode cost from distance. These fallback costs are planning estimates, not observed invoices.