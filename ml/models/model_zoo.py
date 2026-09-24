"""The model comparison zoo for the stock-risk / diminishing-stock classifier.

Six families, chosen to give a real spread for the comparison report: a naive
baseline (majority class - the floor every other model must beat), a linear
model, a single interpretable tree, two ensembles, and a small neural net.
XGBoost/LightGBM are intentionally not used - they are not installable in the
authoring sandbox (see docs/STATUS.md) and HistGradientBoostingClassifier is
scikit-learn's in-box equivalent (same histogram-binning algorithm family).
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model(name: str, seed: int = 42):
    registry = {
        "baseline_majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=seed),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                         random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08,
                                                                         max_iter=200, random_state=seed),
        "mlp_neural_net": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True,
                                                random_state=seed),
    }
    if name not in registry:
        raise KeyError(f"unknown model '{name}'; choices: {sorted(registry)}")
    return registry[name]()


MODEL_NAMES = ["baseline_majority", "logistic_regression", "decision_tree", "random_forest",
              "hist_gradient_boosting", "mlp_neural_net"]
TREE_MODELS = {"decision_tree", "random_forest", "hist_gradient_boosting"}   # SHAP TreeExplainer-eligible
