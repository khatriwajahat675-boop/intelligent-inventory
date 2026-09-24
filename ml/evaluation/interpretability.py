"""Model interpretability: SHAP when available, permutation importance otherwise.

`shap` could not be installed in the authoring sandbox (blocked package
index - see docs/STATUS.md), so this module tries it first and transparently
falls back to scikit-learn's permutation importance (a model-agnostic,
already-installed alternative that answers the same question - "how much
does each feature move the prediction?" - by shuffling it and measuring the
drop in a scoring metric) plus partial dependence for the top features. The
fallback is what actually produced reports/model_comparison/*_importance.*.
Re-running with `pip install shap` on a machine with a working package index
gets real Shapley-value attributions with zero code changes elsewhere -
`explain()` returns the same shape of result either way.
"""
from __future__ import annotations

import numpy as np
from sklearn.inspection import partial_dependence, permutation_importance


def _shap_values(clf, X_background: np.ndarray, X_explain: np.ndarray, model_name: str):
    import shap  # ImportError here is caught by the caller
    if model_name in ("decision_tree", "random_forest", "hist_gradient_boosting"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_explain)
        return sv[1] if isinstance(sv, list) else sv          # binary classifiers: take the positive class
    background = shap.sample(X_background, min(100, len(X_background)))
    explainer = shap.KernelExplainer(lambda x: clf.predict_proba(x)[:, 1], background)
    return explainer.shap_values(X_explain, nsamples=100)


def explain(clf, X_train: np.ndarray, X_explain: np.ndarray, y_explain: np.ndarray, feature_names: list[str],
           model_name: str, sample_n: int = 300, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_explain), size=min(sample_n, len(X_explain)), replace=False)
    Xs = X_explain[idx]

    try:
        shap_vals = _shap_values(clf, X_train, Xs, model_name)
        importance = np.abs(shap_vals).mean(axis=0)
        method = "shap"
        per_feature_signed_mean = shap_vals.mean(axis=0)
    except ImportError:
        pi = permutation_importance(clf, X_explain[idx], y_explain[idx], scoring="roc_auc",
                                    n_repeats=8, random_state=seed, n_jobs=-1)
        importance = pi.importances_mean
        method = "permutation_importance (shap not installed - see module docstring)"
        per_feature_signed_mean = None

    order = np.argsort(importance)[::-1]
    top = order[: min(12, len(order))]
    result = {
        "method": method, "sample_size": int(len(idx)),
        "ranked_features": [{"feature": feature_names[i], "importance": round(float(importance[i]), 5),
                             "signed_mean_effect": (round(float(per_feature_signed_mean[i]), 5)
                                                    if per_feature_signed_mean is not None else None)}
                            for i in order],
    }
    if method == "shap":
        result["shap_values_top_features"] = shap_vals[:, top].tolist()
        result["shap_feature_names_top"] = [feature_names[i] for i in top]
    else:
        # 1-D partial dependence for the top 4 features - the fallback's "which direction" signal
        pdp = []
        for i in top[:4]:
            try:
                pd_res = partial_dependence(clf, X_explain[idx], [int(i)], kind="average", grid_resolution=15)
                pdp.append({"feature": feature_names[i], "grid": pd_res["grid_values"][0].round(4).tolist(),
                           "average_prediction": pd_res["average"][0].round(4).tolist()})
            except Exception:
                continue
        result["partial_dependence_top4"] = pdp
    return result
