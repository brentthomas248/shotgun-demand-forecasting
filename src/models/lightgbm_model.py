"""LightGBM forecaster with Optuna hyperparameter optimisation.

Provides a LightGBMForecaster class following the same fit/predict interface
as the other model modules, plus a standalone ``tune_lightgbm`` function
for Bayesian optimisation via Optuna.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LightGBMConfig:
    """Hyperparameter configuration for LightGBM."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: int = 42
    n_jobs: int = -1


class LightGBMForecaster:
    """LightGBM regression forecaster for demand prediction.

    Supports early stopping on a validation set and feature importance
    extraction.

    Attributes:
        config: LightGBMConfig with hyperparameters.
        feature_names: Feature columns used during fitting.
        model_: Fitted LGBMRegressor instance.
    """

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        """Initialise the LightGBM forecaster.

        Args:
            config: Hyperparameter configuration. Uses defaults if None.
        """
        self.config = config or LightGBMConfig()
        self.feature_names: list[str] | None = None
        self.model_: Any = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        early_stopping_rounds: int = 50,
    ) -> LightGBMForecaster:
        """Fit the LightGBM model with optional early stopping.

        Args:
            X: Training feature matrix.
            y: Training target values.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.
            early_stopping_rounds: Rounds without improvement before stopping.

        Returns:
            self
        """
        import lightgbm
        from dataclasses import asdict

        params = asdict(self.config)
        params.pop("n_jobs")
        model = lightgbm.LGBMRegressor(**params, n_jobs=self.config.n_jobs, verbosity=-1)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lightgbm.early_stopping(stopping_rounds=early_stopping_rounds),
                lightgbm.log_evaluation(period=0),
            ]

        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        model.fit(X, y, **fit_kwargs)
        self.model_ = model
        logger.info("LightGBMForecaster fit on %d samples", len(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions from a feature matrix.

        Args:
            X: Feature matrix (same columns as training).

        Returns:
            Array of predicted values.
        """
        return self.model_.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance scores from the fitted model.

        Returns:
            DataFrame with ``feature`` and ``importance`` columns sorted
            descending by importance.
        """
        importances = self.model_.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
) -> tuple[LightGBMConfig, dict[str, float], LightGBMForecaster]:
    """Run Optuna Bayesian optimisation for LightGBM hyperparameters.

    Minimises MAPE on the validation set across ``n_trials`` trials,
    then retrains on the best configuration.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.
        X_val: Validation feature matrix.
        y_val: Validation target values.
        n_trials: Number of Optuna trials.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (best_config, best_metrics_dict, fitted_forecaster).
    """
    import optuna

    from src.evaluation.metrics import compute_mape

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        config = LightGBMConfig(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=random_state,
        )
        forecaster = LightGBMForecaster(config)
        forecaster.fit(X_train, y_train, X_val, y_val)
        preds = forecaster.predict(X_val)
        return compute_mape(np.asarray(y_val), preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_config = LightGBMConfig(**study.best_params, random_state=random_state)
    best_forecaster = LightGBMForecaster(best_config)
    best_forecaster.fit(X_train, y_train, X_val, y_val)
    preds = best_forecaster.predict(X_val)
    best_metrics = {"mape": compute_mape(np.asarray(y_val), preds)}

    logger.info("LightGBM tuning complete: best MAPE=%.2f%%", best_metrics["mape"])
    return best_config, best_metrics, best_forecaster
