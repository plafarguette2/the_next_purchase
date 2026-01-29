import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from src.modeling.reranker_data import *

@dataclass
class CalibratedReranker:
    base_model: object
    calibrator: IsotonicRegression

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # base predicted prob
        p = self.base_model.predict_proba(X)[:, 1]
        # isotonic calibrated prob
        p_cal = self.calibrator.transform(p)
        # return in sklearn-style 2-col format
        return np.vstack([1 - p_cal, p_cal]).T



def train_binary_reranker(training_df: pd.DataFrame,
                          feature_cols: list[str],
                          test_size: float = 0.2,
                          calib_size: float = 0.2,
                          random_state: int = 1):
    """
    Train base classifier + isotonic calibration.
    Returns CalibratedReranker + metrics on the final holdout set.
    """
    X = training_df[feature_cols]
    y = training_df["y"].astype(int)

    # split: train vs temp (cal+val)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # split temp into calibration vs validation
    X_cal, X_va, y_cal, y_va = train_test_split(
        X_tmp, y_tmp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_tmp
    )

    base = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=6,
        max_iter=200,
        random_state=random_state
    )
    base.fit(X_tr, y_tr)

    # fit isotonic on calibration set using base predicted probs
    p_cal_raw = base.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal)

    reranker = CalibratedReranker(base_model=base, calibrator=iso)

    # evaluate on validation set with calibrated probabilities
    p_va = reranker.predict_proba(X_va)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_va, p_va)) if len(np.unique(y_va)) > 1 else np.nan,
        "avg_precision": float(average_precision_score(y_va, p_va)) if len(np.unique(y_va)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_va, p_va)) if len(np.unique(y_va)) > 1 else np.nan,
        "n_train": int(len(X_tr)),
        "n_cal": int(len(X_cal)),
        "n_val": int(len(X_va)),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_cal": float(y_cal.mean()),
        "pos_rate_val": float(y_va.mean()),
    }

    return reranker, metrics
