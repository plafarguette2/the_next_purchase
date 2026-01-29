import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

def train_als(X_user_item: csr_matrix,
              factors: int = 64,
              regularization: float = 0.01,
              iterations: int = 20,
              alpha: float = 40.0):

    X_conf = (X_user_item * alpha).tocsr()

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=1
    )

    model.fit(X_conf)
    return model

def retrieve_products(model, X_user_item, user2idx, idx2item, user_id, N=200, filtered=True):

    u = user2idx.get(user_id)
    if u is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    item_idx, scores = model.recommend(
        userid=u,
        user_items=X_user_item[u],
        N=N,
        filter_already_liked_items=filtered
    )

    product_ids = idx2item[item_idx]
    return product_ids, scores
