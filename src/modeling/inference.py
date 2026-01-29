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
from src.modeling.reranker_model import *
from src.modeling.retrieval_als import *


# ---------------------------
# Module 5: Reranker inference (features + scoring)
# ---------------------------

def prepare_reranker_artifacts(train_tx: pd.DataFrame):
    """
    Precompute artifacts needed at inference time so you don't recompute per user.
    Returns: user_feats, item_feats, max_train_date
    """
    user_feats, item_feats = compute_aggregated_features(train_tx)
    max_train_date = train_tx["SaleTransactionDate"].max()
    return user_feats, item_feats, max_train_date


def build_features_for_candidates(candidates_df: pd.DataFrame,
                                  user_feats: pd.DataFrame,
                                  item_feats: pd.DataFrame,
                                  max_train_date: pd.Timestamp,
                                  feature_cols: list[str]) -> pd.DataFrame:
    """
    Take candidate rows (ClientID, ProductID, als_score) and return a feature DF
    with all columns in feature_cols (same schema as training).
    """
    df = (
        candidates_df
        .merge(user_feats, on="ClientID", how="left")
        .merge(item_feats, on="ProductID", how="left")
        .assign(
            user_recency_days=lambda x: (max_train_date - x["user_last_date"]).dt.days.astype(np.float32),

            user_txn_cnt=lambda x: x["user_txn_cnt"].fillna(0).astype(np.float32),
            user_unique_items=lambda x: x["user_unique_items"].fillna(0).astype(np.float32),
            user_qty_sum=lambda x: x["user_qty_sum"].fillna(0).astype(np.float32),
            user_spend_sum=lambda x: x["user_spend_sum"].fillna(0).astype(np.float32),

            item_txn_cnt=lambda x: x["item_txn_cnt"].fillna(0).astype(np.float32),
            item_unique_users=lambda x: x["item_unique_users"].fillna(0).astype(np.float32),
            item_qty_sum=lambda x: x["item_qty_sum"].fillna(0).astype(np.float32),
            item_spend_sum=lambda x: x["item_spend_sum"].fillna(0).astype(np.float32),
            item_avg_price=lambda x: x["item_avg_price"].fillna(0).astype(np.float32),

            als_score=lambda x: x["als_score"].fillna(0).astype(np.float32),
        )
    )

    # ensure all needed columns exist (safety)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns at inference: {missing}")

    return df


def score_candidates_with_reranker(reranker_model,
                                  candidates_df: pd.DataFrame,
                                  user_feats: pd.DataFrame,
                                  item_feats: pd.DataFrame,
                                  max_train_date: pd.Timestamp,
                                  feature_cols: list[str]) -> pd.DataFrame:
    """
    Adds calibrated (or uncalibrated) purchase propensity to candidate rows.
    Output granularity: 1 row = (ClientID, ProductID) candidate.
    """
    if candidates_df.empty:
        return candidates_df.assign(p_buy=np.array([], dtype=np.float32))

    feats_df = build_features_for_candidates(
        candidates_df=candidates_df,
        user_feats=user_feats,
        item_feats=item_feats,
        max_train_date=max_train_date,
        feature_cols=feature_cols
    )

    X_feat = feats_df[feature_cols]

    # Works for both sklearn base models and our calibrated wrapper (both expose predict_proba)
    p = reranker_model.predict_proba(X_feat)[:, 1].astype(np.float32)

    out = feats_df.assign(p_buy=p)
    return out


def infer_for_user(model_als,
                   X_user_item: csr_matrix,
                   user2idx: dict,
                   idx2item: np.ndarray,
                   reranker_model,
                   user_feats: pd.DataFrame,
                   item_feats: pd.DataFrame,
                   max_train_date: pd.Timestamp,
                   feature_cols: list[str],
                   user_id: int,
                   N: int = 200,
                   filtered: bool = True) -> pd.DataFrame:
    """
    End-to-end inference for ONE user:
      1) ALS retrieve Top-N candidates
      2) Build features
      3) Predict p_buy
      4) Return ranked candidates
    """
    product_ids, als_scores = retrieve_products(
        model=model_als,
        X_user_item=X_user_item,
        user2idx=user2idx,
        idx2item=idx2item,
        user_id=user_id,
        N=N,
        filtered=filtered
    )

    if len(product_ids) == 0:
        return pd.DataFrame(columns=["ClientID", "ProductID", "als_score", "p_buy"])

    candidates_df = pd.DataFrame({
        "ClientID": int(user_id),
        "ProductID": product_ids.astype(np.int64),
        "als_score": als_scores.astype(np.float32),
    })

    scored = score_candidates_with_reranker(
        reranker_model=reranker_model,
        candidates_df=candidates_df,
        user_feats=user_feats,
        item_feats=item_feats,
        max_train_date=max_train_date,
        feature_cols=feature_cols
    )

    # rank by purchase propensity (you'll later rerank by business score)
    scored = scored.sort_values(["p_buy", "als_score"], ascending=False).reset_index(drop=True)

    return scored[["ClientID", "ProductID", "als_score", "p_buy"]]
