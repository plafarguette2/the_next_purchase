import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

def build_positives(test_tx: pd.DataFrame,
                    min_date: pd.Timestamp | None = None,
                    max_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Positives = (ClientID, ProductID) pairs that occur in the test window.
    """
    df = test_tx.copy()

    if min_date is not None:
        df = df[df["SaleTransactionDate"] >= min_date]
    if max_date is not None:
        df = df[df["SaleTransactionDate"] < max_date]

    positives = (
        df[["ClientID", "ProductID"]]
        .drop_duplicates()
        .assign(y=1)
    )
    return positives


def retrieve_candidates_df(model,
                           X_user_item: csr_matrix,
                           user_ids: np.ndarray,
                           user2idx: dict,
                           idx2item: np.ndarray,
                           N: int = 200,
                           filtered: bool = True) -> pd.DataFrame:
    """
    Retrieval layer output as a DataFrame: (ClientID, ProductID, als_score).
    Loops over users; good enough for hackathon scale (you can subsample users if needed).
    """
    rows = []

    for user_id in user_ids:
        u = user2idx.get(int(user_id))
        if u is None:
            continue

        item_idx, scores = model.recommend(
            userid=u,
            user_items=X_user_item[u],
            N=N,
            filter_already_liked_items=filtered
        )

        if len(item_idx) == 0:
            continue

        rows.append(
            pd.DataFrame({
                "ClientID": int(user_id),
                "ProductID": idx2item[item_idx].astype(np.int64),
                "als_score": scores.astype(np.float32),
            })
        )

    if not rows:
        return pd.DataFrame(columns=["ClientID", "ProductID", "als_score"])

    return pd.concat(rows, ignore_index=True)


def compute_aggregated_features(train_tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute simple user and item aggregates from TRAIN ONLY (no leakage).
    Returns: user_feats, item_feats
    """
    user_feats = (
        train_tx
        .groupby("ClientID", as_index=False)
        .agg(
            user_txn_cnt=("ProductID", "size"),
            user_unique_items=("ProductID", "nunique"),
            user_qty_sum=("Quantity", "sum"),
            user_spend_sum=("SalesNetAmountEuro", "sum"),
            user_last_date=("SaleTransactionDate", "max"),
        )
    )

    item_feats = (
        train_tx
        .groupby("ProductID", as_index=False)
        .agg(
            item_txn_cnt=("ClientID", "size"),
            item_unique_users=("ClientID", "nunique"),
            item_qty_sum=("Quantity", "sum"),
            item_spend_sum=("SalesNetAmountEuro", "sum"),
        )
        .assign(
            item_avg_price=lambda x: (x["item_spend_sum"] / x["item_qty_sum"].clip(lower=1)).astype(np.float32)
        )
    )

    return user_feats, item_feats


def sample_negatives(candidates_labeled: pd.DataFrame,
                     n_neg_per_pos: int = 5,
                     random_state: int = 1) -> pd.DataFrame:
    """
    For each user:
      - keep all positives in candidate pool
      - sample up to n_neg_per_pos * #positives negatives from candidate pool
    Guarantees output has ClientID column.
    """
    rng = np.random.default_rng(random_state)

    # split
    pos = candidates_labeled[candidates_labeled["y"] == 1].copy()
    neg = candidates_labeled[candidates_labeled["y"] == 0].copy()

    if pos.empty:
        return pd.DataFrame(columns=candidates_labeled.columns)

    # how many negatives per user?
    pos_cnt = (
        pos.groupby("ClientID")
           .size()
           .rename("pos_cnt")
           .reset_index()
    )

    # keep only users who have positives, and attach desired negative sample sizes
    neg = neg.merge(pos_cnt, on="ClientID", how="inner")

    # sample negatives per user
    sampled_neg_parts = []
    for client_id, g in neg.groupby("ClientID", sort=False):
        k = int(g["pos_cnt"].iloc[0] * n_neg_per_pos)
        k = min(k, len(g))
        if k <= 0:
            continue
        sampled_neg_parts.append(
            g.sample(n=k, replace=False, random_state=int(rng.integers(0, 1_000_000_000)))
             .drop(columns=["pos_cnt"])
        )

    sampled_neg = (
        pd.concat(sampled_neg_parts, ignore_index=True)
        if sampled_neg_parts else
        pd.DataFrame(columns=candidates_labeled.columns)
    )

    # combine pos + sampled neg
    sampled = pd.concat([pos, sampled_neg], ignore_index=True)

    # ensure schema
    sampled = sampled[["ClientID", "ProductID", "als_score", "y"]]
    return sampled


def build_reranker_training_set(train_tx: pd.DataFrame,
                                test_tx: pd.DataFrame,
                                model,
                                X_user_item: csr_matrix,
                                user2idx: dict,
                                idx2item: np.ndarray,
                                N_candidates: int = 200,
                                n_neg_per_pos: int = 5,
                                filtered: bool = True,
                                random_state: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """
    End-to-end: candidates -> labels -> negative sampling -> feature join.
    Returns:
      - training_df with y + features
      - feature_cols
    """
    # 1) labels from test
    positives = build_positives(test_tx)
    print("positives cols:", positives.columns.tolist(), "len:", len(positives))


    # 2) candidate pool for users who appear in positives (most relevant for next-purchase)
    user_ids = positives["ClientID"].drop_duplicates().to_numpy()
    candidates = retrieve_candidates_df(
        model=model,
        X_user_item=X_user_item,
        user_ids=user_ids,
        user2idx=user2idx,
        idx2item=idx2item,
        N=N_candidates,
        filtered=filtered
    )
    print("candidates cols:", candidates.columns.tolist(), "len:", len(candidates))


    if candidates.empty:
        return pd.DataFrame(), []

    # 3) label candidates
    candidates_labeled = (
        candidates
        .merge(
            positives[["ClientID", "ProductID", "y"]],
            on=["ClientID", "ProductID"],
            how="left"
        )
        .assign(y=lambda x: x["y"].fillna(0).astype(np.int8))
    )
    print("candidates_labeled cols:", candidates_labeled.columns.tolist(), "len:", len(candidates_labeled))


    # 4) negative sampling
    sampled = sample_negatives(
        candidates_labeled=candidates_labeled,
        n_neg_per_pos=n_neg_per_pos,
        random_state=random_state
    )

    print("sampled cols:", sampled.columns.tolist())
    print("sampled index names:", sampled.index.names)


    if sampled.empty:
        return pd.DataFrame(), []

    # 5) add simple TRAIN-only aggregates (no leakage)
    user_feats, item_feats = compute_aggregated_features(train_tx)
    max_train_date = train_tx["SaleTransactionDate"].max()

    df = (
        sampled
        .merge(user_feats, on="ClientID", how="left")
        .merge(item_feats, on="ProductID", how="left")
        .assign(
            user_recency_days=lambda x: (max_train_date - x["user_last_date"]).dt.days.astype(np.float32),
            # simple stabilizers
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

    feature_cols = [
        "als_score",
        "user_txn_cnt", "user_unique_items", "user_qty_sum", "user_spend_sum", "user_recency_days",
        "item_txn_cnt", "item_unique_users", "item_qty_sum", "item_spend_sum", "item_avg_price",
    ]

    df = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    return df, feature_cols


