import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

def make_time_split(transactions: pd.DataFrame, cutoff_date: pd.Timestamp):
    df = transactions.sort_values("SaleTransactionDate")
    train_tx = df.query("SaleTransactionDate < @cutoff_date")
    test_tx = df.query("SaleTransactionDate >= @cutoff_date")
    return train_tx, test_tx

def make_id_maps(clients_df: pd.DataFrame, products_df: pd.DataFrame):
    user_ids = clients_df["ClientID"].drop_duplicates().to_numpy()
    item_ids = products_df["ProductID"].drop_duplicates().to_numpy()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    idx2user = user_ids

    item2idx = {p: i for i, p in enumerate(item_ids)}
    idx2item = item_ids

    return user2idx, idx2user, item2idx, idx2item


def build_interaction_matrix(train_tx, user2idx, item2idx, a: float = 1.0, b: float = 0.5, c: float = 0.5, d: float = 0.25):

    agg = (
        train_tx
        .groupby(["ClientID", "ProductID"], as_index=False)
        .agg(
            freq_bought=("Quantity", "size"),
            n_bought=("Quantity", "sum"),
            spend=("SalesNetAmountEuro", "sum")
        )
    )

    agg["u"] = agg["ClientID"].map(user2idx)
    agg["i"] = agg["ProductID"].map(item2idx)
    agg = agg.dropna(subset=["u", "i"])
    agg["u"] = agg["u"].astype(np.int32)
    agg["i"] = agg["i"].astype(np.int32)

    score = (
        a * (agg["freq_bought"] > 0).astype(np.float32)
        + b * np.log1p(agg["freq_bought"].astype(np.float32))
        + c * np.log1p(agg["n_bought"].astype(np.float32))
        + d * np.log1p(agg["spend"].astype(np.float32))
    ).astype(np.float32)

    X = csr_matrix(
        (score, (agg["u"].to_numpy(), agg["i"].to_numpy())),
        shape=(len(user2idx), len(item2idx))
    )

    return X

