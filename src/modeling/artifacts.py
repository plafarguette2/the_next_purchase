from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class RecoArtifacts:
    # Retrieval
    als_model: object
    X_user_item: csr_matrix
    user2idx: dict
    idx2item: np.ndarray

    # Reranker
    reranker_model: object
    feature_cols: list[str]
    user_feats: pd.DataFrame
    item_feats: pd.DataFrame
    max_train_date: pd.Timestamp

    # Business layer
    stock_ctry: pd.DataFrame
    item_value: pd.DataFrame


def save_artifacts(artifacts: RecoArtifacts, out_dir: str | Path) -> None:
    """
    Save all necessary objects for inference into out_dir using joblib.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.als_model, out_dir / "als_model.joblib")
    joblib.dump(artifacts.X_user_item, out_dir / "X_user_item.joblib")
    joblib.dump(artifacts.user2idx, out_dir / "user2idx.joblib")
    joblib.dump(artifacts.idx2item, out_dir / "idx2item.joblib")

    joblib.dump(artifacts.reranker_model, out_dir / "reranker_model.joblib")
    joblib.dump(artifacts.feature_cols, out_dir / "feature_cols.joblib")
    joblib.dump(artifacts.user_feats, out_dir / "user_feats.joblib")
    joblib.dump(artifacts.item_feats, out_dir / "item_feats.joblib")
    joblib.dump(artifacts.max_train_date, out_dir / "max_train_date.joblib")

    joblib.dump(artifacts.stock_ctry, out_dir / "stock_ctry.joblib")
    joblib.dump(artifacts.item_value, out_dir / "item_value.joblib")


def load_artifacts(in_dir: str | Path) -> RecoArtifacts:
    """
    Load inference artifacts from in_dir.
    """
    in_dir = Path(in_dir)

    als_model = joblib.load(in_dir / "als_model.joblib")
    X_user_item = joblib.load(in_dir / "X_user_item.joblib")
    user2idx = joblib.load(in_dir / "user2idx.joblib")
    idx2item = joblib.load(in_dir / "idx2item.joblib")

    reranker_model = joblib.load(in_dir / "reranker_model.joblib")
    feature_cols = joblib.load(in_dir / "feature_cols.joblib")
    user_feats = joblib.load(in_dir / "user_feats.joblib")
    item_feats = joblib.load(in_dir / "item_feats.joblib")
    max_train_date = joblib.load(in_dir / "max_train_date.joblib")

    stock_ctry = joblib.load(in_dir / "stock_ctry.joblib")
    item_value = joblib.load(in_dir / "item_value.joblib")

    return RecoArtifacts(
        als_model=als_model,
        X_user_item=X_user_item,
        user2idx=user2idx,
        idx2item=idx2item,
        reranker_model=reranker_model,
        feature_cols=feature_cols,
        user_feats=user_feats,
        item_feats=item_feats,
        max_train_date=max_train_date,
        stock_ctry=stock_ctry,
        item_value=item_value
    )
