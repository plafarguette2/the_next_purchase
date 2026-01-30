# src/modeling/pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .artifacts import RecoArtifacts
from .retrieval_als import retrieve_products
from .inference import score_candidates_with_reranker
from .business import apply_stock_country_filter, business_rerank, recommend_topk


def recommend_for_client(
    artifacts: RecoArtifacts,
    *,
    client_id: int,
    clients_df: pd.DataFrame,
    products_df: pd.DataFrame,
    N_candidates: int = 200,
    top_k: int = 10,
    min_stock: float = 1.0,
    filtered: bool = True,
    stock_boost: float = 0.0,
    diversity_boost: float = 0.0,
    diversity_level: str = "FamilyLevel2",
    enforce_gender: bool = True,
) -> pd.DataFrame:
    """
    Single end-to-end inference call for Streamlit.

    Returns Top-K rows with:
      ClientID, ProductID, als_score, p_buy, StockQty, item_value, business_score + product metadata.
    """
    # --- 1) retrieve candidates
    product_ids, als_scores = retrieve_products(
        model=artifacts.als_model,
        X_user_item=artifacts.X_user_item,
        user2idx=artifacts.user2idx,
        idx2item=artifacts.idx2item,
        user_id=int(client_id),
        N=int(N_candidates),
        filtered=filtered
    )

    if len(product_ids) == 0:
        return pd.DataFrame()

    candidates_df = pd.DataFrame({
        "ClientID": int(client_id),
        "ProductID": product_ids.astype(np.int64),
        "als_score": als_scores.astype(np.float32),
    })

    # --- 2) predict p_buy
    scored_df = score_candidates_with_reranker(
        reranker_model=artifacts.reranker_model,
        candidates_df=candidates_df,
        user_feats=artifacts.user_feats,
        item_feats=artifacts.item_feats,
        max_train_date=artifacts.max_train_date,
        feature_cols=artifacts.feature_cols
    )

    # --- 3) stock filter
    filtered_df = apply_stock_country_filter(
        scored_df=scored_df,
        clients_df=clients_df,
        stock_ctry=artifacts.stock_ctry,
        min_stock=float(min_stock),
        products_df=products_df if enforce_gender else None,
        enforce_gender=enforce_gender
    )

    if filtered_df.empty:
        return pd.DataFrame()

    # --- 4) business rerank (+ diversity)
    final_df = business_rerank(
        scored_df=filtered_df,
        item_value=artifacts.item_value,
        products_df=products_df,
        stock_boost=float(stock_boost),
        diversity_boost=float(diversity_boost),
        diversity_level=diversity_level
    )

    # --- 5) take top-k
    top_df = recommend_topk(final_df, k=int(top_k),products_df=products_df)

    # # --- 6) attach product metadata for UI
    # meta_cols = ["ProductID", "Category", "FamilyLevel1", "FamilyLevel2", "Universe"]
    # meta_cols = [c for c in meta_cols if c in products_df.columns]

    # if meta_cols:
    #     top_df = top_df.merge(products_df[meta_cols], on="ProductID", how="left")

    # # nice column order
    # prefer_cols = [
    #     "ClientID", "ProductID",
    #     "Category", "FamilyLevel1", "FamilyLevel2", "Universe",
    #     "als_score", "p_buy",
    #     "StockQty", "item_value", "business_score",
    # ]
    # keep_cols = [c for c in prefer_cols if c in top_df.columns]
    return top_df#[keep_cols].reset_index(drop=True)
