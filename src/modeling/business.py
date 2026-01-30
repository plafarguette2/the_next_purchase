import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression

def compute_item_value(train_tx: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-product expected value proxy from TRAIN ONLY.
    Output: ProductID, item_value (avg net amount per unit)
    """
    item_value = (
        train_tx
        .groupby("ProductID", as_index=False)
        .agg(
            spend_sum=("SalesNetAmountEuro", "sum"),
            qty_sum=("Quantity", "sum")
        )
        .assign(
            item_value=lambda x: (x["spend_sum"] / x["qty_sum"].clip(lower=1)).astype(np.float32)
        )[["ProductID", "item_value"]]
    )
    return item_value


def build_stock_country_lookup(stores: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Join stocks with store countries (if needed) and produce a country-product stock table.
    Your stocks already look like StoreCountry + ProductID + Quantity, so this is mostly identity.
    Output: StoreCountry, ProductID, StockQty
    """
    stock_ctry = (
        stocks
        .rename(columns={"Quantity": "StockQty"})
        .copy()
    )
    return stock_ctry


def apply_stock_country_filter(scored_df: pd.DataFrame,
                               clients_df: pd.DataFrame,
                               stock_ctry: pd.DataFrame,
                               min_stock: float = 1.0,
                               products_df: pd.DataFrame | None = None,
                               enforce_gender: bool = False) -> pd.DataFrame:
    """
    Filters scored candidates to items that are in-stock in the client's country.
    Optionally enforces gender-universe matching:
      - products_df must include ProductID and Universe ('Men'/'Women')
      - clients_df must include ClientGender ('M'/'F'/...).

    Rule (simple & safe):
      - If ClientGender in {'M','F'} and Universe in {'Men','Women'}: must match
      - Otherwise: do not filter (keep)
    """
    df = (
        scored_df
        .merge(clients_df[["ClientID", "ClientCountry", "ClientGender"]], on="ClientID", how="left")
        .merge(
            stock_ctry,
            left_on=["ClientCountry", "ProductID"],
            right_on=["StoreCountry", "ProductID"],
            how="left"
        )
        .assign(
            StockQty=lambda x: x["StockQty"].fillna(0).astype(np.float32)
        )
    )

    # hard stock filter
    df = df[df["StockQty"] >= float(min_stock)].copy()

    # optional hard gender filter
    if enforce_gender:
        if products_df is None:
            raise ValueError("products_df must be provided when enforce_gender=True")

        uni = products_df[["ProductID", "Universe"]].copy()
        df = df.merge(uni, on="ProductID", how="left")

        # only enforce when both sides are in known sets
        g = df["ClientGender"].astype(str)
        u = df["Universe"].astype(str)

        known_gender = g.isin(["M", "F"])
        known_uni = u.isin(["Men", "Women"])

        # mapping match
        match = ((g == "M") & (u == "Men")) | ((g == "F") & (u == "Women"))

        # keep rows where we can't/shouldn't enforce OR where it matches
        df = df[(~(known_gender & known_uni)) | match].copy()

    return df


def business_rerank(scored_df: pd.DataFrame,
                    item_value: pd.DataFrame,
                    products_df: pd.DataFrame | None = None,
                    w_prob: float = 1.0,
                    w_value: float = 1.0,
                    stock_boost: float = 0.0,
                    diversity_boost: float = 0.0,
                    diversity_level: str = "FamilyLevel2") -> pd.DataFrame:
    """
    Compute a final business score and sort, with optional soft diversity/coverage boost.

    Base:
      business_score = p_buy * item_value
    Optional:
      + stock_boost * log1p(StockQty)
      + diversity_boost * novelty(group)   where novelty(group) decreases for frequent groups

    Diversity logic (soft, global):
      - Map each product to a group (e.g., FamilyLevel2)
      - Compute group frequency within the candidate list (globally or per user indirectly)
      - Give a higher boost to under-represented groups to encourage coverage

    Notes:
      - This is a *soft* boost (not a hard constraint).
      - For a true per-user greedy diversified selection, you'd do an iterative MMR-like loop,
        but this keeps things simple and fast.
    """
    df = (
        scored_df
        .merge(item_value, on="ProductID", how="left")
        .assign(
            item_value=lambda x: x["item_value"].fillna(0).astype(np.float32),
            business_score=lambda x: (w_prob * x["p_buy"] * w_value * x["item_value"]).astype(np.float32),
        )
    )

    # stock boost (optional)
    if stock_boost and "StockQty" in df.columns:
        df["business_score"] = (df["business_score"] + stock_boost * np.log1p(df["StockQty"])).astype(np.float32)

    # diversity/coverage boost (optional)
    if diversity_boost and (products_df is not None) and (diversity_level in products_df.columns):
        # attach group label
        df = df.merge(products_df[["ProductID", diversity_level]], on="ProductID", how="left")

        # group frequency within the candidate set (global across df)
        # rare groups get larger boost: novelty = 1/sqrt(freq)
        grp_freq = (
            df[diversity_level]
            .fillna("UNKNOWN")
            .value_counts()
            .rename("grp_freq")
            .reset_index()
            .rename(columns={"index": diversity_level})
        )

        df = (
            df
            .merge(grp_freq, on=diversity_level, how="left")
            .assign(
                grp_freq=lambda x: x["grp_freq"].fillna(1).astype(np.float32),
                diversity_bonus=lambda x: (diversity_boost / np.sqrt(x["grp_freq"])).astype(np.float32),
            )
        )

        df["business_score"] = (df["business_score"] + df["diversity_bonus"]).astype(np.float32)

    # final ordering
    df = df.sort_values(["business_score", "p_buy", "als_score"], ascending=False).reset_index(drop=True)
    return df

# def recommend_topk(scored_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
#     """
#     Return Top-K per user. Assumes scored_df already has business_score.
#     """
#     return (
#         scored_df
#         .groupby("ClientID", group_keys=False)
#         .head(k)
#         .reset_index(drop=True)
#     )

def recommend_topk(scored_df: pd.DataFrame, products_df : pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Return Top-K per user. Assumes scored_df already has business_score.
    """

    meta_cols = ["ProductID", "Category", "FamilyLevel1", "FamilyLevel2", "Universe"]
    meta_cols = [c for c in meta_cols if c in products_df.columns]

    if meta_cols:
        scored_df = scored_df.merge(products_df[meta_cols], on="ProductID", how="left")

    # nice column order
    prefer_cols = [
        "ClientID", "ProductID",
        "Category", "FamilyLevel1", "FamilyLevel2", "Universe",
        "als_score", "p_buy",
        "StockQty", "item_value", "business_score",
    ]
    keep_cols = [c for c in prefer_cols if c in scored_df.columns]
    return (
        scored_df
        .drop_duplicates(subset=["ClientID", "FamilyLevel2"], keep="first")
        .groupby("ClientID", group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )[keep_cols]