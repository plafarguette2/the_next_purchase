import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def make_store_displays(
    store_id: str,
    *,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    stocks: pd.DataFrame,
    stores: pd.DataFrame,
    n_displays: int = 4,
    items_per_display: int = 8,
    top_k_recs: int = 47458,          # generate more candidates than needed
    min_stock: float = 1,           # only keep items with pseudo stock > min_stock
    max_per_family: int = 2,        # max items from same FamilyLevel1 within a display
    category_col: str = "Category",
    family_col: str = "FamilyLevel1",
    universe_col: str = "Universe",
):
    """
    End-to-end store-level entrance display builder.

    What it does:
      1) Builds Store×Product matrix (Quantity) and store-store cosine similarity.
      2) Recommends products for the target store using "similar stores" weighted sales.
      3) Filters recommendations using country-level stock, assuming each store 
      in the same country has access to the same inventory pool.
      4) Filters recs by min_stock.
      5) Builds displays grouped by Category, NO mixing Universe, and max_per_family per FamilyLevel1.

    Returns:
      displays (list of dicts), flat (DataFrame)
    """

    # types
    store_id = str(store_id)
    for col in ["StoreID", "ProductID", "ClientID"]:
        if col in transactions.columns:
            transactions[col] = transactions[col].astype(str)
    products["ProductID"] = products["ProductID"].astype(str)
    stores["StoreID"] = stores["StoreID"].astype(str)
    stocks["ProductID"] = stocks["ProductID"].astype(str)

    # Build Store×Product matrix and similarity
 
    tx = transactions.merge(products[["ProductID", category_col, family_col, universe_col]], on="ProductID", how="left")

    store_product = tx.pivot_table(
        index="StoreID",
        columns="ProductID",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    if store_id not in store_product.index:
        raise ValueError(f"store_id={store_id} not found in transactions/StoreID")

    store_product_norm = normalize(store_product, norm="l2", axis=1)
    store_similarity = cosine_similarity(store_product_norm)
    store_similarity_df = pd.DataFrame(
        store_similarity,
        index=store_product.index,
        columns=store_product.index
    )

    def get_similar_stores(sid, top_n=5):
        return (
            store_similarity_df[sid]
            .drop(sid)
            .sort_values(ascending=False)
            .head(top_n)
        )

    def recommend_products_for_store(sid, top_k=100):
        similar = get_similar_stores(sid, top_n=5)

        scores = pd.Series(0.0, index=store_product.columns)

        for sim_store, weight in similar.items():
            scores += store_product.loc[sim_store].astype(float) * float(weight)

        # "Gap": subtract what target already sells
        scores -= store_product.loc[sid].astype(float)

        return scores.sort_values(ascending=False).head(top_k)

    # Build recommendations DataFrame (candidates)
    recs = recommend_products_for_store(store_id, top_k=top_k_recs)

    recs_df = (
        recs.rename("score")
            .reset_index()
            .rename(columns={"index": "ProductID"})
            .merge(products, on="ProductID", how="left")
    )

    # Pseudo store-level stock (within store country)

    store_country = (
        stores.loc[stores["StoreID"] == store_id, "StoreCountry"]
        .iloc[0]
    )

    # country-level stock per product (this is what you actually have)
    stock_country = (
        stocks[stocks["StoreCountry"] == store_country]
        .groupby("ProductID")["Quantity"]
        .sum()
        .reset_index(name="country_stock")
    )

    # Simplified store stock = country stock
    store_country = (
        stores.loc[stores["StoreID"] == store_id, "StoreCountry"]
        .iloc[0]
    )

    # country-level stock per product
    stock_country = (
        stocks[stocks["StoreCountry"] == store_country]
        .groupby("ProductID")["Quantity"]
        .sum()
        .reset_index(name="store_stock")   # we treat it as store_stock for this simplification
    )

    # keep only products above threshold
    store_stock_lookup = stock_country[stock_country["store_stock"] > min_stock][["ProductID", "store_stock"]]

    # Filter recommendations by pseudo store stock threshold
    recs_df = recs_df.merge(store_stock_lookup, on="ProductID", how="inner")

    # Build Category displays (no mixing Universe, diversity on FamilyLevel1)
    def build_category_displays(
        recs_df: pd.DataFrame,
        n_displays: int,
        items_per_display: int,
        *,
        category_col: str,
        family_col: str,
        universe_col: str,
        score_col: str = "score",
        product_col: str = "ProductID",
        max_per_family: int = 2,
    ):
        required = {product_col, score_col, category_col, family_col, universe_col}
        missing = required - set(recs_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in recs_df: {missing}")

        df = recs_df.copy()
        df = df.dropna(subset=[product_col, score_col, category_col, family_col, universe_col])
        df = df.sort_values(score_col, ascending=False)
        df = df.drop_duplicates(subset=[product_col], keep="first")

        def _pick_items(g: pd.DataFrame) -> pd.DataFrame:
            counts = {}
            chosen_rows = []
            for row in g.itertuples(index=False):
                fam = getattr(row, family_col)
                if counts.get(fam, 0) >= max_per_family:
                    continue
                chosen_rows.append(row)
                counts[fam] = counts.get(fam, 0) + 1
                if len(chosen_rows) >= items_per_display:
                    break
            return pd.DataFrame(chosen_rows, columns=g.columns)

        # score each (Category, Universe) bucket with the diversity-aware selection
        bucket_rows = []
        for (cat, uni), g in df.groupby([category_col, universe_col], sort=False):
            picked = _pick_items(g)
            bucket_rows.append({
                category_col: cat,
                universe_col: uni,
                "bucket_score": float(picked[score_col].sum()),
                "n_items_possible": int(len(picked)),
            })

        buckets = pd.DataFrame(bucket_rows).sort_values(
            ["bucket_score", "n_items_possible"], ascending=False
        )
        buckets = buckets[buckets["n_items_possible"] > 0]

        chosen = buckets.head(n_displays)[[category_col, universe_col]]

        displays = []
        assigned = []
        for i, (cat, uni) in enumerate(chosen.itertuples(index=False), start=1):
            g = df[(df[category_col] == cat) & (df[universe_col] == uni)].copy()
            items = _pick_items(g).copy()
            items["DisplayID"] = i
            items["DisplayName"] = f"Display {i}: {cat} ({uni})"
            displays.append({"DisplayID": i, "Category": cat, "Universe": uni, "Items": items})
            assigned.append(items)

        flat = pd.concat(assigned, ignore_index=True) if assigned else pd.DataFrame()
        return displays, flat

    displays, flat = build_category_displays(
        recs_df,
        n_displays=n_displays,
        items_per_display=items_per_display,
        category_col=category_col,
        family_col=family_col,
        universe_col=universe_col,
        max_per_family=max_per_family,
    )

    return displays, flat