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

        # def _pick_items(g: pd.DataFrame) -> pd.DataFrame:
        #     counts = {}
        #     chosen_rows = []
        #     for row in g.itertuples(index=False):
        #         fam = getattr(row, family_col)
        #         if counts.get(fam, 0) >= max_per_family:
        #             continue
        #         chosen_rows.append(row)
        #         counts[fam] = counts.get(fam, 0) + 1
        #         if len(chosen_rows) >= items_per_display:
        #             break
        #     return pd.DataFrame(chosen_rows, columns=g.columns)

        def _pick_items(g: pd.DataFrame) -> pd.DataFrame:
            counts = {}
            family2_used = set()  ### added line ###
            chosen_rows = []
            for row in g.itertuples(index=False):
                fam = getattr(row, family_col)
                fam2 = getattr(row, "FamilyLevel2")  ### added line ###
                if counts.get(fam, 0) >= max_per_family:
                    continue
                if fam2 in family2_used:  ### added line ###
                    continue
                chosen_rows.append(row)
                counts[fam] = counts.get(fam, 0) + 1
                family2_used.add(fam2)  ### added line ###
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




######################################################################################################



def build_display_from_product(
    store_id: str,
    anchor_product_id: str,
    *,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    stocks: pd.DataFrame,
    stores: pd.DataFrame,
    items_per_display: int = 8,
    top_k_recs: int = 800,
    min_stock: float = 1,
    max_per_family: int = 2,
    category_col: str = "Category",
    family_col: str = "FamilyLevel1",
    family2_col: str = "FamilyLevel2",
    universe_col: str = "Universe",
):
    store_id = str(store_id)
    anchor_product_id = str(anchor_product_id)

    #################
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

    ####################

    # --- Anchor metadata
    prod_meta = products.copy()
    prod_meta["ProductID"] = prod_meta["ProductID"].astype(str)

    anchor_row = prod_meta.loc[prod_meta["ProductID"] == anchor_product_id]
    if anchor_row.empty:
        raise ValueError(f"anchor_product_id={anchor_product_id} not found in products")

    anchor_cat = anchor_row.iloc[0][category_col]
    anchor_uni = anchor_row.iloc[0][universe_col]
    if pd.isna(anchor_cat) or pd.isna(anchor_uni):
        raise ValueError("Anchor product missing Category/Universe")

    # --- Store country
    stores_ = stores.copy()
    stores_["StoreID"] = stores_["StoreID"].astype(str)

    sc = stores_.loc[stores_["StoreID"] == store_id, "StoreCountry"]
    if sc.empty:
        raise ValueError(f"store_id={store_id} not found in stores")
    store_country = sc.iloc[0]

    # --- Country stock (simplification) -> treat as store stock
    stocks_ = stocks.copy()
    stocks_["ProductID"] = stocks_["ProductID"].astype(str)

    stock_country = (
        stocks_[stocks_["StoreCountry"] == store_country]
        .groupby("ProductID")["Quantity"]
        .sum()
        .reset_index(name="store_stock")
    )
    stock_country = stock_country[stock_country["store_stock"] > min_stock]

    # --- Candidate recs (expects your recommend_products_for_store exists)
    recs = recommend_products_for_store(store_id, top_k=top_k_recs)  # Series: idx=ProductID, val=score

    recs_df = (
        recs.rename("score")
            .reset_index()
            .rename(columns={"index": "ProductID"})
    )
    recs_df["ProductID"] = recs_df["ProductID"].astype(str)

    # Merge minimal product metadata (avoid suffix issues)
    recs_df = recs_df.merge(
        prod_meta[["ProductID", category_col, family_col, family2_col, universe_col]],
        on="ProductID",
        how="left"
    )

    # Restrict to anchor (Category, Universe) to avoid mixing
    recs_df = recs_df[
        (recs_df[category_col] == anchor_cat) &
        (recs_df[universe_col] == anchor_uni)
    ].copy()

    # Stock filter
    recs_df = recs_df.merge(stock_country[["ProductID", "store_stock"]], on="ProductID", how="inner")

    # Ensure anchor exists + is in stock
    if anchor_product_id not in set(recs_df["ProductID"]):
        anchor_stock = stock_country.loc[stock_country["ProductID"] == anchor_product_id]
        if anchor_stock.empty:
            raise ValueError("Anchor product not in stock (after min_stock filter).")
        anchor_add = pd.DataFrame([{
            "ProductID": anchor_product_id,
            "score": (recs_df["score"].max() + 1) if not recs_df.empty else 1.0,
            category_col: anchor_cat,
            universe_col: anchor_uni,
            family_col: anchor_row.iloc[0][family_col],
            family2_col: anchor_row.iloc[0][family2_col],
            "store_stock": float(anchor_stock.iloc[0]["store_stock"]),
        }])
        recs_df = pd.concat([anchor_add, recs_df], ignore_index=True)

    # Sort by score
    recs_df = recs_df.sort_values("score", ascending=False).drop_duplicates("ProductID", keep="first")

    # Pick IDs (anchor first) with max_per_family
    chosen_ids = []
    family_counts = {}

    # Add anchor first
    anchor_meta = recs_df.loc[recs_df["ProductID"] == anchor_product_id].iloc[0]
    chosen_ids.append(anchor_product_id)
    family_counts[str(anchor_meta[family_col])] = 1

    # Fill remaining
    for _, r in recs_df.iterrows():
        pid = r["ProductID"]
        if pid == anchor_product_id:
            continue
        fam = str(r[family_col])
        if family_counts.get(fam, 0) >= max_per_family:
            continue
        chosen_ids.append(pid)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if len(chosen_ids) >= items_per_display:
            break

    # Build final display DF in chosen order
    display_df = recs_df.set_index("ProductID").loc[chosen_ids].reset_index()

    display_df["DisplayName"] = f"Anchor display: {anchor_product_id} | {anchor_cat} ({anchor_uni})"
    display_df["StoreID"] = store_id

    cols = ["StoreID", "DisplayName", "ProductID", "score", "store_stock",
            category_col, family_col, family2_col, universe_col]
    cols = [c for c in cols if c in display_df.columns]
    return display_df[cols]