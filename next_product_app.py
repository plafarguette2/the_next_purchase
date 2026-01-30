##########################################################################################
# ################################### Imports ############################################

from src.load_data import load_clients_data, load_products_data, load_stocks_data, load_stores_data, load_transactions_data, load_web_sessions_mock_data
from src.modeling.artifacts import load_artifacts
from src.modeling.pipeline import recommend_for_client
from src.store_recommendations import make_store_displays, build_display_from_product
from config import IMAGESPATH
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

##########################################################################################
################################### Mock Data ############################################

web_sessions_df = load_web_sessions_mock_data()
web_sessions_df['product_id'] = web_sessions_df['product_id'].astype(str)
np.random.seed(8)

##########################################################################################
##################################### Data ###############################################

@st.cache_data
def load_all_data():
    transactions_df = load_transactions_data()
    clients_df = load_clients_data()
    products_df = load_products_data()
    stores_df = load_stores_data()
    stocks_df = load_stocks_data()
    products_df_2 = load_products_data()
    return transactions_df, clients_df, products_df, stores_df, stocks_df, products_df_2


transactions_df, clients_df, products_df, stores_df, stocks_df ,products_df_2 = load_all_data()

##########################################################################################
##################################### Models #############################################

@st.cache_resource
def load_model_artifacts():
    return load_artifacts("model")

artifacts = load_model_artifacts()

##########################################################################################
################################## Constants #############################################

customer_segments = ['All'] + list(clients_df['ClientSegment'].unique())
customer_countries = ['All'] + list(clients_df['ClientCountry'].unique())

image_product_family_level1_dict = {
    'Bat': f'{IMAGESPATH}/Bat.svg',
    'Bike': f'{IMAGESPATH}/Bike.svg',
    'Cap': f'{IMAGESPATH}/Cap.svg',
    'Clubs': f'{IMAGESPATH}/Clubs.svg',
    'Dress': f'{IMAGESPATH}/Dress.svg',
    'Glove': f'{IMAGESPATH}/Glove.svg',
    'Gloves': f'{IMAGESPATH}/Gloves.svg',
    'Goggles': f'{IMAGESPATH}/Goggles.svg',
    'Helmet': f'{IMAGESPATH}/Helmet.svg',
    'Jersey': f'{IMAGESPATH}/Jersey.svg',
    'Knee Pads': f'{IMAGESPATH}/Knee Pads.svg',
    'Mat': f'{IMAGESPATH}/Mat.svg',
    'Pads': f'{IMAGESPATH}/Pads.svg',
    'Poles': f'{IMAGESPATH}/Poles.svg',
    'Puck': f'{IMAGESPATH}/Puck.svg',
    'Racket': f'{IMAGESPATH}/Racket.svg',
    'Shoes': f'{IMAGESPATH}/Shoes.svg',
    'Shorts': f'{IMAGESPATH}/Shorts.svg',
    'Shuttlecock': f'{IMAGESPATH}/Shuttlecock.svg',
    'Skates': f'{IMAGESPATH}/Skates.svg',
    'Skis': f'{IMAGESPATH}/Skis.svg',
    'Stick': f'{IMAGESPATH}/Stick.svg',
    'Swimsuit': f'{IMAGESPATH}/Swimsuit.svg',
    'T-shirt': f'{IMAGESPATH}/T-shirt.svg',
    'Ball': f'{IMAGESPATH}/Ball.svg'
}

##########################################################################################
################################## Useful functions ######################################

def get_product_price(product_id:str) -> float : 
    product_transactions = transactions_df[transactions_df['ProductID']==product_id]
    prices = product_transactions['SalesNetAmountEuro'] / product_transactions['Quantity']
    return f'{np.median(prices):.2f}'

# Predicting expected sales is a next step so this function helps generate mock data to display
def get_expected_sales(product_stock:float)-> int:
    if product_stock <=5:
        return np.random.randint(1,product_stock)
    else :
        return np.random.randint(np.floor(product_stock/2),product_stock)


##########################################################################################
##################################### General Setting ####################################

st.title("Customer Purchase Recommendation System")
st.set_page_config(layout="wide",initial_sidebar_state="expanded")
tab1, tab2, tab3 = st.tabs(['Client-level Recommendations','Store-level Recommendations','Monitoring Dashboard'])

##########################################################################################
##################################### Sidebar ############################################

with st.sidebar:
    st.title('Control Panel')
    st.write('')
    client_id = st.sidebar.text_input(
        label = '**👤 Client ID**',
        placeholder = 'Enter client ID',
        value='4508698145640552159'
        )
    recommended_products_customer_nb = int(st.sidebar.text_input(
    label = '**🛍️Number of products to recommend**',
    placeholder = 'Enter nb of products',
    value='5'
    ))
    st.write('---')
    store_id = st.sidebar.text_input(
        label = '**🏪 Store ID**',
        placeholder = 'Enter store ID',
        value='1450109522794525790'
        )
    displays_nb = int(st.sidebar.text_input(
        label = '**🖼️ Displays**',
        placeholder = 'Enter number of displays',
        value='3'
        ))
    product_by_display = int(st.sidebar.text_input(
        label = '**🏷️ Products by display**',
        placeholder = 'Enter number of products',
        value='4'
        ))  
    #st.write('---')
    # st.write('**🔽 Filters**')
    # country = st.sidebar.selectbox(
    #     label = 'Country',
    #     options = customer_countries,
    #     placeholder = 'Select country'
    # )
    # customer_segment = st.sidebar.selectbox(
    #     label = 'Customer Segment',
    #     options = customer_segments,
    #     placeholder = 'Select segment'
    #     )
    
##########################################################################################
#################################### Computations ########################################

displays, flat = make_store_displays(
    store_id=store_id,
    transactions=transactions_df,
    products=products_df,
    stocks=stocks_df,
    stores=stores_df,
    n_displays=displays_nb,
    items_per_display=product_by_display,
    min_stock=1,
    max_per_family=2,
)

customer_recommended_products = recommend_for_client(
    artifacts,
    client_id=int(client_id),
    clients_df=clients_df,
    products_df=products_df_2,
    N_candidates=200,
    top_k=recommended_products_customer_nb,
    min_stock=1.0,
    stock_boost=0.02,
    diversity_boost=0.05,
    diversity_level="FamilyLevel1",
    enforce_gender=False 
)

##########################################################################################
######################### Tab 1 : Client-level Recommendations ###########################

with tab1:
    for i in range(recommended_products_customer_nb):
        rec_product_id = customer_recommended_products['ProductID'].iloc[i]
        rec_product_family_level_1 = products_df_2[products_df_2['ProductID']==rec_product_id]['FamilyLevel1'].iloc[0]
        rec_product_category = products_df_2[products_df_2['ProductID']==rec_product_id]['Category'].iloc[0]
        rec_product_family_level_2 = products_df_2[products_df_2['ProductID']==rec_product_id]['FamilyLevel2'].iloc[0]
        rec_product_universe = products_df_2[products_df_2['ProductID']==rec_product_id]['Universe'].iloc[0]
        rec_product_stock = customer_recommended_products['StockQty'].iloc[i]
        rec_product_price = get_product_price(str(rec_product_id))      

        left_col, content_col, right_col = st.columns([1,2,1])
        with content_col:
            with st.container(border=True):
                subcol_image, subcol_description = st.columns([1,4])
                with subcol_image:
                    st.image(image_product_family_level1_dict[rec_product_family_level_1], width=100)
                with subcol_description:
                    subcol_badge, subcol_product_name = st.columns([1,6])
                    with subcol_badge:
                        st.badge(f'#{i+1}', color='blue', width=100)
                    with subcol_product_name:
                        st.write(f'**{rec_product_family_level_2}**')
                    st.caption(f'€ {rec_product_price}')
                    subcol_metric1, subcol_metric2 = st.columns([1,1])
                    with subcol_metric1:
                        st.badge(label=f'{rec_product_category}', color='grey', width='content')
                        if rec_product_universe=='Women':
                            color='violet'
                        else:
                            color='blue'
                        st.badge(label=f'{rec_product_universe}', color=color, width='content')
                    with subcol_metric2:
                        st.badge(label=f'{rec_product_family_level_1}', color='orange', width='content')
                        if rec_product_stock <= 2:
                            stock_status = 'Low stock'
                            stock_color = 'yellow'
                        else: 
                            stock_status = 'In stock'
                            stock_color = 'green'
                        st.badge(label=stock_status,icon=":material/package_2:", color=stock_color)

##########################################################################################
######################### Tab 2 : Store-level Recommendations ############################

with tab2:
    for i in range(displays_nb):
        nb_items = len(displays[i]['Items'])
        with st.container(border=True):
            st.subheader(f'Display {displays[i]["DisplayID"]} - {displays[i]["Category"]} ({displays[i]["Universe"]}) ')
            col1, col2 = st.columns([1,1])
            with col1:
                for j in range(0,nb_items,2):
                    product_category = displays[i]['Items']['Category'].iloc[j]
                    product_family_level_1 = displays[i]['Items']['FamilyLevel1'].iloc[j]
                    product_id = displays[i]['Items']['ProductID'].iloc[j]
                    product_family_level_2 = displays[i]['Items']['FamilyLevel2'].iloc[j]
                    product_store_stock = displays[i]['Items']['store_stock'].iloc[j]
                    product_price = get_product_price(product_id=product_id)
                    expected_sales_unit = get_expected_sales(product_stock=product_store_stock)
                    with st.container(border=True):
                        subcol_image, subcol_description = st.columns([1,4])
                        with subcol_image:
                            st.image(image_product_family_level1_dict[product_family_level_1], width=100)
                        with subcol_description:
                            subcol_badge, subcol_product_name = st.columns([1,6])
                            with subcol_badge:
                                st.badge(f'#{j+1}', color='blue', width=100)
                            with subcol_product_name:
                                st.write(f'**{product_family_level_2}**')
                            st.caption(f'{product_family_level_1}')
                            subcol_metric1, subcol_metric2 = st.columns([1,1])
                            with subcol_metric1:
                                with st.container(border=True): 
                                    subcol_caption, subcol_value = st.columns([2,3])
                                    with subcol_caption:
                                        st.caption('Price')
                                    with subcol_value:
                                        st.write(f'€ {product_price}')
                            with subcol_metric2:
                                with st.container(border=True): 
                                    subcol_caption2, subcol_value2 = st.columns([4,2])
                                    with subcol_caption2:
                                        st.caption('Exp. sales')
                                    with subcol_value2:
                                        st.write(f'{expected_sales_unit}')

                        subcol_stock, subcol_empty = st.columns([1,1])
                        with subcol_stock:
                            if product_store_stock <= 2:
                                st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='red')
                            elif product_store_stock <=5:
                                st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='yellow')
                            else:
                                st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='green')
            
            with col2:
                for j in range(1, nb_items, 2):
                    product_category = displays[i]['Items']['Category'].iloc[j]
                    product_family_level_1 = displays[i]['Items']['FamilyLevel1'].iloc[j]
                    product_id = displays[i]['Items']['ProductID'].iloc[j]
                    product_store_stock = displays[i]['Items']['store_stock'].iloc[j]
                    product_family_level_2= displays[i]['Items']['FamilyLevel2'].iloc[j]
                    product_price = get_product_price(product_id=product_id)
                    expected_sales_unit = get_expected_sales(product_stock=product_store_stock)
                    with st.container(border=True):
                        subcol_image, subcol_description = st.columns([1,4])
                        with subcol_image:
                            st.image(image_product_family_level1_dict[product_family_level_1], width=100)
                        with subcol_description:
                            subcol_badge, subcol_product_name = st.columns([1,6])
                            with subcol_badge:
                                st.badge(f'#{j+1}', color='blue', width=100)
                            with subcol_product_name:
                                st.write(f'**{product_family_level_2}**')
                            st.caption(f'{product_family_level_1}')
                            subcol_metric1, subcol_metric2 = st.columns([1,1])
                            with subcol_metric1:
                                with st.container(border=True): 
                                    subcol_caption, subcol_value = st.columns([2,3])
                                    with subcol_caption:
                                        st.caption('Price')
                                    with subcol_value:
                                        st.write(f'€ {product_price}')
                            with subcol_metric2:
                                with st.container(border=True): 
                                    subcol_caption2, subcol_value2 = st.columns([4,2])
                                    with subcol_caption2:
                                        st.caption('Exp. sales')
                                    with subcol_value2:
                                        st.write(f'{expected_sales_unit}')

                        subcol_stock, subcol_empty = st.columns([1,1])
                        with subcol_stock:
                            if product_store_stock <= 2:
                                st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='red')
                            elif product_store_stock <=5:
                                st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='yellow')
                            else:
                                st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='green')                


    # Initialize session state for multiple custom displays
    if "custom_displays" not in st.session_state:
        st.session_state.custom_displays = []  # List to store product IDs

    # Button to add a new display
    if st.button("Add customized display", icon=":material/add:"):
        st.session_state.custom_displays.append("")  # Placeholder for new product ID

    # Loop over all custom displays
    for idx in range(len(st.session_state.custom_displays)):
        # Input for this specific display
        product_id = st.text_input(
            f"Enter Product ID for custom display {idx+1}",
            value=st.session_state.custom_displays[idx],
            key=f"custom_input_{idx}"
        )
        # Save input back to session_state
        st.session_state.custom_displays[idx] = product_id

        if product_id:
            custom_display = build_display_from_product(
                store_id,
                product_id,
                transactions=transactions_df,
                products=products_df,
                stocks=stocks_df,
                stores=stores_df,
                items_per_display=product_by_display,
                min_stock=3,
                max_per_family=2
            )
            anchor_product = products_df[products_df['ProductID']==product_id]['FamilyLevel2'].iloc[0]
            nb_items = len(custom_display)
            # render display
            with st.container(border=True):
                st.subheader(f"Custom display - Product {anchor_product}")
                customize_col1, customize_col2 = st.columns([1,1])

                with customize_col1:
                    for j in range(0,nb_items,2):
                        product_category = custom_display['Category'].iloc[j]
                        product_family_level_1 = custom_display['FamilyLevel1'].iloc[j]
                        product_id = custom_display['ProductID'].iloc[j]
                        product_family_level_2 = custom_display['FamilyLevel2'].iloc[j]
                        product_store_stock = custom_display['store_stock'].iloc[j]
                        product_price = get_product_price(product_id=product_id)
                        expected_sales_unit = get_expected_sales(product_stock=product_store_stock)
                    
                        with st.container(border=True):
                            subcol_image, subcol_description = st.columns([1,4])
                            with subcol_image:
                                st.image(image_product_family_level1_dict[product_family_level_1], width=100)
                            with subcol_description:
                                subcol_badge, subcol_product_name = st.columns([1,6])
                                with subcol_badge:
                                    st.badge(f'#{j+1}', color='blue', width=100)
                                with subcol_product_name:
                                    st.write(f'**{product_family_level_2}**')
                                st.caption(f'{product_family_level_1}')
                                subcol_metric1, subcol_metric2 = st.columns([1,1])
                                with subcol_metric1:
                                    with st.container(border=True): 
                                        subcol_caption, subcol_value = st.columns([2,3])
                                        with subcol_caption:
                                            st.caption('Price')
                                        with subcol_value:
                                            st.write(f'€ {product_price}')
                                with subcol_metric2:
                                    with st.container(border=True): 
                                        subcol_caption2, subcol_value2 = st.columns([4,2])
                                        with subcol_caption2:
                                            st.caption('Exp. sales')
                                        with subcol_value2:
                                            st.write(f'{expected_sales_unit}')

                            subcol_stock, subcol_empty = st.columns([1,1])
                            with subcol_stock:
                                if product_store_stock <= 2:
                                    st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='red')
                                elif product_store_stock <=5:
                                    st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='yellow')
                                else:
                                    st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='green')

                with customize_col2:
                    for j in range(1,nb_items,2):
                        product_category = custom_display['Category'].iloc[j]
                        product_family_level_1 = custom_display['FamilyLevel1'].iloc[j]
                        product_id = custom_display['ProductID'].iloc[j]
                        product_family_level_2 = custom_display['FamilyLevel2'].iloc[j]
                        product_store_stock = custom_display['store_stock'].iloc[j]
                        product_price = get_product_price(product_id=product_id)
                        expected_sales_unit = get_expected_sales(product_stock=product_store_stock)
                    
                        with st.container(border=True):
                            subcol_image, subcol_description = st.columns([1,4])
                            with subcol_image:
                                st.image(image_product_family_level1_dict[product_family_level_1], width=100)
                            with subcol_description:
                                subcol_badge, subcol_product_name = st.columns([1,6])
                                with subcol_badge:
                                    st.badge(f'#{j+1}', color='blue', width=100)
                                with subcol_product_name:
                                    st.write(f'**{product_family_level_2}**')
                                st.caption(f'{product_family_level_1}')
                                subcol_metric1, subcol_metric2 = st.columns([1,1])
                                with subcol_metric1:
                                    with st.container(border=True): 
                                        subcol_caption, subcol_value = st.columns([2,3])
                                        with subcol_caption:
                                            st.caption('Price')
                                        with subcol_value:
                                            st.write(f'€ {product_price}')
                                with subcol_metric2:
                                    with st.container(border=True): 
                                        subcol_caption2, subcol_value2 = st.columns([4,2])
                                        with subcol_caption2:
                                            st.caption('Exp. sales')
                                        with subcol_value2:
                                            st.write(f'{expected_sales_unit}')

                            subcol_stock, subcol_empty = st.columns([1,1])
                            with subcol_stock:
                                if product_store_stock <= 2:
                                    st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='red')
                                elif product_store_stock <=5:
                                    st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='yellow')
                                else:
                                    st.badge(f'{product_store_stock} units',icon=":material/package_2:", color='green')

##########################################################################################
############################# Tab 3 : Monitoring Dashboard ###############################

with tab3:
    st.header("Online Recommendation Monitoring")

    # ---------------------------
    # Controls
    # ---------------------------
    col_ctrl1, col_ctrl2 = st.columns(2)

    with col_ctrl1:
        country_list = ["All"] + sorted(web_sessions_df["country"].unique())
        selected_country = st.selectbox("Select country", country_list, index=5)

    with col_ctrl2:
        product_list = ["All"] + sorted(web_sessions_df["product_id"].unique())
        selected_product = st.selectbox("Select product", product_list)

    # ---------------------------
    # Filtering
    # ---------------------------
    df = web_sessions_df.copy()
    df["session_date"] = pd.to_datetime(df["session_date"])

    if selected_country != "All":
        df_country = df[df["country"] == selected_country]
    else:
        df_country = df.copy()

    if selected_product != "All":
        df_filtered = df_country[df_country["product_id"] == selected_product]
    else:
        df_filtered = df_country.copy()

    df_filtered["date"] = df_filtered["session_date"].dt.date
    df_country["date"] = df_country["session_date"].dt.date  # for table

    # CTR & Conversion Rate (rolling, dual axis)


    # Aggregate per session
    session_metrics = (
        df_filtered.groupby("session_id")
        .agg(
            date=("session_date", "first"),
            clicked_any=("is_clicked", "max"),  # 1 if at least one click
            bought_any=("is_bought", "max")    # 1 if at least one purchase
        )
        .reset_index()
    )

    # Daily CTR and Conversion Rate
    daily_metrics = (
        session_metrics.groupby(session_metrics["date"].dt.date)
        .agg(
            impressions=("session_id", "count"),
            ctr=("clicked_any", "mean"),           # fraction of sessions with >=1 click
            conversion_rate=("bought_any", "mean") # fraction of sessions with >=1 purchase
        )
        .reset_index()
    )

    # Rolling average
    rolling_days = 7
    daily_metrics["ctr_roll"] = daily_metrics["ctr"].rolling(rolling_days, min_periods=1).mean()
    daily_metrics["conversion_rate_roll"] = daily_metrics["conversion_rate"].rolling(rolling_days, min_periods=1).mean()

    # Plot
    fig_rates = go.Figure()
    fig_rates.add_trace(
        go.Scatter(
            x=daily_metrics["date"],
            y=daily_metrics["ctr_roll"],
            name="CTR (rolling)",
            yaxis="y1",
            mode="lines+markers"
        )
    )
    fig_rates.add_trace(
        go.Scatter(
            x=daily_metrics["date"],
            y=daily_metrics["conversion_rate_roll"],
            name="Conversion Rate (rolling)",
            yaxis="y2",
            mode="lines+markers"
        )
    )

    fig_rates.update_layout(
        title=f"CTR & Conversion Rate Evolution (Last 30 Days) - {selected_country} (Mock Data)",
        xaxis_title="Date",
        yaxis=dict(title="CTR"), 
        yaxis2=dict(title="Conversion Rate", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        height=450
    )
    st.plotly_chart(fig_rates, use_container_width=True)

    # Funnel (last 7 days)
   
    last_7_days = df_filtered["session_date"].max() - pd.Timedelta(days=6)
    df_7d = df_filtered[df_filtered["session_date"] >= last_7_days]

    # Use new definition for funnel
    recommended_sessions = df_7d["session_id"].nunique()
    clicked_sessions = df_7d.groupby("session_id")["is_clicked"].max().sum()  # number of sessions with at least one click
    bought_sessions = df_7d.groupby("session_id")["is_bought"].max().sum()    # number of sessions with at least one purchase

    fig_funnel = go.Figure(
        go.Funnel(
            y=["Recommended Sessions", "Sessions Clicked", "Sessions Purchased"],
            x=[recommended_sessions, clicked_sessions, bought_sessions]
        )
    )
    fig_funnel.update_layout(
        title="Recommendation Funnel (Last 7 Days) (Mock Data)",
        height=400
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Top 10 most recommended products (country only)

    top_products = (
        df_country.groupby("product_id")
        .size()
        .reset_index(name="times_recommended")
        .sort_values("times_recommended", ascending=False)
        .head(10)
    )

    # Ensure same dtype for merge
    top_products["product_id"] = top_products["product_id"].astype(int)
    products_df["ProductID"] = products_df["ProductID"].astype(int)

    top_products = top_products.merge(
        products_df[["ProductID", "FamilyLevel2"]],
        left_on="product_id",
        right_on="ProductID",
        how="left"
    )
    top_products['product_id'] = top_products['product_id'].astype(str)

    st.write("**Top 10 Most Recommended Products (Mock Data)**")
    st.dataframe(
        top_products[["product_id", "FamilyLevel2", "times_recommended"]],
        use_container_width=True
    )

    ################################################################################

    categories = products_df["Category"].unique()

    coverage_mock = pd.DataFrame({
        "Category": categories,
        # simulate realistic availability
        "nb_products_available": np.random.randint(30, 200, size=len(categories))
    })

    # simulate biased recommendation engine
    coverage_mock["coverage_pct"] = np.clip(
        np.random.beta(a=2, b=5, size=len(categories)),  # skewed towards low coverage
        0.1,
        0.9
    )

    coverage_mock["nb_products_recommended"] = (
        coverage_mock["nb_products_available"] * coverage_mock["coverage_pct"]
    ).astype(int)

    coverage_mock["coverage_pct"] = (
        coverage_mock["nb_products_recommended"] /
        coverage_mock["nb_products_available"]
    )

    # -----------------------------
    # Plot
    # -----------------------------
    fig_coverage = px.bar(
        coverage_mock.sort_values("coverage_pct", ascending=False),
        x="Category",
        y="coverage_pct",
        title="Recommendation Coverage by Category (Mock Data)",
        labels={"coverage_pct": "Coverage (%)"},
        text_auto=".0%"
    )

    fig_coverage.update_layout(
        yaxis_tickformat=".0%",
        xaxis_title="Category",
        yaxis_title="Coverage",
        height=420
    )

    st.plotly_chart(fig_coverage, use_container_width=True)

    # Mock A/B testing data

    dates = pd.date_range("2026-01-01", "2026-01-30", freq="D")

    ab_df = pd.DataFrame({
        "date": dates,
        # Recommended group sells more on average
        "units_sold_recommended": np.random.poisson(lam=105, size=len(dates)),
        "units_sold_control": np.random.poisson(lam=85, size=len(dates))
    })

    # Rolling average for smoother curves
    rolling_days = 3
    ab_df["recommended_roll"] = ab_df["units_sold_recommended"].rolling(
        rolling_days, min_periods=1
    ).mean()

    ab_df["control_roll"] = ab_df["units_sold_control"].rolling(
        rolling_days, min_periods=1
    ).mean()

    # -----------------------------
    # Plot
    # -----------------------------
    fig_ab = go.Figure()

    fig_ab.add_trace(
        go.Scatter(
            x=ab_df["date"],
            y=ab_df["recommended_roll"],
            name="Recommended group",
            mode="lines+markers"
        )
    )

    fig_ab.add_trace(
        go.Scatter(
            x=ab_df["date"],
            y=ab_df["control_roll"],
            name="Control group",
            mode="lines+markers"
        )
    )

    fig_ab.update_layout(
        title="A/B Testing – Units Sold Over Time",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=420,
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig_ab, use_container_width=True)












#st.balloons()