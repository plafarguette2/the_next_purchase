##########################################################################################
# ################################### Imports ############################################

from src.load_data import load_clients_data, load_products_data, load_stocks_data, load_stores_data, load_transactions_data
from src.store_recommendations import make_store_displays
from config import IMAGESPATH
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

##########################################################################################
################################### Mock Data ############################################

recommended_product_ids = ['7307348902933965958', '1686112861025120737', '6444962679803347392',
                           '7933410289764905369', '8649733852988864613', '1598302186363721653',
                           '43220326960179274', '4756273066838708224', '738601866461520214',
                           '5137787290282025561', '131684282376766098', '5135449292488127071',
                           '44114231881774132', '2988727891119413443', '883558141088536721',
                           '6009228299796499764', '5643052698518512004', '1663437793332066358',
                           '1093068947787812522']

product_price = 867
expected_sales_unit = 120

##########################################################################################
##################################### Data ###############################################

transactions_df = load_transactions_data()
clients_df = load_clients_data()
products_df = load_products_data()
stores_df = load_stores_data()
stocks_df = load_stocks_data()

##########################################################################################
################################## Constants #############################################

customer_segments = ['All'] + list(clients_df['ClientSegment'].unique())
customer_countries = ['All'] + list(clients_df['ClientCountry'].unique())

image_product_category_dict = {
    'Football' : f"{IMAGESPATH}/futbol-solid-full.svg",
    'Cycling' : f"{IMAGESPATH}/bicycle-solid-full.svg",
    'Badminton' : f"{IMAGESPATH}/sports_tennis_24dp_1E3050.svg",
    'Handball' : f"{IMAGESPATH}/sports_handball_24dp_1E3050.svg",
    'Tennis' : f"{IMAGESPATH}/sports_tennis_24dp_1E3050.svg",
    'Golf' : f"{IMAGESPATH}/golf-ball-tee-solid-full.svg",
    'Hockey' : f"{IMAGESPATH}/hockey-puck-solid-full.svg",
    'Volleyball' : f"{IMAGESPATH}/volleyball-solid-full.svg",
    'Baseball' : f"{IMAGESPATH}/baseball-bat-ball-solid-full.svg",
    'Basketball' : f"{IMAGESPATH}/basketball-solid-full.svg",
    'Rugby' : f"{IMAGESPATH}/sports_rugby_24dp_1E3050.svg",
    'Cricket' : f"{IMAGESPATH}/sports_cricket_24dp_1E3050.svg",
    'Running' : f"{IMAGESPATH}/person-running-solid-full.svg",
    'Beach' : f"{IMAGESPATH}/umbrella-beach-solid-full.svg",
    'Softball' : f"{IMAGESPATH}/softball-solid-full.svg",
    'Skiing' : f"{IMAGESPATH}/person-skiing-solid-full.svg",
    'Gymnastics' : f"{IMAGESPATH}/sports_gymnastics_24dp_1E3050.svg",
    'Skating' : f"{IMAGESPATH}/person-skating-solid-full.svg",
    'Swimming' : f"{IMAGESPATH}/person-swimming-solid-full.svg"
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
recommended_products_customer_nb = 5

##########################################################################################
##################################### Sidebar ############################################

with st.sidebar:
    st.title('Control Panel')
    st.write('')
    client_id = st.sidebar.text_input(
        label = '**👤 Client ID**',
        placeholder = 'Enter client ID'
        )
    st.write('---')
    store_id = st.sidebar.text_input(
        label = '**🏪 Store ID**',
        placeholder = 'Enter store ID',
        value='2686511472610728845'
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
    st.write('---')
    st.write('**🔽 Filters**')
    country = st.sidebar.selectbox(
        label = 'Country',
        options = customer_countries,
        placeholder = 'Select country'
    )
    customer_segment = st.sidebar.selectbox(
        label = 'Customer Segment',
        options = customer_segments,
        placeholder = 'Select segment'
        )
    
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

##########################################################################################
######################### Tab 1 : Client-level Recommendations ###########################

with tab1:
    for i in range(recommended_products_customer_nb):
        product_price = get_product_price(recommended_product_ids[i])
        with st.container(border=True):
            col_icon, col_product_name = st.columns([1,10])

            with col_icon:
                st.write('')
                st.image(f'{IMAGESPATH}/counter_{i+1}_24dp_1E3050_FILL0_wght400_GRAD0_opsz24.svg', width=60)
            with col_product_name:
                st.subheader(f'{recommended_product_ids[i]}')
                st.write(f'{products_df[products_df["ProductID"]==recommended_product_ids[i]]["FamilyLevel1"].iloc[0]} - {product_price} €')
            
            st.write('---')

            col_image, col_factors = st.columns([1,3])
            with col_image :
                st.image(image_product_category_dict[products_df[products_df['ProductID']==recommended_product_ids[i]]['Category'].iloc[0]], width=200)
            with col_factors: 
                st.write('**Key Recommendation Factors**')

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
                            st.image(image_product_category_dict[product_category], width=100)
                        with subcol_description:
                            subcol_badge, subcol_product_name = st.columns([1,6])
                            with subcol_badge:
                                st.badge(f'#{j+1}', color='blue', width=100)
                            with subcol_product_name:
                                #st.write(f'**{product_id}**')
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
                            st.image(image_product_category_dict[product_category], width=100)
                        with subcol_description:
                            subcol_badge, subcol_product_name = st.columns([1,6])
                            with subcol_badge:
                                st.badge(f'#{j+1}', color='blue', width=100)
                            with subcol_product_name:
                                #st.write(f'**{product_id}**')
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
            # render display
            with st.container(border=True):
                st.subheader(f"Custom display - Product {product_id}")
                customize_col1, customize_col2 = st.columns([1,1])

                with customize_col1:
                    for i in range(int(np.floor((product_by_display)/2))):
                        with st.container(border=True):
                            subcol_image, subcol_description = st.columns([1,4])
                            with subcol_image:
                                st.image(
                                    image_product_category_dict[
                                        products_df[products_df['ProductID']==recommended_product_ids[2*i]]['Category'].iloc[0]
                                    ],
                                    width=100
                                )
                            with subcol_description:
                                subcol_badge, subcol_product_name = st.columns([1,6])
                                with subcol_badge:
                                    st.badge(f'#{2*i+1}', color='blue', width=100)
                                with subcol_product_name:
                                    st.write(f'**{recommended_product_ids[2*i]}**')
                                st.caption(
                                    f'{products_df[products_df["ProductID"]==recommended_product_ids[2*i]]["FamilyLevel1"].iloc[0]}'
                                )
                                subcol_metric1, subcol_metric2 = st.columns([1,1])
                                with subcol_metric1:
                                    with st.container(border=True):
                                        subcol_caption, subcol_value = st.columns([1,1])
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
                                product_stock = np.random.randint(1,20)
                                color = "green" if product_stock > 5 else "yellow" if product_stock > 2 else "red"
                                st.badge(f'{product_stock} units', icon=":material/package_2:", color=color)

                with customize_col2:
                    for i in range(int(np.floor((product_by_display)/2))):
                        with st.container(border=True):
                            subcol_image, subcol_description = st.columns([1,4])
                            with subcol_image:
                                st.image(
                                    image_product_category_dict[
                                        products_df[products_df['ProductID']==recommended_product_ids[2*i+1]]['Category'].iloc[0]
                                    ],
                                    width=100
                                )
                            with subcol_description:
                                subcol_badge, subcol_product_name = st.columns([1,6])
                                with subcol_badge:
                                    st.badge(f'#{2*i+2}', color='blue', width=100)
                                with subcol_product_name:
                                    st.write(f'**{recommended_product_ids[2*i+1]}**')
                                st.caption(
                                    f'{products_df[products_df["ProductID"]==recommended_product_ids[2*i+1]]["FamilyLevel1"].iloc[0]}'
                                )
                                subcol_metric1, subcol_metric2 = st.columns([1,1])
                                with subcol_metric1:
                                    with st.container(border=True):
                                        subcol_caption, subcol_value = st.columns([1,1])
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
                                product_stock = np.random.randint(1,10)
                                color = "green" if product_stock > 5 else "yellow" if product_stock > 2 else "red"
                                st.badge(f'{product_stock} units', icon=":material/package_2:", color=color)

##########################################################################################
################################## Mock data generation ##################################

# -------------------------------
# Mock Data Generation
# -------------------------------
np.random.seed(42)

# Products
n_products = 50
products = pd.DataFrame({
    'ProductID': range(1, n_products+1),
    'Name': [f'Product {i}' for i in range(1, n_products+1)],
    'Category': np.random.choice(['Shoes', 'Shirts', 'Balls', 'Accessories'], n_products)
})

# Stores
n_stores = 10
stores = pd.DataFrame({
    'StoreID': range(1, n_stores+1),
    'StoreName': [f'Store {i}' for i in range(1, n_stores+1)],
})

# Dates
dates = pd.date_range(start='2026-01-01', end='2026-01-31')

# Recommendations table (top recommended products per store)
recommendations = []
for store_id in stores['StoreID']:
    rec_products = np.random.choice(products['ProductID'], size=10, replace=False)
    for p in rec_products:
        recommendations.append({
            'StoreID': store_id,
            'ProductID': p,
            'Score': np.round(np.random.uniform(0.5, 1.0), 2)
        })
recommendations_df = pd.DataFrame(recommendations)
recommendations_df = recommendations_df.merge(products, on='ProductID', how='left')

# Clicks and conversions per day
click_data = []
for date in dates:
    for p in products['ProductID']:
        clicks = np.random.poisson(lam=20)
        purchases = np.random.binomial(n=clicks, p=0.2)
        click_data.append({
            'Date': date,
            'ProductID': p,
            'Clicks': clicks,
            'Purchases': purchases
        })
clicks_df = pd.DataFrame(click_data)
clicks_df = clicks_df.merge(products, on='ProductID', how='left')

# Conversion rate per day
conversion_df = clicks_df.groupby('Date').agg({
    'Clicks':'sum',
    'Purchases':'sum'
}).reset_index()
conversion_df['ConversionRate'] = conversion_df['Purchases'] / conversion_df['Clicks']

# A/B test mock (5 stores recommended vs 5 control)
ab_data = []
for p in products['ProductID']:
    for store_id in stores['StoreID']:
        group = 'A' if store_id <=5 else 'B'
        units_sold = np.random.poisson(lam=15 if group=='A' else 10)
        ab_data.append({
            'ProductID': p,
            'StoreID': store_id,
            'Group': group,
            'UnitsSold': units_sold
        })
ab_df = pd.DataFrame(ab_data)
ab_df = ab_df.merge(products, on='ProductID', how='left')

# Recommendation coverage (percentage of products recommended)
coverage_df = pd.DataFrame({
    'StoreID': stores['StoreID'],
    'TotalProducts': n_products,
    'RecommendedProducts': np.random.randint(5, 15, n_stores)
})
coverage_df['CoveragePercent'] = coverage_df['RecommendedProducts'] / coverage_df['TotalProducts'] * 100
coverage_df = coverage_df.merge(stores, on='StoreID', how='left')

##########################################################################################
############################# Tab 3 : Monitoring Dashboard ###############################

with tab3:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        selected_store = st.selectbox("Select Store", stores['StoreName'])

    with col_ctrl2:
        selected_product = st.selectbox("Select Product", products['Name'])

    with col_ctrl3:
        top_n = st.number_input("Top N Products", min_value=1, max_value=20, value=10)

    # -------------------------------
    # Layout: 2 columns for charts
    # -------------------------------
    col1, col2 = st.columns(2)

    # -------------------------------
    # Column 1
    # -------------------------------
    with col1:
        st.subheader(f"Top Recommended Products for {selected_store}")
        store_id = stores.loc[stores['StoreName']==selected_store, 'StoreID'].values[0]
        top_recs = recommendations_df[recommendations_df['StoreID']==store_id].sort_values('Score', ascending=False)
        st.dataframe(top_recs[['ProductID','Name','Category','Score']].head(top_n), use_container_width=True)
        
        st.subheader("Conversion Rate Over Time")
        fig_conv = px.line(conversion_df, x='Date', y='ConversionRate', title="Daily Conversion Rate")
        st.plotly_chart(fig_conv, use_container_width=True)

    # -------------------------------
    # Column 2
    # -------------------------------
    with col2:
        st.subheader("Clicks on Recommended Products Over Time")
        top_product_ids = recommendations_df[recommendations_df['StoreID']==store_id]['ProductID'].tolist()
        clicks_plot_df = clicks_df[clicks_df['ProductID'].isin(top_product_ids)].groupby('Date').agg({'Clicks':'sum'}).reset_index()
        fig_clicks = px.line(clicks_plot_df, x='Date', y='Clicks', title="Total Clicks on Recommended Products")
        st.plotly_chart(fig_clicks, use_container_width=True)
        
        st.subheader(f"A/B Test Units Sold for {selected_product}")
        product_id = products.loc[products['Name']==selected_product, 'ProductID'].values[0]
        ab_plot_df = ab_df[ab_df['ProductID']==product_id].groupby(['Group','StoreID']).agg({'UnitsSold':'sum'}).reset_index()
        fig_ab = px.line(ab_plot_df, x='StoreID', y='UnitsSold', color='Group', markers=True,
                        title=f"Units Sold by Store (Group A vs B) for {selected_product}")
        st.plotly_chart(fig_ab, use_container_width=True)

    # -------------------------------
    # Full width charts below columns
    # -------------------------------
    st.subheader("Conversion Funnel for Recommended Products")
    funnel_total_clicks = clicks_df['Clicks'].sum()
    funnel_total_added = clicks_df['Clicks'].sum() * 0.5  # mock added to basket
    funnel_total_purchases = clicks_df['Purchases'].sum()
    fig_funnel = go.Figure(go.Funnel(
        y = ["Recommended", "Added to Basket", "Purchased"],
        x = [funnel_total_clicks, funnel_total_added, funnel_total_purchases]
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.subheader("Recommendation Coverage by Store")
    fig_cov = px.bar(coverage_df, x='StoreName', y='CoveragePercent', title="Percentage of Products Recommended per Store")
    st.plotly_chart(fig_cov, use_container_width=True)















st.balloons()