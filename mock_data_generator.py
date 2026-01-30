from src.load_data import load_clients_data, load_stocks_data
import pandas as pd
import numpy as np

np.random.seed(42)
nb_sessions = 50_000
reco_by_session = 5

clients_df = load_clients_data()
stocks_df = load_stocks_data()

client_ids = clients_df['ClientID'].sample(25000, replace=False).values
countries = clients_df.set_index('ClientID')['ClientCountry'].to_dict()

available_products = list(stocks_df[stocks_df['Quantity'] > 0]['ProductID'].unique())



dates = pd.date_range(start="2026-01-01 00:00:00", end="2026-01-30 23:59:59", freq="S")

rows = []
session_counter = 12344234

for _ in range(nb_sessions):
    session_id = f"S{session_counter}"
    session_counter += 1
    
    client_id = np.random.choice(client_ids)
    country = countries.get(client_id, "USA")
    session_date = np.random.choice(dates)

    n_recos = reco_by_session
    reco_products = np.random.choice(available_products, size=n_recos, replace=False)
    
    for product_id in reco_products:
        is_clicked = np.random.binomial(1, 0.07)
        
        
        is_bought = np.random.binomial(1, 0.30) if is_clicked else 0
        
        rows.append({
            "session_id": session_id,
            "client_id": client_id,
            "country": country,
            "product_id": product_id,
            "is_clicked": is_clicked,
            "is_bought": is_bought,
            "session_date": session_date
        })

web_sessions_df = pd.DataFrame(rows)

web_sessions_df.to_csv(f'mock_data/web_sessions.csv', sep=',', header=True, index=False)
