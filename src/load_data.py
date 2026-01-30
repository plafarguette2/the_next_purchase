import numpy as np 
import pandas as pd
from config import DATADIRPATH

def load_clients_data(datadirpath:str=DATADIRPATH) -> pd.DataFrame : 
    return pd.read_csv(f"{datadirpath}/clients.csv")

def load_products_data(datadirpath:str=DATADIRPATH) -> pd.DataFrame : 
    return pd.read_csv(f"{datadirpath}/products.csv")

def load_stocks_data(datadirpath:str=DATADIRPATH) -> pd.DataFrame : 
    return pd.read_csv(f"{datadirpath}/stocks.csv")

def load_stores_data(datadirpath:str=DATADIRPATH) -> pd.DataFrame : 
    return pd.read_csv(f"{datadirpath}/stores.csv")

def load_transactions_data(datadirpath:str=DATADIRPATH) -> pd.DataFrame : 
    return pd.read_csv(f"{datadirpath}/transactions.csv")

### mock data

def load_web_sessions_mock_data() -> pd.DataFrame :
    return pd.read_csv("mock_data/web_sessions.csv")