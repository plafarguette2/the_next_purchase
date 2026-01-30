## The_next_purchase

This repository implements a **Customer Purchase Recommendation System**, it was built during the 2026 Eleven - HEC data commercial proposal by team 4 composed of : 
- Harasym Martyna
- Ochterbeck Maxim
- Lafarguette Pierre
- Salerno Marco
- Sijelmassi Iliass
- Capizzi Simone

It provides a **Streamlit interface** where the user can explore : 
- A **customer-level** product **recommendation engine** for both online and in-store sales.
- A **store-level** product **recommendation engine** for in-store sales.
- A **dashboard** for **monitoring recommendation engines performance**.

This README describes how to run the Streamlit app locally.

```arduino 
the_next_purchase
|   next_product_app.py
|   config.py
|   mock_data_generator.py
|   modeling.ipynb
|   requirements.txt
|   
+---data
|       (your data files go here)
|
+---mock_data
|       web_sessions.csv
|
+---images
|       (images for the app)
|            
\---src
        load_data.py
        store_recommendations.py
        \---modeling
                artifacts.py
                business.py
                config.py
                data_prep.py
                inference.py
                pipeline.py
                reranker_data.py
                reranker_model.py
                retrieve_als.py

```

## Running the Streamlit app

Clone this repository:
```bash 
git clone https://github.com/plafarguette2/the_next_purchase.git
cd the_next_purchase
```

Create a ```/data``` folder at root and add data files provided by Eleven.

Install all dependencies :
```bash 
pip install -r requirements.txt
```
**Launch the training pipeline** to create the models by running the notebook modeling.ipynb

Then simply run:
```bash
streamlit run next_product_app.py
```
and go to http://localhost:8501/
