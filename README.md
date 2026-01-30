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
\---lib
        load_data.py

```

## Running the Streamlit app

**Option 1: Using Docker**

Clone this repository:
```bash 
git clone https://github.com/plafarguette2/URW_Data_challenge.git
cd URW_Data_challenge
```

Create a ```/data``` folder at root and add data files provided by URW and geodata files sent by our team.


Build the Docker image:
```bash
docker build -t urw_mall_app .
```

Run the container:
```bash
docker run -d -p 0.0.0.0:8080:8501 urw_mall_app
```

Open your browser and go to:

http://localhost:8080/

**Option 2: Run locally without Docker**

After cloning the repository, adding data and installing dependencies, simply run:
```bash
streamlit run urw_mall_app.py
```
