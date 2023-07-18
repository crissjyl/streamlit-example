from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Constants
project_id = 'appbuilder-388321'
location = 'us-central1'

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

st.title("Sentiment Analysis of Amazon Product Reviews")    


@st.cache_data(ttl=600)
def run_query(query):
    query_job = client.query(query)
    raw_rows = query_job.result()
    rows = [dict(row) for row in raw_rows]
    return rows
rows = run_query("SELECT * FROM amazon_product_reviews.sentiment ORDER BY Product_Name")

df = pd.DataFrame(rows)
st.dataframe(df)
    

