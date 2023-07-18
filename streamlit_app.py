from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
# Langchain
from langchain.prompts import PromptTemplate

# Vertex AI
from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.schema import HumanMessage, SystemMessage



# Constants
project_id = 'appbuilder-388321'
location = 'us-central1'

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)
vertexai.init(project=project_id, location=location)
st_callback = StreamlitCallbackHandler(st.container())

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
    

