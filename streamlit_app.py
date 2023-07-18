from collections import namedtuple
import sys
import altair as alt
import math
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
# Langchain
import langchain
from langchain.prompts import PromptTemplate

# Vertex AI
from google.cloud import aiplatform
import vertexai
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
vertexai.init(project=project_id, location=location, credentials=credentials)

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
    
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens = 400,
    temperature = 0.2,
    top_p = 0.8,
    top_k = 40,
    verbose = True,
)

def aste(review):
    template = """
    Perform Aspect Sentiment Triplet Extract task. Given {review}, tag all (aspect, opinion, sentiment) triplets. Aspect and opinion should be substring of the sentence. Sentiment should be selected from ['negative', 'neutral', 'positive'].
    Return a list containing three strings. Return the list only, without any other comments or texts.\n

    review: Material is flimsy and cheap.
    label:('material', 'filmsy and cheap', 'negative')
        
    review: I'm afraid to ride it. The seat remains wobbly after many attempts of tightening it.
    label: ('seat', 'wobbly', 'negative')

    review: {review}
    label:
    """
    prompt = PromptTemplate(
        input_variables = ['review'],
        template = template,)
    final_prompt = prompt.format(review=review)
    st.info(llm(final_prompt))

with st.form('promptForm'):
    review = st.text_input('Prompt:','')
    submitted = st.form_submit_button('Submit')
    if submitted:
        aste(review)

st.markdown("model_name: red[text-bison@001]")
st.markdown("max_output_tokens: red[400]")
st.markdown("temperature: red[0.2]")
st.markdown("top_p: red[0.8]")
st.markdown("top_k: red[40]")

code = '''
    template = """
    Perform Aspect Sentiment Triplet Extract task. Given {review}, tag all (aspect, opinion, sentiment) triplets. Aspect and opinion should be substring of the sentence. Sentiment should be selected from ['negative', 'neutral', 'positive'].
    Return a list containing three strings. Return the list only, without any other comments or texts.\n

    review: Material is flimsy and cheap.
    label:('material', 'filmsy and cheap', 'negative')
        
    review: I'm afraid to ride it. The seat remains wobbly after many attempts of tightening it.
    label: ('seat', 'wobbly', 'negative')

    review: {review}
    label:
    """
    prompt = PromptTemplate(
        input_variables = ["review"],
        template = template,)
    final_prompt = prompt.format(review=review)
    return llm(final_prompt)'''
st.code(code, language='python')




