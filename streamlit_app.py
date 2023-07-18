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
from langchain.chains import RetrievalQA



# Constants
project_id = 'appbuilder-388321'
location = 'us-central1'

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)
vertexai.init(project=project_id, location=location, credentials=credentials)

st.set_page_config(layout="wide")
st.title("Sentiment Analysis of Amazon Product Reviews")    
st.subheader("Prompt Templates: Ask LLM to perform Aspect Sentiment Triplet Extract task")

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
    review = st.text_input('Review:','')
    submitted = st.form_submit_button('Submit')
    if submitted:
        aste(review)

code = '''
    model_name="text-bison@001",
    max_output_tokens = 400,
    temperature = 0.2,
    top_p = 0.8,
    top_k = 40,
    verbose = True

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

st.divider()
st.subheader("Q&A: Provide Context in the Prompt")


def ask_question(question):
    prompt= f"""
    Consider your input context the following context that is delimited by triple backticks with: 

    Sentiment should be based on the results_text column and your response should be selected from ['negative', 'neutral', 'positive'].
    If asked about aspects, analyze the results_text column values, perform an Aspect Sentiment Triplet Extract task and return aspects and opinions.

    Question: What are the top two negative aspects of "Adjustable Adult And Kids Bicycle Bike Training Wheels Fits 24" to 28" reviews?
    Answer: The U shaped parts are too tight and the price is too high.

    Question: {question}
    Answer:
    """
    st.info(llm.predict(prompt))

with st.form('qaForm'):
    question = st.text_input('Question:','')
    submitted = st.form_submit_button('Submit')
    if submitted:
        ask_question(question)

code2 = '''
def ask_question(question):
    prompt = """
    Use {df} table to answer question about sentiment about products. 
    Sentiment should be based on the results_text column and your response should be selected from ['negative', 'neutral', 'positive'].
    If asked about aspects, analyze the results_text column values, perform an Aspect Sentiment Triplet Extract task and return aspects and opinions.

    Question: What are the top two negative aspects of "Adjustable Adult And Kids Bicycle Bike Training Wheels Fits 24" to 28" reviews?
    Answer: The U shaped parts are too tight and the price is too high.

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables = ['question'],
        template = template,)
    final_prompt = prompt.format(question=question)
    st.info(llm(final_prompt))
'''
st.code(code2, language='python')