from collections import namedtuple
import sys
import altair as alt
import math
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from typing import List

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
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Constants
project_id = 'appbuilder-388321'
location = 'us-central1'

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)
vertexai.init(project=project_id, location=location, credentials=credentials)

# Vertex Utility Functions
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    print("Waiting")
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens = 400,
    temperature = 0.2,
    top_p = 0.8,
    top_k = 40,
    verbose = True,)
####################################### PART 1 ########################################
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
########################################## PART 2 ######################################
st.divider()
st.subheader("Q&A with RetrievalQA Chain")

@st.cache_data(ttl=600)
df_qa = pd.read_csv('/Users/liuchristie/Projects/streamlit/streamlit-example/output2.csv')
st.dataframe(df_qa)
loader = DataFrameLoader(df_qa, page_content_column="text")

documents = loader.load()

@st.cache_data(ttl=600)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=0)

texts = text_splitter.split_documents(documents)

db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

def ask_question(question):
    template = """Use the provided context to answer the input question.
    {context}

    Question: {question}
    Answer: """
    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    st.info(qa.run(question))

with st.form('qaForm'):
    question = st.text_input('Question:','')
    submitted = st.form_submit_button('Submit')
    if submitted:
        ask_question(question)

code2 = '''
def ask_question(question):
  template = """Use the provided context to answer the input question. 
  {context}

  Question: {question}
  Answer: """
  prompt = PromptTemplate(
      template=template, input_variables=["context", "question"]
  )
  chain_type_kwargs = {"prompt": prompt}
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)
  qa.run(question)
  '''
st.code(code2, language='python')

