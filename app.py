import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
import os
from typing_extensions import Concatenate
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logo = Image.open('icon.png')
st.set_page_config(page_title = 'Indian Budget Query App', page_icon = logo)

st.image('icon.png')
st.markdown("<h1 style='text-align:center; color:black;'>Indian Budget Query App</h1>", unsafe_allow_html = True)

with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

cassio.init(token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'), database_id=os.getenv('ASTRA_DB_ID'))

llm = OpenAI()
embedding = OpenAIEmbeddings()

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="budgetSpeechPdfData",
    session=None,
    keyspace=None,
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


question = st.text_input('Please enter your question')
answer = astra_vector_index.query(question, llm=llm).strip()
st.write(answer)
