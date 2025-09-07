from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import streamlit as st

load_dotenv()

# load nvidia api key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")


def create_vector_embeddings():
  if "vectordb" not in st.session_state:
    st.session_state.embeddings = NVIDIAEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(documents=st.session_state.docs)
    st.session_state.vectordb = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

st.title("RAG Document QA with Nvidia NIM")

prompt = ChatPromptTemplate.from_template(
  """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
If you don't know the answer, say i don't know. dont try to create
from your side.
<context> {context} <context>
Question : {input}
"""
)



if st.button("Document Embedding"):
  create_vector_embeddings()
  st.write("Vector Store DB is ready")


user_input = st.text_input("Please Enter your question from the document")

if user_input:
    if "vectordb" not in st.session_state:
       st.warning("Please click 'Document Embedding' first to load the documents.")
    else:
        document_chain = create_stuff_documents_chain(llm,prompt)
        retrieval = st.session_state.vectordb.as_retriever()
        retrieval_chain = create_retrieval_chain(retrieval,document_chain)
        response = retrieval_chain.invoke({"input":user_input})
        st.write(response['answer'])



        # with streamlit expander
        with st.expander("Document Similarity Search"):
        #find the relevant chunks
            for i, docs in enumerate(response["context"]):
                st.write(docs.page_content)
                st.write("--------------")
            



