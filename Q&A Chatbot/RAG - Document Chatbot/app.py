import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os


from dotenv import load_dotenv
load_dotenv()
## load the Groq API
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="Llama3-8b-8192")
#Or groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(model_name="Gemma-7b-It", groq_api_key = groq_api_key)

# prompt template
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data Ingestion Step
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 250)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents=st.session_state.docs[:100])
        st.session_state.vectorstore = FAISS.from_documents(documents=st.session_state.final_documents,embedding=st.session_state.embeddings)

st.title("RAG Document Q&A with Groq and Llama 3")

st.write("Please Start with the Document Embedding")
if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready...")


user_prompt = st.text_input("Enter your query from the research paper")


import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({"input":user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response["answer"])


    # with streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------")
