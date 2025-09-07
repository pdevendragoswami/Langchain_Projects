import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please provide the response to the user queries"),
        ("user","Question:{question}")

    ]
)

def generate_response(question,llm,temperature,max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer



# title for the app
st.title("Enhanced Q&A Chatbot With Ollama")

# Sidebar for the settings
st.sidebar.title("Settings")

## Drop down to select the various Open AI llm model
llm = st.sidebar.selectbox("Select an Ollama model",["mistral"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=350, value=200)

# main interface for user input
st.write("GO ahead and ask your questions")
user_input = st.text_input("You :")

if user_input:
    response =  generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)

else:
    st.write("Please provide the query")
