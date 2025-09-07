import streamlit as st
from langchain_openai import ChatOpenAI
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"


#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer


# title for the app
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar for the settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Open AI API Key:", type="password")

## Drop down to select the various Open AI llm model
llm = st.sidebar.selectbox("Select an Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=350, value=200)

# main interface for user input
st.write("GO ahead and ask your questions")
user_input = st.text_input("You :")

if user_input and api_key:
    response =  generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
elif user_input:
    st.write("Please provide the API Key.")
else:
    st.write("Please provide the query")
