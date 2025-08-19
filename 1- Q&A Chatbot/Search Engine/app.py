import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")


st.title("Langchain - Chat with search")
"""
In this example, we're using `StreamlitCalbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more Langchain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, I'm a chatbot who can search on the web. How Can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(str(msg["content"]))

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key, model_name = "Llama3-8b-8192", streaming=True )
    tools = [wiki,arxiv,search]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.invoke(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.text(response)
