import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain: Chat with SQL DB")
st.title("Langchain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database - Student.db", "Connect to your MySQL Database"]

selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide my Sql hostname")
    mysql_user = st.sidebar.text_input("MySQL_user")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API Key", type="password")

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host = None, mysql_user = None, mysql_password = None,mysql_db = None):
    if db_uri == LOCALDB:
        db_filepath = (Path(__file__).parent/"student.db").absolute()
        print(db_filepath)
        creator = lambda : sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    
    elif db_uri == MYSQL:
        if not(mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MYySQL connnection details")
            st.stop()

        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if not db_uri:
    st.info("Please enter the database information and uri")

# only create LLM and agent if API key is provided
if not api_key:
    st.info("Please add the Groq API Key")
else:
    llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key = api_key, streaming=True)

    if db_uri==MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password,mysql_db)
    else:
        db = configure_db(db_uri)

    ## toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm = llm)

    agent = create_sql_agent(llm=llm,toolkit=toolkit, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)


    # chat history
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role":"assistant","content":"How can I help you?"}]


    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # user query
    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role":"user","content":user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            print("Available tables:", db.get_usable_table_names())
            print(db.get_table_info(["student"]))
            response = agent.invoke({"input":user_query},callbacks=[streamlit_callback])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)
