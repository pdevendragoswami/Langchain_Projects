import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.tools import Tool


# Setup the streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver")

api_key = st.sidebar.text_input("Please provide your api key :", type="password")

if not api_key:
    st.warning("Please provide the Groq API Key")
    st.stop()
else:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)

    # initializing the tools
    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = Tool(name="WikiPedia", func=wikipedia_wrapper.run,
                          description="A tool for searching the internet to find the various information on the topics mentioned")
    
    MATH_PROMPT = PromptTemplate.from_template(
    "Translate the following math problem into a pure Python arithmetic expression.\n"
    "Only return the expression, no code, no imports, no eval, no explanation.\n\n"
    "Question: {question}\nExpression:"
)
    
    python_tool = PythonAstREPLTool()

    math_chain = LLMMathChain.from_llm(llm=llm,prompt= MATH_PROMPT, verbose = True)
    calculator_tool = Tool(name="calculator",func=python_tool.run,
                           description="A tool for answering the math related question. only input mathematical expression need to be provided")
    
    prompt_template = """
    You are a agent tasked for solving the user's mathematical question. Logically arrive at the solution and provide a detailed explaination
    and display it pointwise for the question below. \n 
    Question:{question}
    Answer :
    """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template
    )

    # combine all the tools inot chain
    chain = prompt | llm

    reasoning_tool = Tool.from_function(func=lambda x: chain.invoke({"question": x}).content,
                                        name="Reasoning tool",description="A tool for answering logic based and reasoning questions")

    # initialize tha agents

    assistant_agents = initialize_agent(tools=[wikipedia_tool, calculator_tool, reasoning_tool],
                                        llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                        verbose = False,handle_parsing_errors =True)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role":"assistant","content":"Hey I am a math chatbot who can answer all your maths questions."}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # interaction
    user_input = st.text_input("Enter your question here:", "I have 5 bananas and 7 grapes. I ate 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 pack of blueberries. Each pack of blueberries contain 25 berries. How many total pieces of fruit do I have at the end?")

    if st.button("find my answer"):
        if user_input:
            with st.spinner("Generate response..."):
                st.session_state.messages.append({"role":"user","content":user_input})
                st.chat_message("user").write(user_input)

                st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
                response = assistant_agents.invoke(user_input,callbacks = [st_cb])
                st.session_state.messages.append({"role":"assistant","content":response["output"]})
                st.write("Response :")
                st.success(response["output"])

        else:
            st.info("Please provide the question first...")


