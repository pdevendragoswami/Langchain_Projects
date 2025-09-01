import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit app setup
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL:")

# get the groq api key
api_key = st.sidebar.text_input("Please Enter your Groq API Key", type="password")

# URL input
url = st.text_input("URL", label_visibility="collapsed")

map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Write a short summary of the following:\n\n{text}"
)
combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="Combine the following summaries into a coherent summary of about 300 words:\n\n{text}"
)


if api_key and url:
    if st.button("Summarize the content"):
        if not api_key.strip() or not url.strip():
            st.error("Please provide the information to get start")
        elif not validators.url(url):
            st.error("Please enter a valid URL (YouTube or website).")
        else:
            try:
                with st.spinner("Fetching and summarizing content..."):
                    # Initialize LLM
                    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

                    ##loading the website/youtube video data
                    if "youtube.com" in url:
                        loader = YoutubeLoader.from_youtube_url(url,add_video_info=True,language="en")
                    else:
                        loader = UnstructuredURLLoader(urls=[url],ssl_verify=False)
                    
                    docs = loader.load()

                    if not docs:
                        st.error("Could not load transcript from the YouTube video.")
                        st.stop()

                    else:
                        # Split text into manageable chunks
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        split_docs = splitter.split_documents(docs)
                        # Optional: preview lengths
                        for i, doc in enumerate(split_docs):
                            st.write(f"Chunk {i+1} length:", len(doc.page_content))
                        # chain for summarization
                        chain = load_summarize_chain(llm=llm,chain_type="map_reduce",map_prompt=map_prompt,combine_prompt=combine_prompt)
                        output_summary = chain.invoke(split_docs)

                        st.success(output_summary["output_text"])
            except Exception as e:
                st.exception(f"Exception: {str(e)}")

elif api_key and not url:
    st.info("Please enter a URL to summarize.")
elif url and not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
else:
    st.info("Enter both a URL and your Groq API key to start.")

