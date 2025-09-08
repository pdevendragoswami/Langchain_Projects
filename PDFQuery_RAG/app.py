from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()


loader = PyPDFLoader("apjspeech.pdf")
docs = loader.load()
#print(f"Docs -: {docs} \n\n\n\n")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_docs = text_splitter.split_documents(docs)
#print(f"final_docs:{final_docs}")

embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectordb = Chroma.from_documents(embedding=embedding,documents=docs)
question = input("What is your query?")
result =  vectordb.similarity_search(query=question)
print(result)





