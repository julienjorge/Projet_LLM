from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
#from langchain import hub
from dotenv import load_dotenv
import os
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader


client = weaviate.client("http://localhost:8080")

loader = DirectoryLoader("/Users/AI/Desktop/text_parsing", glob="**/*.txt", loader_cls = TextLoader, loader_kwargs={"encoding": "utf-8"}) 

docs = loader.load()


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")   
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=300,
    chunk_overlap=50
    )   
splitted_docs = splitter.split_documents(docs)



#client = weaviate.connect_to_local(
#    host="localhost", 
#    port=8080,
#    grpc_port=50051,
#)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")




weaviate.from_documents(splitted_docs, 
    embeddings, 
    client=client,
    index_name="projet_LLM"
)

