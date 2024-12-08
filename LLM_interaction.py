import os
from tkinter.messagebox import QUESTION
from openai import file_from_path
import streamlit as st
import pickle as pk
import time
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
from torch import embedding
load_dotenv()


st.title('Support KB Articles read 📚')
main_placefolder = st.empty()

st.sidebar.title("Enter your KB article link")
links = st.sidebar.text_input("URL").strip()
print(f"Link : {links}")
vector_path = "Faiss_vb.pkl"
process_click = st.sidebar.button("Process URL")

llm = OpenAI(temperature = 0.5)

if process_click:
    loader = UnstructuredURLLoader(urls=[links])
    main_placefolder.text("Data loading... processing..")
    data = loader.load()
    
    #split the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ['\n\n','\n','.',','],
        chunk_size = 1000
    )
    main_placefolder.text("Splitting the data... processing..")
    docs = text_splitter.split_documents(data)

    print(f"Number of documents: {len(data)}")
    print(f"First document: {docs[0] if docs else 'No documents found'}")

    #Create Embedding
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Building embedding Vector DB... processing..")
    time.sleep(2)

    #save FAISS in pkl format
    with open (vector_path,"wb") as f:
        pk.dump(vector_db, f)

query = main_placefolder.text_input("Question :").strip()

if query:
    if os.path.exists(vector_path):
        with open(vector_path, "rb") as f:
            vectorstore = pk.load(f)
            chain = RetrievalQA.from_llm(llm = llm, retriever = vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs = True)
            st.header("Answer")
            st.subheader(result["answer"])