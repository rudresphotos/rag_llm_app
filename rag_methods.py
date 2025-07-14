import os
import dotenv
import streamlit as st

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

dotenv.load_dotenv()

DB_DOCS_LIMIT = 10

def stream_llm_response(llm_stream, messages):
    response = ""
    for chunk in llm_stream.stream(messages):
        response += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response})


def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(doc_file.read())
                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Unsupported type: {doc_file.type}")
                            continue
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    finally:
                        os.remove(file_path)
        if docs:
            _split_and_load_docs(docs)
            st.toast("Document loaded.", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        if url not in st.session_state.rag_sources:
            docs = WebBaseLoader(url).load()
            st.session_state.rag_sources.append(url)
            _split_and_load_docs(docs)
            st.toast(f"Loaded URL: {url}", icon="✅")


def initialize_vector_db(docs):
    embedding = (
        AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        if os.getenv("AZURE_OPENAI_API_KEY")
        else OpenAIEmbeddings()
    )
    db = FAISS.from_documents(docs, embedding)
    db.save_local("faiss_index")
    return db


def _split_and_load_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = splitter.split_documents(docs)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(chunks)
    else:
        st.session_state.vector_db.add_documents(chunks)
        st.session_state.vector_db.save_local("faiss_index")


def stream_llm_rag_response(llm_stream, messages):
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    retriever = st.session_state.vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm_stream, prompt)
    )
    response = "*(RAG)*\n"
    for chunk in chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response})
