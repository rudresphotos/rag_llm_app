import os
import shutil
import streamlit as st
import re
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_DOCS_LIMIT = 10

# --- Embedding Model ---
def get_embedding_model():
    return AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    )

# --- Split Documents ---
def _split_and_load_docs(docs):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)

# --- FAISS: Create and Save Vector DB ---
def create_faiss_from_documents(chunks, collection_name):
    persist_dir = f"faiss_dbs/{collection_name}"
    os.makedirs(persist_dir, exist_ok=True)
    db = FAISS.from_documents(chunks, embedding=get_embedding_model())
    db.save_local(persist_dir)
    st.session_state.vector_db = db

# --- FAISS: Load Existing DB ---
def load_vector_db(collection_name):
    persist_dir = f"faiss_dbs/{collection_name}"
    index_path = os.path.join(persist_dir, "index.faiss")
    embedding = get_embedding_model()

    if os.path.exists(index_path):
        db = FAISS.load_local(
            persist_dir,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        st.session_state.vector_db = db
    else:
        st.warning(f"⚠️ Vector DB for '{collection_name}' not found. Upload documents first.")
        st.session_state.vector_db = None

# --- List all .txt files in docs/ ---
def list_docs_files():
    import glob
    files = glob.glob("docs/*.txt")
    return [os.path.basename(f) for f in files]

# --- Load a single file from docs/ ---
def load_single_doc_file(filename):
    collection_name = filename.rsplit(".", 1)[0]
    file_path = os.path.join("docs", filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if docs:
        chunks = _split_and_load_docs(docs)
        create_faiss_from_documents(chunks, collection_name)
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        if filename not in st.session_state.rag_sources:
            st.session_state.rag_sources.append(filename)

# --- Load Uploaded Files ---
def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        loaded = loader.load()
                        docs += loaded
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")

                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            chunks = _split_and_load_docs(docs)
            create_faiss_from_documents(chunks, "user_uploads")
            st.toast(
                f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.",
                icon="✅"
            )

# --- Load from URL ---
def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    loaded = loader.load()
                    docs += loaded
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    chunks = _split_and_load_docs(docs)
                    create_faiss_from_documents(chunks, "url_uploads")
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")
            else:
                st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

# --- RAG Chain Construction ---
def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are RudrGPT, a helpful assistant. You will have to answer user's queries.
         You will have some context to help with your answers, but not always completely related or helpful.
         You can also use your knowledge to assist answering the user's queries.\n
         {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = ""
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})
