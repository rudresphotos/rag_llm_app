import streamlit as st
import os
import dotenv
import uuid

st.set_page_config(
    page_title="RudrGPT",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

MODELS = ["azure-openai/gpt-4o"]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are RudrGPT, a helpful AI assistant."

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello there!"}]

# Load FAISS index if exists
if "vector_db" not in st.session_state:
    if os.path.exists("faiss_index"):
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
        st.session_state.vector_db = FAISS.load_local(
            "faiss_index",
            embedding,
            allow_dangerous_deserialization=True
        )

with st.sidebar:
    st.selectbox("Select a Model", options=MODELS, key="model")

    is_vector_db_loaded = (
        "vector_db" in st.session_state and st.session_state.vector_db is not None
    )
    st.toggle(
        "Use RAG",
        value=is_vector_db_loaded,
        key="use_rag",
        disabled=not is_vector_db_loaded,
    )

    st.button(
        "Clear Chat",
        on_click=lambda: st.session_state.messages.clear(),
        type="primary",
    )

    st.header("RAG Sources:")

    st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )

    st.text_input(
        "Introduce a URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    with st.expander(f"Documents in DB ({len(st.session_state.rag_sources)})"):
        st.write(st.session_state.rag_sources)

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")

if not azure_api_key or not azure_endpoint:
    st.error("Azure API key or endpoint not set.")
    st.stop()

llm_stream = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    deployment_name="gpt-4o",
    api_key=azure_api_key,
    temperature=0.3,
    streaming=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            SystemMessage(content=st.session_state.system_prompt),
            *(
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ),
        ]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
