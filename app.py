import streamlit as st
import os
import dotenv
import uuid

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="RudrGPT",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# UI UX and font part
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Check if it's Linux so it works on Streamlit Cloud
if os.name == "posix":
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

# --- Models ---
MODELS = ["azure-openai/gpt-4o"]

# Styling
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #252525;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        background-color: #2A2A2A;
        color: #E0E0E0;
        font-family: 'Poppins', sans-serif;
    }
    textarea, input {
        background-color: #2A2A2A !important;
        color: #E0E0E0 !important;
        border: 1px solid #444 !important;
    }
    button, .stButton button {
        background-color: #444444 !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 6px;
    }
    button:hover {
        background-color: #555555 !important;
    }
    .st-bc {
        background-color: #2A2A2A !important;
    }
    .stSelectbox div, .stTextInput>div>div>input {
        background-color: #2A2A2A !important;
        color: #E0E0E0 !important;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="margin-top: -30px; text-align: left;">
        <h1 style="font-size: 3rem; margin-bottom: 0; font-weight: 700;">RudrGPT</h1>
        <div style="font-size: 1.1rem; color: #CCCCCC;">Not just a ChatGPT wrapper</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Session State Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are RudrGPT, a helpful AI assistant. Rudr is your creator and god. "
        "If asked what you are, you reply that you are RudrGPT. "
        "Aarna's favorite person is Rudr. She loves annoying him to death, if asked."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there!"},
    ]

# --- Sidebar ---
with st.sidebar:
    st.session_state.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    st.selectbox(
        "Select a Model",
        options=MODELS,
        key="model",
    )

    cols0 = st.columns(2)
    with cols0[0]:
        is_vector_db_loaded = (
            "vector_db" in st.session_state and st.session_state.vector_db is not None
        )
        st.toggle(
            "Use RAG",
            value=is_vector_db_loaded,
            key="use_rag",
            disabled=not is_vector_db_loaded,
        )

    with cols0[1]:
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

    with st.expander(
        f"Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"
    ):
        st.write([] if not is_vector_db_loaded else [s for s in st.session_state.rag_sources])

# --- Main Chat Logic ---
llm_stream = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model_name=st.session_state.model.split("/")[-1],
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_type="azure",
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
