import streamlit as st
import os
import uuid
import dotenv
import json
from datetime import datetime

# --- Load Environment Variables ---
dotenv.load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="RudrGPT",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
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
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="margin-top: -30px; text-align: left;">
        <h1 style="font-size: 3rem; margin-bottom: 0; font-weight: 700;">RudrGPT</h1>
        <div style="font-size: 1.1rem; color: #CCCCCC;">Ask anything about the ITeS/BPM industry</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Azure Storage ---
from storage_utils import (
    download_all_docs_to_local,
    upload_conversation_log
)

# --- RAG Imports ---
from rag_methods import (
    list_docs_files,
    load_single_doc_file,
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are RudrGPT, an intelligent assistant designed by Rudr. "
        "You specialize in providing insights about the ITES and BPM industry, "
        "particularly on the impact and applications of emerging technologies such as Generative AI. "
        "You are also capable of answering general questions with clarity and accuracy. "
        "When relevant documents are available, use them to ground your responses. "
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there!"}
    ]

# --- Load Pre-uploaded Docs from Azure only once ---
if "persistent_docs_loaded" not in st.session_state:
    # Clean docs folder first
    if os.path.exists("docs"):
        for f in os.listdir("docs"):
            os.remove(os.path.join("docs", f))
    else:
        os.makedirs("docs")

    # Download and load docs
    download_all_docs_to_local("docs")
    st.session_state.rag_sources = []
    st.session_state.vector_db = None

    for doc_file in os.listdir("docs"):
        file_path = os.path.join("docs", doc_file)
        if os.path.isfile(file_path):
            load_doc_to_db(file_path)

    st.session_state.persistent_docs_loaded = True

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Docs Collection")
    docs_files = list_docs_files()
    loaded_docs = [s for s in st.session_state.rag_sources if s in docs_files]
    selectable_docs = ["None"] + [doc for doc in docs_files if doc not in loaded_docs]
    selected_doc = st.selectbox("Select a file from docs", selectable_docs)

    if selected_doc != "None":
        load_single_doc_file(selected_doc)
        st.success(f"✅ Loaded: {selected_doc}")

    st.markdown("#### 📚 Loaded Docs")
    if st.session_state.rag_sources:
        for i, doc in enumerate(st.session_state.rag_sources, 1):
            st.write(f"{i}. {doc}")
    else:
        st.write("_No docs loaded yet._")

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
        "Upload a document (session only)",
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
        st.write([] if not is_vector_db_loaded else st.session_state.rag_sources)

# --- Validate Azure OpenAI Environment ---
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not azure_api_key or not azure_endpoint:
    st.error("Azure API key or endpoint not set. Please configure environment variables.")
    st.stop()

# --- LLM Client ---
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

llm_stream = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_version="2025-01-01-preview",
    model_name="gpt-4o",
    api_key=azure_api_key,
    temperature=0.1,
    streaming=True,
)

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
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

    # --- Save conversation log with timestamp ---
    log_json = json.dumps(st.session_state.messages, indent=2)
    timestamp = datetime.utcnow().isoformat().replace(":", "-")
    filename = f"{st.session_state.session_id}_{timestamp}.json"
    upload_conversation_log(log_json, filename)
