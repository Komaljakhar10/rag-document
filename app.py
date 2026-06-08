import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
import numpy as np
import faiss
import os

# -------------------------
# Load API Key
# -------------------------
load_dotenv()

api_key = (
    st.secrets["GEMINI_API_KEY"]
    if "GEMINI_API_KEY" in st.secrets
    else os.getenv("GEMINI_API_KEY")
)

client = genai.Client(
    api_key=api_key
)

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# -------------------------
# Chunk Function
# -------------------------

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks

# -------------------------
# UI
# -------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)
st.title("🤖 AI Document Assistant")
st.caption(
    "Upload a PDF and ask questions using RAG + Gemini"
)
with st.sidebar:
    st.header("Controls")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("Upload New PDF"):
        ...

# -------------------------
# Reset Button
# -------------------------

if st.button("🔄 Upload New PDF"):

    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.pdf_processed = False
    st.session_state.messages = []

    st.rerun()

# -------------------------
# PDF Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# -------------------------
# Process PDF Only Once
# -------------------------

if uploaded_file and not st.session_state.pdf_processed:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("PDF Loaded")

    chunks = chunk_text(text)

    st.write(
        f"Total Chunks: {len(chunks)}"
    )

    embeddings = []

    with st.spinner(
        "Creating embeddings..."
    ):

        for chunk in chunks:

            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )

            embeddings.append(
                response.embeddings[0].values
            )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    st.success(
        "Embeddings Created"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.pdf_processed = True

    st.success(
        "FAISS Index Created"
    )
    st.subheader("📊 PDF Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
        "Pages",
        len(reader.pages)
    )
    with col2:
        st.metric(
        "Chunks",
        len(chunks)
    )
    if st.session_state.pdf_processed:
        st.success(
        "✅ PDF Processed Successfully"
    )

# -------------------------
# Question Answering
# -------------------------

if st.session_state.pdf_processed:

    st.success(
        "PDF Ready For Questions"
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        # Create query embedding
        query_embedding = (
            client.models.embed_content(
                model="gemini-embedding-001",
                contents=question
            )
        )

        query_vector = np.array(
            [
                query_embedding
                .embeddings[0]
                .values
            ],
            dtype=np.float32
        )

        # Search FAISS
        D, I = st.session_state.index.search(
            query_vector,
            k=3
        )

        # Build context
        context = ""

        for idx in I[0]:

            context += (
                st.session_state.chunks[idx]
            )

            context += "\n\n"

        # Prompt
        prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

If the answer is not found in the context, say:

'I could not find this information in the document.'

Context:
{context}

Question:
{question}
"""

        # Generate Answer
        with st.spinner(
            "Generating answer..."
        ):

            try:

                response = (
                    client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                )

                st.markdown("### 🤖 Answer")
                st.info(response.text)

                # Save chat history
                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": response.text
                    }
                )

                # Sources
                with st.expander(
                    "View Sources"
                ):

                    for idx in I[0]:

                        st.write(
                            st.session_state.chunks[idx]
                        )

                        st.markdown("---")

            except Exception as e:

                st.error(
                    f"Gemini API Error: {e}"
                )

    # Clear Chat Button
    if st.button(
        "Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()

# -------------------------
# Chat History
# -------------------------

if st.session_state.messages:

    st.subheader(
        "Chat History"
    )

    for chat in reversed(
        st.session_state.messages
    ):

        st.chat_message("user").write(
    chat["question"]
) 
        st.chat_message("assistant").write(
    chat["answer"]
)

        st.markdown("---")