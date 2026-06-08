import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
import numpy as np
import faiss
import os

# Load API key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -------------------------
# Chunking Function
# -------------------------
def chunk_text(text,
               chunk_size=500,
               overlap=50):

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
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Document Q&A Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("PDF Loaded")

    # Create Chunks
    chunks = chunk_text(text)

    st.write(
        f"Total Chunks: {len(chunks)}"
    )

    # Generate Embeddings
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

    # Create FAISS Index
    st.write("Embeddings Shape:")
    st.write(embeddings.shape)

    st.write("Total Embeddings:")
    st.write(len(embeddings))
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    st.success(
        "FAISS Index Created"
    )

    # User Question

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        # Create question embedding
        query_embedding = (
            client.models.embed_content(
                model="gemini-embedding-001",
                contents=question
            )
        )
        
        st.write("Chunks:", len(chunks))
        st.write("Embeddings List:", len(embeddings))
        query_vector = np.array(
            [
                query_embedding
                .embeddings[0]
                .values
            ],
            dtype=np.float32
        )

    # Search FAISS
        D, I = index.search(
        query_vector,
        k=3
    )
        context = ""
        for idx in I[0]:
            context += chunks[idx]
            context += "\n\n"
# Create prompt
        prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

If the answer is not found in the context,
say:
'I could not find this information in the document.'

Context:
{context}

Question:
{question}
"""

    # Send prompt to Gemini
        response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Show answer
        st.subheader("Answer")

        st.write(response.text)
        st.session_state.messages.append(
    {
        "question": question,
        "answer": response.text
    }
)

    # Show source chunks
        with st.expander("View Sources"):
            for idx in I[0]:
                st.write(f"Chunk {idx}")
                st.write(chunks[idx])
                st.markdown("---")
        st.subheader("Chat History")

for chat in reversed(
    st.session_state.messages
):

    st.markdown(
        f"**Question:** {chat['question']}"
    )

    st.markdown(
        f"**Answer:** {chat['answer']}"
    )

    st.markdown("---")