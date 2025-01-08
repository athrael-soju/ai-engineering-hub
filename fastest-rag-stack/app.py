import os
import gc
import uuid
import base64
import tempfile

import docx2txt
import streamlit as st
from dotenv import load_dotenv

# For reading PDF via llama_index's SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader

# Your existing code in rag_code.py
from rag_code import EmbedData, QdrantVDB_QB, Retriever, RAG

############################
# SESSION & ENV SETUP
############################
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

load_dotenv()

collection_name = "chat_with_docs"
batch_size = 32

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

############################
# READING FILE HELPERS
############################
def read_pdf_file(file):
    """
    Reads PDF data from an uploaded file-like object using 
    LlamaIndex's SimpleDirectoryReader. We must temporarily 
    save the file to disk so the reader can process it.
    """
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp.flush()
        tmp_name = tmp.name

    try:
        loader = SimpleDirectoryReader(
            input_files=[tmp_name], 
            required_exts=[".pdf"],
            recursive=False
        )
        docs = loader.load_data()
        for d in docs:
            text += d.text + "\n"
    except Exception as e:
        st.write(f"Error reading PDF: {e}")
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_name)
        except:
            pass
    return text

def read_docx_file(file):
    """
    Reads DOCX using docx2txt. 
    Again, we must temporarily save the bytes to disk.
    """
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.read())
        tmp.flush()
        tmp_name = tmp.name

    try:
        text = docx2txt.process(tmp_name)
    except Exception as e:
        st.write(f"Error reading DOCX: {e}")
    finally:
        try:
            os.remove(tmp_name)
        except:
            pass
    return text

def read_txt_file(file):
    """
    Read raw text from a .txt file. 
    """
    try:
        # file_uploader returns a BytesIO, so decode
        text = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.write(f"Error reading TXT: {e}")
        text = ""
    return text

def display_text_preview(text, max_chars=300):
    """
    Show a short snippet of text to the user
    """
    st.markdown(f"```\n{text[:max_chars]}\n... (truncated) ...\n```")

############################
# APP UI
############################

st.title("Folder Simulation with Multi-File Upload (One-by-one Processing)")
st.sidebar.button("Clear ↺", on_click=reset_chat)

# Step 1: Let user pick multiple files from a folder
uploaded_files = st.file_uploader(
    "Select multiple PDF/DOCX/TXT files (Ctrl+Click or Shift+Click to select many)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"You've selected {len(uploaded_files)} file(s). We'll index them one by one.")

    # Create a Qdrant collection
    qdrant_vdb = QdrantVDB_QB(
        collection_name=collection_name,
        batch_size=batch_size,
        vector_dim=1024  # Adjust if your embed model uses dimension 1024
    )
    qdrant_vdb.define_client()
    qdrant_vdb.create_collection()

    # Create embedding object
    embedder = EmbedData(embed_model_name="BAAI/bge-large-en-v1.5", batch_size=batch_size)

    # Process each file in a loop
    for i, file in enumerate(uploaded_files, start=1):
        st.write(f"**[{i}/{len(uploaded_files)}]** Processing file: `{file.name}`")
        text_content = ""
        # Read file content based on extension
        if file.name.lower().endswith(".pdf"):
            text_content = read_pdf_file(file)
        elif file.name.lower().endswith(".docx"):
            text_content = read_docx_file(file)
        elif file.name.lower().endswith(".txt"):
            text_content = read_txt_file(file)

        # If there's no text, skip
        if not text_content.strip():
            st.write("No text extracted. Skipping...")
            continue

        st.write("Here's a snippet of extracted text:")
        display_text_preview(text_content)

        # Embed the text (just one chunk - the entire text)
        st.write("Generating embeddings...")
        embeddings = embedder.generate_embedding([text_content])

        # Ingest into Qdrant
        st.write("Ingesting into Qdrant collection...")
        qdrant_vdb.ingest_file_data([text_content], embeddings)

        st.write(f"Done with: {file.name}")
        st.write("---")

    # Once all are ingested, build the RAG pipeline
    retriever = Retriever(vector_db=qdrant_vdb, embeddata=embedder)
    query_engine = RAG(
        retriever=retriever,
        llm_name="Meta-Llama-3.3-70B-Instruct"
    )

    # Cache the engine
    st.session_state.file_cache["combined_engine"] = query_engine
    st.success("All selected documents indexed successfully!")

# Chat section
st.write("## Chat Interface")

if "messages" not in st.session_state:
    reset_chat()

# Display chat messages so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user query
prompt = st.chat_input("Ask a question about your docs...")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve query engine
    query_engine = st.session_state.file_cache.get("combined_engine", None)

    if not query_engine:
        with st.chat_message("assistant"):
            st.markdown("No indexed documents found. Please upload files first.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            streaming_response = query_engine.query(prompt)
            for chunk in streaming_response:
                try:
                    new_text = chunk.raw["choices"][0]["delta"]["content"]
                    full_response += new_text
                    message_placeholder.markdown(full_response + "▌")
                except:
                    pass

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
