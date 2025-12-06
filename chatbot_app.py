import streamlit as st
import pandas as pd
import os

# --- 1. RAG Imports ---
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
import chromadb

# ----------------------------------------------------
# 2. Configuration and Aesthetic
# ----------------------------------------------------
st.set_page_config(
    page_title="Invoice Query POC", # Changed title for simplicity
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Define the directory where the processed index will be saved
PERSIST_DIR = "./invoice_index_storage" 
CSV_FILE_PATH = 'invoice_data.csv'

# Custom CSS for the aesthetic look (Minimal Black/White Theme)
st.markdown("""
<style>
/* Base Dark Theme Configuration */
:root {
    --primary-color: #FFFFFF;
    --background-color: #000000;
    --secondary-background-color: #111111;
    --text-color: #FFFFFF;
    --font: sans-serif;
}
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
}

/* Style the Search Input Box for a large, central look */
.stForm {
    width: 80%; /* Center the form in a wide layout */
    margin-left: auto;
    margin-right: auto;
    padding-top: 30px; /* Space from the top */
}

/* Style the answer box for a cool, aesthetic presentation */
.answer-box {
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #FFFFFF; /* White border for contrast */
    border-radius: 8px;
    background-color: #111111; /* Dark background */
    font-size: 1.1em;
    line-height: 1.6;
    color: #FFFFFF;
}

/* Hide the fixed-bottom CSS as the search box is now at the top */
.fixed-bottom {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Data Loading and RAG Setup
# ----------------------------------------------------

# --- Load Invoice Data ---
@st.cache_data
def load_data(file_path):
    """Loads the invoice data from the CSV file."""
    try:
        df = pd.read_csv(file_path)
        date_cols = ['Issue_Date', 'Due_Date', 'Approval_Date', 'Payment_Date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_invoices = load_data(CSV_FILE_PATH)


# --- RAG Initialization (The Brain) ---
@st.cache_resource
def setup_rag(df):
    """
    Sets up the RAG system: Loads from storage if available, otherwise builds and saves.
    """
    st.info("Initializing RAG pipeline (checking storage first)...")
    
    # 1. Define the LLM (Large Language Model) using Groq (Zero-Cost, Fast)
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
    llm = Groq(
        model="llama-3.1-8b-instant", 
        api_key=GROQ_API_KEY,
        temperature=0.1 
    )
    Settings.llm = llm

    # 2. Setup Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    # --- Index Creation/Loading Logic ---
    if not os.path.exists(PERSIST_DIR):
        st.warning("Index not found. Building index (This is the slow part, please wait 3-5 minutes)...")
        
        # Convert DataFrame rows into Documents
        documents = []
        for index, row in df.iterrows():
            invoice_text = f"Invoice Record ID: {row['Invoice_ID']}, Company: {row['Company_Name']}, Vendor: {row['Vendor_Name']}, Type: {row['Type']}, Amount: {row['Total_Amount']} {row['Currency']}, Status: {row['Status']}, Due Date: {row['Due_Date']}, Approval Required By: {row['Approval_Required_By']}, Description: {row['Invoice_Description']}"
            documents.append(Document(text=invoice_text, doc_id=row['Invoice_ID']))

        # Initialize ChromaDB Client and Vector Store (required for first build)
        db = chromadb.Client() 
        chroma_collection = db.get_or_create_collection("invoice_poc_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
        index.storage_context.persist(persist_dir=PERSIST_DIR) 
        
        st.success("Index built and saved! Future loads will be fast.")
        
    else:
        # Load the existing index from the saved storage directory
        st.success("Index found! Loading from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=5)
    
    st.success("RAG Initialization complete and ready!")
    return query_engine

# ----------------------------------------------------
# 4. Streamlit UI and RAG Execution
# ----------------------------------------------------

# --- Initialize RAG and Handle Data Error ---
if df_invoices.empty:
    st.error("Data not loaded. Please ensure 'invoice_data.csv' is correctly placed and named.", icon="🚨")
    st.stop() 

# Load the RAG Query Engine
query_engine = setup_rag(df_invoices)


# --- Initialize Session State for current answer/query ---
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = "Ask a query above to see the answer here."


# --- UI Elements: Sidebar (Metrics and Raw Data) ---
with st.sidebar:
    st.markdown("## 📊 Invoice Metrics")
    total_invoices = len(df_invoices)
    open_invoices = df_invoices[df_invoices['Status'].isin(['Pending Approval', 'In Progress', 'Overdue'])].shape[0]
    total_value_usd = df_invoices[df_invoices['Currency'] == 'USD']['Total_Amount'].sum()

    st.metric("Total Invoices Loaded", total_invoices)
    st.metric("Invoices Needing Action", open_invoices)
    st.metric("Total Value (USD)", f"${total_value_usd:,.2f}")
    
    st.markdown("---")
    st.info("The chatbot queries this data directly.")
    
    # We remove the raw data table from the sidebar to focus on the search results
    # If you want to put the raw data back, add it here.


# ----------------------------------------------------
# 5. TOP SEARCH BAR AND EXECUTION
# ----------------------------------------------------

# Title is at the very top
st.title("📄 Invoice Query Tool")

# Use a container to center the search box
with st.container():
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    # Search form for the aesthetic look
    with st.form(key='search_form', clear_on_submit=True):
        col_input, col_submit = st.columns([10, 1])
        
        user_query = col_input.text_input(
            "What is the status of invoice INV-...",
            label_visibility="collapsed",
            key="user_query_input",
            placeholder="E.g., What is the status of INV-1003-AP? Or, How many invoices are overdue?"
        )
        
        submit_button = col_submit.form_submit_button(label='Search', type="primary")

    if submit_button and user_query:
        st.session_state["last_query"] = user_query # Save query

        with st.spinner(f'Searching for "{user_query}"...'):
            # Execute RAG Query Engine
            try:
                response = query_engine.query(user_query)
                st.session_state["last_answer"] = str(response)
            except Exception as e:
                st.session_state["last_answer"] = f"❌ Error during query: {e}"
        
        # Force rerun to display the answer immediately
        st.rerun()

# ----------------------------------------------------
# 6. DISPLAY RESULTS (Cool Way)
# ----------------------------------------------------

# Only display the answer if a query has been made
if st.session_state["last_query"]:
    st.markdown("---")
    st.subheader(f"🔍 Result for: **{st.session_state['last_query']}**")
    
    # The cool, aesthetic result box
    st.markdown(f'<div class="answer-box">🤖 {st.session_state["last_answer"]}</div>', unsafe_allow_html=True)

# Footer/Contextual information
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("Powered by RAG, Streamlit, and the zero-cost Groq API.")
