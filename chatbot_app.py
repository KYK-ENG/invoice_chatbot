import streamlit as st
import pandas as pd
import os

# --- 1. RAG Imports ---
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
import chromadb

# Define the directory where the processed index will be saved
PERSIST_DIR = "./invoice_index_storage" 
CSV_FILE_PATH = 'invoice_data.csv'

# Use a global placeholder for status messages that we can clear later
status_placeholder = st.empty()


# ----------------------------------------------------
# 2. Configuration and Aesthetic
# ----------------------------------------------------
st.set_page_config(
    page_title="Invoice Query POC", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

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
    width: 80%; 
    margin-left: auto;
    margin-right: auto;
    padding-top: 30px; 
}

/* Style the answer box for a cool, aesthetic presentation */
.answer-box {
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #FFFFFF; 
    border-radius: 8px;
    background-color: #111111; 
    font-size: 1.1em;
    line-height: 1.6;
    color: #FFFFFF;
}

/* Hide the fixed-bottom CSS as the search box is now at the top */
.fixed-bottom {
    display: none;
}

/* Hide the Streamlit warning related to the columns for a clean look */
div[data-testid="stColumn"] {
    overflow: visible; /* prevents cut-off */
}

/* Remove extra whitespace around the metrics sidebar */
.css-163ttbj {
    padding-top: 1rem;
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
    Sets up the RAG system, suppressing success messages for a clean UI.
    """
    # Use st.status to show progress, which auto-hides after completion
    with status_placeholder.status("Initializing RAG data services...", expanded=True) as status_msg:
        
        # 1. Define the LLM (Large Language Model) using Groq
        GROQ_API_KEY = st.secrets["groq"]["api_key"]
        llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.1) 
        Settings.llm = llm

        # 2. Setup Embedding Model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        
        # --- Index Creation/Loading Logic ---
        if not os.path.exists(PERSIST_DIR):
            status_msg.update(label="Index not found. Building data index (This is the only slow run)...", state="running", expanded=True)
            
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
            
            status_msg.update(label="Data index complete and saved! Ready for querying.", state="complete", expanded=False)
            
        else:
            status_msg.update(label="Loading data index from storage...", state="running", expanded=False)
            # Load the existing index from the saved storage directory
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            
            status_msg.update(label="Data index loaded! Ready for querying.", state="complete", expanded=False)

        query_engine = index.as_query_engine(similarity_top_k=5)
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


# --- UI Elements: Sidebar (Metrics only) ---
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


# ----------------------------------------------------
# 5. TOP SEARCH BAR AND EXECUTION
# ----------------------------------------------------

# Title is at the very top
st.title("📄 Invoice Query Tool")

# Use a container to center the search box
with st.container():
    st.markdown("<br>", unsafe_allow_html=True) 
    
    # Search form for the aesthetic look
    with st.form(key='search_form', clear_on_submit=True):
        # FIX: Changed column ratio from [10, 1] to [9, 2] to make the button wider
        col_input, col_submit = st.columns([9, 2]) 
        
        user_query = col_input.text_input(
            "What is the status of invoice INV-...",
            label_visibility="collapsed",
            key="user_query_input",
            placeholder="E.g., What is the status of INV-1003-AP? Or, How many invoices are overdue?"
        )
        
        submit_button = col_submit.form_submit_button(label='Search', type="primary")

    if submit_button and user_query:
        st.session_state["last_query"] = user_query 

        with st.spinner(f'Searching for "{user_query}"...'):
            # Execute RAG Query Engine
            try:
                response = query_engine.query(user_query)
                st.session_state["last_answer"] = str(response)
            except Exception as e:
                if "401" in str(e) or "authentication" in str(e).lower():
                     st.session_state["last_answer"] = "❌ Authentication Error: Please check your Groq API key in the Streamlit Secrets."
                else:
                    st.session_state["last_answer"] = f"❌ Error during query: {e}"
        
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
