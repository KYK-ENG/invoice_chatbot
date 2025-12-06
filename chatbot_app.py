import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- NEW UI LIBRARY ---
import streamlit_shadcn_ui as ui 

# --- 1. RAG Imports ---
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
import chromadb

# Define the directory where the processed index will be saved
PERSIST_DIR = "./invoice_index_storage" 
CSV_FILE_PATH = 'invoice_data.csv'


# ----------------------------------------------------
# 2. Configuration and Aesthetic (Updated Theme)
# ----------------------------------------------------
st.set_page_config(
    page_title="Invoice Query Tool", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Custom CSS for the aesthetic look (Softer Dark Theme)
st.markdown("""
<style>
/* Base Dark Theme Configuration (Softened #000000 to #0F0F0F) */
:root {
    --background-color: #0F0F0F; 
    --secondary-background-color: #1A1A1A;
    --text-color: #E0E0E0;
}
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}
h1, h2, h3, h4 {
    color: #FFFFFF;
}

/* Style the Search Input Box for a large, central look */
.stForm {
    width: 90%; 
    margin-left: auto;
    margin-right: auto;
    padding-top: 20px; 
}

/* Style the answer box for a cool, aesthetic presentation */
.answer-box {
    padding: 30px;
    margin-top: 20px;
    border-radius: 12px;
    background-color: var(--secondary-background-color); 
    border-left: 5px solid #00BFFF; /* A modern blue accent */
    font-size: 1.2em;
    line-height: 1.6;
    color: #FFFFFF;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* Hide the fixed-bottom CSS as the search box is now at the top */
.fixed-bottom {
    display: none;
}

/* Hide the technical status output from the final UI */
.stStatus {
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
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True) 
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_invoices = load_data(CSV_FILE_PATH)


# --- FIX: Function to cache the Chroma client and avoid persistence errors ---
@st.cache_resource
def get_chroma_client():
    """Initializes and returns the ChromaDB client."""
    return chromadb.Client()


# --- RAG Initialization (The Brain) ---
@st.cache_resource
def setup_rag(df):
    """
    Sets up the RAG system. The print statements help track progress during the
    initial slow build, but are not rendered to the UI, avoiding the cache error.
    """
    chroma_client = get_chroma_client() 

    # 1. Define the LLM (Groq for Zero-Cost, Speed)
    GROQ_API_KEY = st.secrets["groq"]["api_key"]
    llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.1) 
    Settings.llm = llm

    # 2. Setup Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    
    # --- Index Creation/Loading Logic (The Persistence Fix) ---
    if not os.path.exists(PERSIST_DIR):
        print("--- Index not found. Building data index (This is the slow run)...") # Use print instead of st.warning
        
        # Convert DataFrame rows into Documents
        documents = []
        for index, row in df.iterrows():
            invoice_text = f"Invoice Record ID: {row['Invoice_ID']}, Company: {row['Company_Name']}, Vendor: {row['Vendor_Name']}, Type: {row['Type']}, Amount: {row['Total_Amount']} {row['Currency']}, Status: {row['Status']}, Due Date: {row['Due_Date']}, Approval Required By: {row['Approval_Required_By']}, Description: {row['Invoice_Description']}"
            documents.append(Document(text=invoice_text, doc_id=row['Invoice_ID']))

        # Initialize ChromaDB Vector Store
        chroma_collection = chroma_client.get_or_create_collection("invoice_poc_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
        index.storage_context.persist(persist_dir=PERSIST_DIR) 
        
        print("--- Data index complete and saved! Future loads will be fast. ---")
        
    else:
        print("--- Loading data index from storage... ---")

        # --- CRITICAL FIX FOR KEYERROR: 'default' ---
        chroma_collection = chroma_client.get_or_create_collection("invoice_poc_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR,
            vector_store=vector_store,
        )
        
        index = load_index_from_storage(storage_context)
        
        print("--- Data index loaded! Ready for querying. ---")

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


# ----------------------------------------------------
# 5. METRICS (Modern Cards)
# ----------------------------------------------------

# Calculate metrics
total_invoices = len(df_invoices)
open_invoices = df_invoices[df_invoices['Status'].isin(['Pending Approval', 'In Progress', 'Overdue'])].shape[0]
total_value_usd = df_invoices[df_invoices['Currency'] == 'USD']['Total_Amount'].sum()
overdue_invoices = df_invoices[df_invoices['Status'] == 'Overdue'].shape[0]

st.markdown("## 📊 Invoice Insights")
cols = st.columns(4)

with cols[0]:
    ui.card(title="Total Invoices", content=f"{total_invoices:,}", description="Total records in RAG index", key="card1")
with cols[1]:
    # Highlight actionable items (In Progress, Pending, Overdue)
    ui.card(title="Invoices Needing Action", content=f"{open_invoices:,}", description="Pending/In Progress/Overdue", key="card2")
with cols[2]:
    # Highlight Overdue items
    ui.card(title="Overdue Invoices", content=f"{overdue_invoices:,}", description="Immediate attention required", key="card3")
with cols[3]:
    # Total value of USD invoices
    ui.card(title="Total Value (USD)", content=f"${total_value_usd:,.2f}", description="Total value in USD", key="card4")

st.markdown("---")


# ----------------------------------------------------
# 6. TOP SEARCH BAR AND EXECUTION
# ----------------------------------------------------

st.title("💡 AI Query Search")

# Use a container to center the search box
with st.container():
    st.markdown("<br>", unsafe_allow_html=True) 
    
    # Search form for the aesthetic look
    with st.form(key='search_form', clear_on_submit=True):
        # FIX: Corrected button column width
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
# 7. DISPLAY RESULTS (Cool Way)
# ----------------------------------------------------

# Only display the answer if a query has been made
if st.session_state["last_query"]:
    st.markdown("---")
    st.subheader(f"✅ AI Result for: **{st.session_state['last_query']}**")
    
    # The cool, aesthetic result box
    st.markdown(f'<div class="answer-box">{st.session_state["last_answer"]}</div>', unsafe_allow_html=True)

# Footer/Contextual information
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption(f"App deployed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by RAG and Groq.")
