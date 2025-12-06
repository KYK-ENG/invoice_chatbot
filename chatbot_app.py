import streamlit as st
import pandas as pd
import os

# --- 1. RAG Imports ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# NEW IMPORTS FOR GROQ
from llama_index.llms.groq import Groq # <--- Groq is now correctly imported
import chromadb


# ----------------------------------------------------
# 2. Configuration and Aesthetic
# ----------------------------------------------------
st.set_page_config(
    page_title="Invoice Chatbot POC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the aesthetic look (Keeping the minimal black/white)
st.markdown("""
<style>
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
/* Ensure the input box at the bottom is visible */
.stTextInput > div > div > input {
    color: #FFFFFF; 
    background-color: #111111;
    border: 1px solid #444444;
}
/* Fix the chat input to the bottom of the screen */
.fixed-bottom {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px;
    background-color: #000000; 
    border-top: 1px solid #111111; 
    z-index: 1000;
}
/* Adjust main content padding to prevent overlap with fixed footer */
.main > div {
    padding-bottom: 80px; 
}
/* Chat bubble styling for minimal look */
.user-bubble {
    background-color: #111111;
    color: #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
}
.assistant-bubble {
    background-color: #333333;
    color: #FFFFFF;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. Data Loading and RAG Setup
# ----------------------------------------------------

# --- Load Invoice Data (Modified to handle CSV directly) ---
@st.cache_data
def load_data(file_path):
    """Loads the invoice data from the CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Convert date columns for proper display/processing
        date_cols = ['Issue_Date', 'Due_Date', 'Approval_Date', 'Payment_Date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

CSV_FILE_PATH = 'invoice_data.csv'
df_invoices = load_data(CSV_FILE_PATH)


# Define the directory where the processed index will be saved
PERSIST_DIR = "./invoice_index_storage" 

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
        model="llama-3.1-8b-instant", # The current stable Groq model fix
        api_key=GROQ_API_KEY,
        temperature=0.1 
    )
    Settings.llm = llm # Set the global LLM

    # 2. Setup Embedding Model (Local, open-source model)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model # Set the global embedding model

    # --- Index Creation/Loading Logic (Fixes the 3-5 minute delay) ---
    if not os.path.exists(PERSIST_DIR):
        st.warning("Index not found. Building index (This is the slow part, please wait 3-5 minutes)...")
        
        # 3. Convert DataFrame rows into LlamaIndex Documents
        documents = []
        for index, row in df.iterrows():
            # Create a text string representing the whole invoice record
            invoice_text = f"Invoice Record ID: {row['Invoice_ID']}, Company: {row['Company_Name']}, Vendor: {row['Vendor_Name']}, Type: {row['Type']}, Amount: {row['Total_Amount']} {row['Currency']}, Status: {row['Status']}, Due Date: {row['Due_Date']}, Approval Required By: {row['Approval_Required_By']}, Description: {row['Invoice_Description']}"
            documents.append(Document(text=invoice_text, doc_id=row['Invoice_ID']))

        # Initialize ChromaDB Client and Vector Store (required for first build)
        db = chromadb.Client() 
        chroma_collection = db.get_or_create_collection("invoice_poc_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create Storage Context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the Index
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            embed_model=embed_model
        )
        
        # Save the index to disk for faster loading next time
        index.storage_context.persist(persist_dir=PERSIST_DIR) 
        
        st.success("Index built and saved! Future loads will be fast.")
        
    else:
        # Load the existing index from the saved storage directory
        st.success("Index found! Loading from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # Use a simpler, non-streaming query engine 
    query_engine = index.as_query_engine(similarity_top_k=5)
    
    st.success("RAG Initialization complete and ready!")
    return query_engine


# ----------------------------------------------------
# 4. Streamlit UI and RAG Execution
# ----------------------------------------------------

st.title("📄 Invoice Chatbot POC")
st.markdown("### AP/AR Query Tool")

# --- Initialize RAG and Handle Data Error ---
if df_invoices.empty:
    st.error("Data not loaded. Please ensure 'invoice_data.csv' is correctly placed and named.", icon="🚨")
    st.stop() # Stop if data is missing

# Load the RAG Query Engine
query_engine = setup_rag(df_invoices)


# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "assistant", "content": "Hello! I have loaded 20 invoices. How can I help you check a status, amount, or approval?"}]


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
    
    st.markdown("---")
    st.markdown("### 📋 Raw Data Reference")
    # Fix the warning and place the DataFrame in the sidebar
    st.dataframe(df_invoices, width='stretch')


# --- Display Chat History (Main Screen) ---
st.markdown("### 💬 Conversation History")

# Display every message in the history using the custom bubble styling
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        

# ----------------------------------------------------
# 5. FIXED CHAT INPUT (The Google/Gemini Search Bar Style)
# ----------------------------------------------------

# Use a container to hold the chat input box fixed to the bottom
input_container = st.container()

# Apply the custom CSS class to fix the container to the bottom
input_container.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
with input_container:
    with st.form(key='chat_form', clear_on_submit=True):
        col_input, col_submit = st.columns([10, 1])
        
        user_query = col_input.text_input(
            "Ask a question about an invoice...",
            label_visibility="collapsed",
            key="user_query_input"
        )
        
        submit_button = col_submit.form_submit_button(label='Send', type="primary")

    if submit_button and user_query:
        # 1. Add user query to history immediately
        st.session_state['chat_history'].append({"role": "user", "content": user_query})
        
        with st.spinner('Thinking...'):
            # 2. Execute RAG Query Engine
            try:
                # The query engine searches the vector store and generates the response
                response = query_engine.query(user_query)
                ai_response = str(response)
            except Exception as e:
                ai_response = f"Sorry, an error occurred during the query: {e}"

        # 3. Add AI response to history
        st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
        
        # 4. Rerun app to update the history display
        st.rerun()

input_container.markdown('</div>', unsafe_allow_html=True) # Close the fixed-bottom div