import os
import json
import time
import requests
from typing import Dict, List, Any, Optional
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Set page configuration
st.set_page_config(
    page_title="KT-Flow RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def format_size(size_bytes):
    """Format size in bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def format_date(timestamp):
    """Format timestamp to human-readable date"""
    if not timestamp:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

def call_api(endpoint, method="GET", data=None, files=None, params=None, stream=False):
    """Make API call with error handling"""
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, stream=stream)
        elif method == "POST":
            response = requests.post(url, json=data, files=files, params=params, stream=stream)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            st.error(f"Unsupported method: {method}")
            return None
        
        # Check for errors
        response.raise_for_status()
        
        # Return response based on content type
        if stream:
            return response
        elif response.headers.get("content-type") == "application/json":
            return response.json()
        else:
            return response.text
            
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', str(e))
                st.error(f"Error details: {error_detail}")
            except:
                st.error(f"Status code: {e.response.status_code}")
        return None

# Session state initialization
if 'active_conversation' not in st.session_state:
    st.session_state.active_conversation = None
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'system_health' not in st.session_state:
    st.session_state.system_health = {}

# Load conversations
def load_conversations():
    """Load conversations from the API"""
    conversations = call_api("/conversations")
    if conversations and "conversations" in conversations:
        st.session_state.conversations = conversations["conversations"]
    else:
        st.session_state.conversations = []

# Function to display chat messages
def display_messages():
    """Display chat messages with user and assistant styling"""
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if "metadata" in msg and msg["metadata"] and "sources" in msg["metadata"]:
                    with st.expander("View Sources"):
                        for idx, source in enumerate(msg["metadata"]["sources"], 1):
                            st.markdown(f"**Source {idx}:** {source['source']}")
                            st.text(source['content'])

# Function to load conversation
def load_conversation(conversation_id):
    """Load a specific conversation and its messages"""
    conversation = call_api(f"/conversations/{conversation_id}")
    if conversation and "conversation" in conversation:
        st.session_state.active_conversation = conversation["conversation"]
        st.session_state.chat_messages = conversation["conversation"]["messages"]
        return True
    return False

# Function to create new conversation
def create_conversation(title=None):
    """Create a new conversation"""
    data = {}
    if title:
        data["title"] = title
    
    result = call_api("/conversations", method="POST", data=data)
    if result and "conversation_id" in result:
        st.session_state.active_conversation = {
            "conversation_id": result["conversation_id"],
            "title": result["title"] or "New Conversation",
            "messages": []
        }
        st.session_state.chat_messages = []
        load_conversations()
        return True
    return False

# Function to process user message
def process_user_message(message):
    """Process user message, add to UI and send to API"""
    if not st.session_state.active_conversation:
        create_conversation(f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Add message to UI
    st.session_state.chat_messages.append({"role": "user", "content": message})
    
    # Send to API
    conversation_id = st.session_state.active_conversation["conversation_id"]
    message_result = call_api(
        f"/conversations/{conversation_id}/messages", 
        method="POST", 
        data={"role": "user", "content": message}
    )
    
    # Ask question
    if st.session_state.stream_response:
        handle_streaming_response(message, conversation_id)
    else:
        handle_regular_response(message, conversation_id)

# Handle streaming response
def handle_streaming_response(question, conversation_id):
    """Handle streaming response from the API"""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        
        # Call streaming API
        params = {"include_chat_history": True}
        data = {
            "query": question,
            "k": st.session_state.result_count,
            "temperature": st.session_state.temperature,
            "include_sources": True,
            "conversation_id": conversation_id
        }
        
        with requests.post(
            f"{API_URL}/ask/stream", 
            json=data,
            params=params,
            stream=True
        ) as response:
            if response.status_code != 200:
                st.error(f"Error: {response.status_code}")
                return
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data_json = json.loads(data_str)
                            if "text" in data_json:
                                full_response += data_json["text"]
                                message_placeholder.markdown(full_response + "▌")
                            elif "sources" in data_json:
                                sources = data_json["sources"]
                        except json.JSONDecodeError:
                            continue
        
        # Update with final response
        message_placeholder.markdown(full_response)
        
        # Display sources if available
        if sources:
            with st.expander("View Sources"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"**Source {idx}:** {source['source']}")
                    st.text(source['content'])
    
    # Add assistant response to session state
    st.session_state.chat_messages.append({
        "role": "assistant", 
        "content": full_response,
        "metadata": {"sources": sources} if sources else {}
    })

# Handle regular response
def handle_regular_response(question, conversation_id):
    """Handle regular (non-streaming) response from the API"""
    # Call the ask API
    params = {"include_chat_history": True}
    data = {
        "query": question,
        "k": st.session_state.result_count,
        "temperature": st.session_state.temperature,
        "include_sources": True,
        "conversation_id": conversation_id
    }
    
    result = call_api("/ask", method="POST", data=data, params=params)
    
    if result and "answer" in result:
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(result["answer"])
            if "sources" in result and result["sources"]:
                with st.expander("View Sources"):
                    for idx, source in enumerate(result["sources"], 1):
                        st.markdown(f"**Source {idx}:** {source['source']}")
                        st.text(source['content'])
        
        # Add to session state
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "metadata": {"sources": result["sources"]} if "sources" in result else {}
        })

# UI Layout
def render_sidebar():
    """Render sidebar with controls and info"""
    with st.sidebar:
        st.title("🧠 KT-Flow RAG")
        
        # System Health
        st.header("System Status")
        if st.button("Check Health"):
            health = call_api("/health")
            stats = call_api("/stats")
            if health and stats:
                st.session_state.system_health = {**health, **stats}
        
        if st.session_state.system_health:
            st.metric("Document Count", st.session_state.system_health.get("document_count", 0))
            st.metric("Chunk Count", st.session_state.system_health.get("chunk_count", 0))
            st.text(f"Status: {st.session_state.system_health.get('status', 'Unknown')}")
            st.progress(1.0 if st.session_state.system_health.get("status") == "healthy" else 0.5)
        
        # Settings
        st.header("Settings")
        st.session_state.result_count = st.slider("Result Count", 1, 10, 3)
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        st.session_state.stream_response = st.checkbox("Stream Response", True)
        
        # Conversations List
        st.header("Conversations")
        if st.button("New Conversation"):
            create_conversation()
        
        if st.button("Refresh Conversations"):
            load_conversations()
        
        # Display conversations
        for conv in st.session_state.conversations:
            conv_title = conv.get("title", "Untitled") 
            if len(conv_title) > 20:
                conv_title = f"{conv_title[:17]}..."
            
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                if st.button(conv_title, key=f"conv_{conv['conversation_id']}"):
                    load_conversation(conv["conversation_id"])
            with col2:
                if st.button("🗑️", key=f"del_{conv['conversation_id']}", help="Delete conversation"):
                    call_api(f"/conversations/{conv['conversation_id']}", method="DELETE")
                    if st.session_state.active_conversation and st.session_state.active_conversation["conversation_id"] == conv["conversation_id"]:
                        st.session_state.active_conversation = None
                        st.session_state.chat_messages = []
                    load_conversations()
                    st.experimental_rerun()
        
        # Document Upload Section
        st.header("Upload Documents")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "json", "csv", "pdf"])
            document_type = st.selectbox("Document Type", ["text", "markdown", "json", "csv", "pdf"])
            source_name = st.text_input("Source Name (optional)")
            submit_button = st.form_submit_button("Upload")
            
            if submit_button and uploaded_file is not None:
                file_size = uploaded_file.size
                if file_size > MAX_UPLOAD_SIZE:
                    st.error(f"File too large: {format_size(file_size)}. Maximum size: {format_size(MAX_UPLOAD_SIZE)}")
                else:
                    # Prepare the file for upload
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")
                    }
                    
                    # Prepare form data
                    form_data = {"document_type": document_type}
                    if source_name:
                        form_data["source"] = source_name
                    
                    # Call the API
                    with st.spinner('Uploading and processing document...'):
                        response = requests.post(f"{API_URL}/ingest/file", files=files, data=form_data)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Document uploaded! {result.get('chunks', 0)} chunks created.")
                            # Update health stats
                            health = call_api("/health")
                            stats = call_api("/stats")
                            if health and stats:
                                st.session_state.system_health = {**health, **stats}
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")

def render_main_area():
    """Render the main chat area"""
    if not st.session_state.active_conversation:
        st.header("Welcome to KT-Flow RAG System")
        st.write("Select a conversation from the sidebar or create a new one to start.")
        
        # Display some stats if available
        if st.session_state.system_health:
            st.subheader("System Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Document Count", st.session_state.system_health.get("document_count", 0))
            with col2:
                st.metric("Chunk Count", st.session_state.system_health.get("chunk_count", 0))
            with col3:
                st.metric("Status", st.session_state.system_health.get("status", "Unknown"))
            
            # If documents exist, allow creating conversation directly
            if st.session_state.system_health.get("document_count", 0) > 0:
                if st.button("Start New Conversation", key="welcome_new_conv"):
                    create_conversation()
                    st.experimental_rerun()
        
        # Help information
        with st.expander("How to use this app"):
            st.markdown("""
            1. **Upload Documents** - Use the sidebar to upload documents for knowledge base creation
            2. **Create Conversations** - Start a new conversation to interact with the knowledge base
            3. **Ask Questions** - Enter questions and the system will retrieve relevant information
            4. **View Sources** - See which sources were used to answer your questions
            """)
        
        # Show sample questions if documents exist
        if st.session_state.system_health.get("document_count", 0) > 0:
            with st.expander("Sample Questions to Ask"):
                st.markdown("- What are the main topics covered in the documents?")
                st.markdown("- Can you summarize the key points from the latest meeting notes?")
                st.markdown("- What are the next steps mentioned in the documents?")
                st.markdown("- Who are the key stakeholders mentioned?")
    else:
        # Show conversation header
        st.header(st.session_state.active_conversation.get("title", "Conversation"))
        
        # Display chat messages
        display_messages()
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")
        if user_input:
            process_user_message(user_input)

# Main execution
def main():
    try:
        # Check API connection
        health = call_api("/health")
        if health:
            st.session_state.system_health = health
            # Load conversations on startup
            if not st.session_state.conversations:
                load_conversations()
        
        # Render UI
        render_sidebar()
        render_main_area()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Make sure the API server is running and accessible.")

if __name__ == "__main__":
    main()
