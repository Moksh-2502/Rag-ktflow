"""
Environment settings for the RAG application
This file is used to set environment variables programmatically
"""
import os

# Set Ollama-specific environment variables
os.environ["OLLAMA_GPU_LAYERS"] = "25"
os.environ["OLLAMA_KEEP_ALIVE"] = "5m"

# RAG application settings
# Using the quantized Mistral instruction model (4-bit quantization for better memory efficiency)
os.environ["OLLAMA_MODEL"] = "llama3:8b-instruct-q4_K_M"
os.environ["OLLAMA_GPU_MEMORY_LIMIT"] = "6144"  # 6GB GPU memory limit
os.environ["CHROMA_PERSIST_DIR"] = "chroma_db"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["NORMALIZE_EMBEDDINGS"] = "True"


