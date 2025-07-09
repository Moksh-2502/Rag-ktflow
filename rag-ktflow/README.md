# KTFlow RAG System

A free, open-source local Retrieval-Augmented Generation (RAG) system for developer knowledge transfer.

## Features

- **100% Local**: All components run locally without external API dependencies
- **FastAPI Backend**: Modern, high-performance API with async support
- **ChromaDB Vector Store**: Local vector database for document storage
- **SentenceTransformers**: Efficient document embedding using all-MiniLM-L6-v2
- **Ollama Integration**: Connect to local LLMs like Mistral and LLaMA 3
- **Streaming Support**: Get streaming responses for a better user experience
- **File Upload**: Support for text, markdown, JSON, CSV, and PDF files
- **Token-Aware Chunking**: Split documents according to token boundaries for accurate context retrieval
- **Improved Error Handling**: Enhanced validation, retry logic, and fallback mechanisms for production reliability

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally with at least one model (e.g., mistral)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-ktflow.git
cd rag-ktflow
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Ollama (if not already done):
```bash
# Install Ollama from https://ollama.ai/
# Then pull a model
ollama pull mistral
```

### Running the Application

Start the FastAPI server:
```bash
python app.py
```

The API will be available at http://localhost:8000. You can also access the Swagger documentation at http://localhost:8000/docs.

## API Endpoints

### Health Check
```
GET /health
```
Returns the status of the service.

### Ingest Document
```
POST /ingest
```
Ingest a document into the vector store. The request body should include:
- `content`: Document content (text)
- `source`: Source identifier
- `document_type`: Type of document (text, markdown, json, csv, pdf)
- `metadata`: Optional metadata

### Ingest File
```
POST /ingest/file
```
Upload a file to be ingested. Use form data with:
- `file`: File to upload
- `source`: Source identifier
- `document_type`: (Optional) Type of document
- `metadata`: (Optional) JSON string with metadata

### Ask Question
```
POST /ask
```
Answer a question using RAG.  JSON body fields:
* `query` (str, required) – your question
* `k` (int, optional, default 3) – number of chunks to retrieve
* `temperature` (float, optional, default 0.7)
* `max_tokens` (int, optional, default 500)
* `include_sources` (bool, optional) – attach source chunks in the answer

Optional **query-string parameters**
* `force_rag=true` – bypass chat and always run RAG
* `include_chat_history=false` – do not prepend previous messages (recommended for fact look-ups)

### Ask Question (Streaming)
```
POST /ask/stream  (Server-Sent Events)
```
Same payload/params as `/ask`, but the response is streamed as SSE events with a `text` field per token and a final `sources` event.

## Example Usage

### Ingest a Document

```bash
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "FastAPI is a modern web framework for building APIs with Python. It is fast, easy to use, and comes with great documentation.",
  "source": "fastapi-docs",
  "document_type": "text",
  "metadata": {
    "author": "Sebastián Ramírez",
    "topic": "web frameworks"
  }
}'
```

### Example (non-streaming)

```bash
# Unix/macOS single-line
curl -X POST "http://localhost:8000/ask?force_rag=true&include_chat_history=false" \
     -H "Content-Type: application/json" \
     -d '{"query":"What is FastAPI?","k":3,"include_sources":true}'

# Windows CMD multi-line
curl -X POST "http://localhost:8000/ask?force_rag=true&include_chat_history=false" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\":\"What is FastAPI?\",\"k\":3,\"include_sources\":true}"
```

### Example (streaming)

```bash
# Requires -N to disable curl buffering
curl -N -X POST "http://localhost:8000/ask/stream?force_rag=true&include_chat_history=false" \
     -H "Content-Type: application/json" \
     -d '{"query":"What is the user growth percentage?","k":6}'
```

## Environment Variables

You can customize the application by setting these environment variables:

- `OLLAMA_MODEL`: Name of the Ollama model to use (default: mistral)
- `CHROMA_PERSIST_DIR`: Directory to store ChromaDB data (default: chroma_db)
- `USE_TOKEN_CHUNKING`: Whether to use token-based chunking (default: False)
- `TOKENIZER_NAME`: Name of tokenizer to use with token chunking (default: None)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.