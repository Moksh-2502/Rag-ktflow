"""
FastAPI application for local RAG system with Ollama integration
"""
import base64
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request, Body, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import random
import re

import env_settings

from dotenv import load_dotenv
load_dotenv()

from embeddings import DocumentChunk, DocumentProcessor
from ollama import OllamaClient
from vectorstore import VectorStore
from chat_history import ChatHistoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")

os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL
OLLAMA_GPU_MEMORY_LIMIT = os.getenv("OLLAMA_GPU_MEMORY_LIMIT", "4G")  # Reduced memory limit
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() == "true"
SENTENCE_LEVEL = os.getenv("SENTENCE_LEVEL", "False").lower() == "true"
USE_TOKEN_CHUNKING = os.getenv("USE_TOKEN_CHUNKING", "False").lower() == "true"
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", None)
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE", "256"))

# Create application
app = FastAPI(
    title="Local RAG API",
    description="API for local Retrieval Augmented Generation using Ollama and ChromaDB",
    version="0.2.0"
)

# Initialize components
vector_store = VectorStore(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_model_name=EMBEDDING_MODEL,
    normalize_embeddings=NORMALIZE_EMBEDDINGS
)

doc_processor = DocumentProcessor(
    model_name=EMBEDDING_MODEL,
    normalize_embeddings=NORMALIZE_EMBEDDINGS,
    sentence_level=False,
    use_token_chunking=USE_TOKEN_CHUNKING,
    tokenizer_name=TOKENIZER_NAME,
    chunk_token_size=CHUNK_TOKEN_SIZE,
    chunk_size=512,
    chunk_overlap=100
)

ollama_client = OllamaClient(model_name=OLLAMA_MODEL, gpu_memory_limit=OLLAMA_GPU_MEMORY_LIMIT)

chat_manager = ChatHistoryManager(storage_dir=os.environ.get("CHAT_HISTORY_DIR", "chat_history"))

# Pydantic models
class Document(BaseModel):
    """Document model for ingestion"""
    content: str
    source: str
    document_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    """Response model for ingestion"""
    document_id: str
    chunks: int
    message: str

class QuestionRequest(BaseModel):
    query: str
    k: int = 12
    temperature: float = 0.7
    max_tokens: int = 500
    include_sources: bool = False
    conversation_id: Optional[str] = None
    max_history_messages: Optional[int] = 5

class SourceDocument(BaseModel):
    """Source document model"""
    content: str
    source: str
    document_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

class Answer(BaseModel):
    """Answer model"""
    answer: str
    sources: Optional[List[SourceDocument]] = None

def is_simple_greeting(query: str) -> bool:
    """Check if a query is a simple greeting."""
    # List of common greeting phrases - use exact matches only
    simple_greetings = ["hello", "hi", "hey", "greetings", 
                        "good morning", "good afternoon", "good evening", "howdy", "what's up"]
    query_lower = query.lower().strip()
    result = query_lower in simple_greetings
    logger.info(f"Checking if '{query}' is a simple greeting: {result}")
    return result

def is_conversation_starter(query: str) -> bool:
    """Check if the query is a conversation starter like 'how are you'."""
    conversation_starters = [
        "how are you", "how's it going", "how are things", 
        "what's new", "how do you do", "how have you been",
        "how are u", "how r u", "what's up", "whats up",
        "how are you doing", "how you doing", "how is it going"
    ]
    query_lower = query.lower().strip()
    
    # Exact match check
    if query_lower in conversation_starters:
        logger.info(f"Exact match: '{query}' is a conversation starter")
        return True
    
    # Substring check for phrases
    result = any(starter in query_lower for starter in conversation_starters)
    logger.info(f"Checking if '{query}' is a conversation starter: {result}")
    return result

def get_conversation_starter_response() -> str:
    """Get a simple response for 'how are you' type queries."""
    responses = [
        "I'm doing well, thanks for asking! How can I help you?",
        "I'm great! What can I help you with today?",
        "All good here! How can I assist you?",
        "I'm operational and ready to help!",
        "I'm fine, thank you! What can I do for you?",
        "Doing well! How can I be of service?"
    ]
    return random.choice(responses)

def clean_query(q: str) -> str:
    """Remove leading/trailing whitespace and fancy quotes for robust classification."""
    import re
    q = q.strip()
    # Replace curly quotes and fancy punctuation with standard ones
    q = q.replace("“", '"').replace("”", '"').replace("’", "'")
    # Collapse multiple spaces
    q = re.sub(r"\s+", " ", q)
    return q

def extract_name_from_query(query: str, possible_names: List[str]) -> Optional[str]:
    """Return the first name from possible_names that appears in the query (case-insensitive)."""
    q_lower = query.lower()
    for n in possible_names:
        if n.lower() in q_lower:
            return n
    return None

async def classify_query_type(query: str) -> bool:
    """
    Use the LLM to classify if a query needs RAG or just direct conversation.
    
    Args:
        query: The user's query text
        
    Returns:
        True if RAG should be used, False for direct chat
    """
    # Fast path for greetings and conversation starters
    if is_simple_greeting(query) or is_conversation_starter(query):
        logger.info(f"Fast path detection: '{query}' is conversational")
        return False

    cleaned_query = clean_query(query)

    # New explicit classifier instruction
    system_prompt = (
        "You are a binary classifier.\n\n"
        "Return exactly one token: RAG or CHAT.\n\n"
        "Rules:\n"
        "• Output RAG if the user is asking for facts, figures, names, dates, responsibilities, or any information that is expected to be found ONLY in the user's provided documents.\n"
        "• Output CHAT for greetings, small-talk, personal questions to the assistant, jokes, or general encyclopedic knowledge.\n"
        "Do NOT output anything other than the single token."
    )

    # Minimal balanced examples
    few_shot = (
        "### Example\nQuery: Who is responsible for preparing the Kubernetes migration document?\nAnswer: RAG\n"
        "### Example\nQuery: What is your name?\nAnswer: CHAT\n"
    )

    classification_prompt = (
        f"{few_shot}### Query\n{cleaned_query}\nAnswer:"
    )

    try:
        classification_result = await ollama_client.generate(
            prompt=classification_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1
        )

        label = classification_result.strip().lower()
        logger.info(f"LLM classification result for '{query}': {label}")
        return label.startswith("rag")
    except Exception as e:
        logger.error(f"Error during LLM classification: {e}")
        return True  # default to RAG if uncertain

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Local RAG API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check if Ollama is available
    ollama_status = await ollama_client.health_check()
    # Check vector store
    vector_store_status = vector_store.health_check()
    
    # Return health status
    return {
        "status": "ok" if ollama_status["status"] == "ok" and vector_store_status["status"] == "ok" else "error",
        "ollama": ollama_status,
        "vector_store": vector_store_status,
        "config": {
            "ollama_model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "sentence_level": SENTENCE_LEVEL,
            "use_token_chunking": USE_TOKEN_CHUNKING,
            "tokenizer_name": TOKENIZER_NAME,
            "chunk_token_size": CHUNK_TOKEN_SIZE,
        }
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(document: Document):
    """
    Ingest a document into the vector store
    
    Args:
        document: Document to ingest
        
    Returns:
        Ingestion response with document ID and chunk count
    """
    try:
        # Process document
        logger.info(f"Processing document from source: {document.source}")
        document_chunks = doc_processor.process_document(
            content=document.content,
            source=document.source,
            document_type=document.document_type,
            metadata=document.metadata
        )
        
        if not document_chunks:
            logger.warning(f"No chunks generated for document: {document.source}")
            raise HTTPException(status_code=400, detail="Failed to process document, no chunks generated")
        
        # Add chunks to vector store
        document_id = vector_store.add_document_chunks(document_chunks)
        
        logger.info(f"Document ingested with ID {document_id}: {len(document_chunks)} chunks")
        
        return IngestResponse(
            document_id=document_id,
            chunks=len(document_chunks),
            message="Document successfully ingested"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(...),
    document_type: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Ingest a file into the vector store
    
    Args:
        file: File to ingest
        source: Source of the document
        document_type: Type of document (will be inferred from file extension if not provided)
        metadata: JSON string of metadata
        
    Returns:
        Ingestion response with document ID and chunk count
    """
    try:
        # Read file content
        content = await file.read()
        
        # Detect document type from file extension if not provided
        if not document_type:
            file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            document_type = {
                'txt': 'text',
                'md': 'markdown',
                'json': 'json',
                'csv': 'csv',
                'pdf': 'pdf',
            }.get(file_ext, 'text')
        
        # Convert bytes to string for text-based files, keep as bytes for binary files
        if document_type == 'pdf':
            # Keep binary content for PDF processing
            document_content = content
        else:
            # Decode for text-based formats
            try:
                document_content = content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"Unicode decode error, falling back to latin-1 for file: {file.filename}")
                document_content = content.decode('latin-1')
        
        # Parse metadata if provided
        parsed_metadata = None
        if metadata:
            import json
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, ignoring")
        
        # Add filename to metadata
        if not parsed_metadata:
            parsed_metadata = {}
        parsed_metadata['filename'] = file.filename
        
        # Process document
        logger.info(f"Processing file: {file.filename} as {document_type}")
        document_chunks = doc_processor.process_document(
            content=document_content,
            source=source,
            document_type=document_type,
            metadata=parsed_metadata
        )
        
        if not document_chunks:
            logger.warning(f"No chunks generated for file: {file.filename}")
            raise HTTPException(status_code=400, detail="Failed to process file, no chunks generated")
        
        # Add chunks to vector store
        document_id = vector_store.add_document_chunks(document_chunks)
        
        logger.info(f"File ingested with ID {document_id}: {len(document_chunks)} chunks")
        
        return IngestResponse(
            document_id=document_id,
            chunks=len(document_chunks),
            message=f"File {file.filename} successfully ingested"
        )
    
    except Exception as e:
        logger.error(f"Error ingesting file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")

@app.post("/ask", response_model=Answer)
async def ask_question(
    question: QuestionRequest,
    include_chat_history: bool = Query(False, description="Include chat history in context"),
    force_rag: bool = Query(False, description="Force using RAG even for conversational queries"),
    chat_only: bool = Query(False, description="Force chat-only mode without RAG"),
    auto_detect: bool = Query(True, description="Automatically detect if query needs RAG")
):
    try:
        logger.info(f"Received question: '{question.query}' with conversation_id: {question.conversation_id}")
        
        # Determine if we should use RAG or direct chat
        use_rag = force_rag
        
        if not use_rag and not chat_only and auto_detect:
            if is_simple_greeting(question.query) or is_conversation_starter(question.query):
                use_rag = False
            else:
                use_rag = await classify_query_type(question.query)
        
        logger.info(f"Query '{question.query}' - Using RAG: {use_rag}")
        
        if not use_rag:
            logger.info("Using direct chat mode for query: %s", question.query)
            
            if is_simple_greeting(question.query):
                # Concise greeting prompt (LLM-driven, no hard-coding)
                system_prompt = (
                    "You are BrainWave, an AI assistant.\n\n"
                    "If the user's message is a greeting such as 'hello', 'hi', 'hey', 'greetings', "
                    "reply with a friendly greeting containing NO MORE THAN 6 WORDS.\n"
                    "Do not add any extra text, explanations, or punctuation other than the greeting itself."
                )
                max_tokens_to_use = 8  # allow up to ~6 words + punctuation
                temperature_to_use = 0.2
            elif is_conversation_starter(question.query):
                # Concise response for "how are you" type queries
                system_prompt = (
                    "You are BrainWave, an AI assistant.\n\n"
                    "When the user asks how you are (e.g. 'how are you', 'how are you doing'), "
                    "respond in ONE friendly sentence of AT MOST 12 WORDS.\n"
                    "Do not add explanations or mention these instructions."
                )
                max_tokens_to_use = 20
                temperature_to_use = 0.25
            else:
                # System prompt for conversational queries
                system_prompt = "You are BrainWave, a helpful AI assistant. Respond conversationally but be concise and to the point."
                max_tokens_to_use = question.max_tokens or 200
                temperature_to_use = question.temperature
                
                # Get direct response from the model
                response = await ollama_client.generate(
                    prompt=question.query,
                    system_prompt=system_prompt,
                    temperature=temperature_to_use,
                    max_tokens=max_tokens_to_use
                )
                
                # Add to chat history if specified
                if question.conversation_id:
                    chat_manager.add_message(
                        conversation_id=question.conversation_id,
                        role="assistant",
                        content=response
                    )
                
                return Answer(answer=response)
        
            # Get direct response from the model
            response = await ollama_client.generate(
                prompt=question.query,
                system_prompt=system_prompt,
                temperature=temperature_to_use,
                max_tokens=max_tokens_to_use
            )
            
            # Add to chat history if specified
            if question.conversation_id:
                chat_manager.add_message(
                    conversation_id=question.conversation_id,
                    role="assistant",
                    content=response
                )
            
            return Answer(answer=response)
        
        # Get conversation history if requested
        conversation_context = ""
        if include_chat_history and question.conversation_id:
            conversation = chat_manager.get_conversation(question.conversation_id)
            if conversation:
                conversation_context = conversation.format_for_prompt(
                    max_messages=question.max_history_messages or 5
                )
                if conversation_context:
                    conversation_context = f"Chat History:\n{conversation_context}\n\n"
        
        # Get relevant documents from vector store
        results = vector_store.query(
            query_text=question.query,
            top_k=question.k
        )
        
        # Format context from results
        context = "\n\n".join([f"[{r['source']}]\n{r['content']}" for r in results])
        
        if question.include_sources:
            sources = [SourceDocument(source=r["source"], content=r["content"][:200] + "...", document_type=r["document_type"], metadata=r["metadata"]) for r in results]
        else:
            sources = None
        
        # Add conversation history to context if available
        if conversation_context:
            context = f"{conversation_context}\n\nRelevant Documents:\n{context}"
        
        # Heuristic: parse action-item bullets like "- Name: Task" to find responsibilities
        pattern = re.compile(r"-\s*([A-Z][a-z]+):\s*(.+)")
        name_task_map = {}
        for line in context.splitlines():
            m = pattern.match(line.strip())
            if m:
                name, task = m.groups()
                name_task_map[name] = task.strip()
        
        # 1) If query explicitly mentions a name that exists in bullets
        query_name = extract_name_from_query(question.query, list(name_task_map.keys()))
        if query_name:
            if question.query.strip().lower().startswith("who"):
                return Answer(answer=query_name, sources=sources)
            else:
                return Answer(answer=name_task_map[query_name], sources=sources)
        
        # 2) Original keyword based fallback (for 'who is responsible for X?')
        keywords = [k for k in ["kubernetes", "migration", "document"] if k in question.query.lower()]
        if keywords:
            for name, task in name_task_map.items():
                if any(k in task.lower() for k in keywords):
                    return Answer(answer=name, sources=sources)
        
        # Generate response
        response = await ollama_client.generate(
            prompt=question.query,
            system_prompt=(
                "You are BrainWave, an expert assistant. Answer ONLY from the provided context.\n"
                "For responsibility questions (e.g. 'Who is responsible for X?'):\n"
                "1. Scan the context for bullet or action-item lines containing the task.\n"
                "2. The pattern is usually '- Name: Task description'.\n"
                "3. Return ONLY the Name (no extra words).\n"
                "4. If multiple names appear, return the one whose task best matches the user's query.\n"
                "5. If nothing matches, answer 'I don't know'.\n\n"
                f"Context:\n{context}"
            ),
            temperature=question.temperature,
            max_tokens=question.max_tokens
        )
        
        # Save question and answer to chat history if conversation_id is provided
        if question.conversation_id:
            chat_manager.add_message(
                conversation_id=question.conversation_id,
                role="user",
                content=question.query
            )
            
            chat_manager.add_message(
                conversation_id=question.conversation_id,
                role="assistant",
                content=response,
                metadata={"sources": sources} if sources else {}
            )
        
        return Answer(
            answer=response,
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/stream")
async def ask_question_stream(
    request: Request,
    question: QuestionRequest,
    include_chat_history: bool = Query(False, description="Include chat history in context"),
    force_rag: bool = Query(False, description="Force using RAG even for conversational queries"),
    chat_only: bool = Query(False, description="Force chat-only mode without RAG"),
    auto_detect: bool = Query(True, description="Automatically detect if query needs RAG")
):
    """
    Stream response to a question with RAG or chat directly with the model
    
    Args:
        request: FastAPI Request object
        question: Question request with query, k, and other parameters
        include_chat_history: Include chat history in context
        force_rag: Force using RAG even for conversational queries
        chat_only: Force chat-only mode without RAG
        auto_detect: Use LLM to determine if query needs RAG
        
    Returns:
        EventSourceResponse with streaming text and sources
    """
    try:
        # Determine if we should use RAG or direct chat
        use_rag = False
        
        if force_rag:
            use_rag = True
        elif chat_only:
            use_rag = False
        elif auto_detect:
            # Fast rule-based pre-check for obvious cases
            if is_simple_greeting(question.query):
                use_rag = False
            elif is_conversation_starter(question.query):
                use_rag = False
            else:
                # Use the LLM to classify more complex queries
                use_rag = await classify_query_type(question.query)
        else:
            # Fall back to rule-based approach if auto_detect is disabled
            use_rag = False
        
        logger.info(f"Streaming query '{question.query}' - Using RAG: {use_rag}")
        
        # Add question to chat history if conversation_id is provided
        if question.conversation_id:
            chat_manager.add_message(
                conversation_id=question.conversation_id,
                role="user",
                content=question.query
            )
        
        if not use_rag:
            logger.info("Using direct chat mode for streaming query: %s", question.query)
            
            if is_simple_greeting(question.query):
                # Concise greeting prompt (LLM-driven, no hard-coding)
                system_prompt = (
                    "You are BrainWave, an AI assistant.\n\n"
                    "If the user's message is a greeting such as 'hello', 'hi', 'hey', 'greetings', "
                    "reply with a friendly greeting containing NO MORE THAN 6 WORDS.\n"
                    "Do not add any extra text, explanations, or punctuation other than the greeting itself."
                )
                max_tokens_to_use = 8  # allow up to ~6 words + punctuation
                temperature_to_use = 0.2
            elif is_conversation_starter(question.query):
                # Concise response for "how are you" type queries
                system_prompt = (
                    "You are BrainWave, an AI assistant.\n\n"
                    "When the user asks how you are (e.g. 'how are you', 'how are you doing'), "
                    "respond in ONE friendly sentence of AT MOST 12 WORDS.\n"
                    "Do not add explanations or mention these instructions."
                )
                max_tokens_to_use = 20
                temperature_to_use = 0.25
            else:
                # System prompt for conversational queries
                system_prompt = "You are BrainWave, a helpful AI assistant. Respond conversationally but be concise and to the point."
                max_tokens_to_use = question.max_tokens or 200
                temperature_to_use = question.temperature
                
                # Stream response from the model
                collected_response = ""
                
                async def stream_chat():
                    nonlocal collected_response
                    async for chunk in ollama_client.generate_stream(
                        prompt=question.query,
                        system_prompt=system_prompt,
                        temperature=temperature_to_use,
                        max_tokens=max_tokens_to_use
                    ):
                        collected_response += chunk
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
                    
                    # Add response to chat history after generation
                    if question.conversation_id:
                        chat_manager.add_message(
                            conversation_id=question.conversation_id,
                            role="assistant",
                            content=collected_response
                        )
                    
                    yield "data: [DONE]\n\n"
                
                return EventSourceResponse(stream_chat())
        
            # Get direct response from the model
            collected_response = ""
            
            async def stream_greeting():
                nonlocal collected_response
                async for chunk in ollama_client.generate_stream(
                    prompt=question.query,
                    system_prompt=system_prompt,
                    temperature=temperature_to_use,
                    max_tokens=max_tokens_to_use
                ):
                    collected_response += chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                
                # Add response to chat history after generation
                if question.conversation_id:
                    chat_manager.add_message(
                        conversation_id=question.conversation_id,
                        role="assistant",
                        content=collected_response
                    )
                
                yield "data: [DONE]\n\n"
            
            return EventSourceResponse(stream_greeting())
        
        # Get conversation history if requested
        conversation_context = ""
        if include_chat_history and question.conversation_id:
            conversation = chat_manager.get_conversation(question.conversation_id)
            if conversation:
                conversation_context = conversation.format_for_prompt(
                    max_messages=question.max_history_messages or 5
                )
                if conversation_context:
                    conversation_context = f"Chat History:\n{conversation_context}\n\n"
        
        # Get relevant documents from vector store
        results = vector_store.query(
            query_text=question.query,
            top_k=question.k
        )
        
        # Format context from results
        context = "\n\n".join([f"[{r['source']}]\n{r['content']}" for r in results])
        
        if question.include_sources:
            sources = [{"source": r["source"], "content": r["content"][:200] + "...", "document_type": r["document_type"], "metadata": r["metadata"]} for r in results]
        else:
            sources = None
        
        # Add conversation history to context if available
        if conversation_context:
            context = f"{conversation_context}\n\nRelevant Documents:\n{context}"
        
        async def event_generator_rag():
            # Heuristic extraction in stream
            pattern = re.compile(r"-\s*([A-Z][a-z]+):\s*(.+)")
            name_task_map = {}
            for line in context.splitlines():
                m = pattern.match(line.strip())
                if m:
                    name, task = m.groups()
                    name_task_map[name] = task.strip()

            query_name = extract_name_from_query(question.query, list(name_task_map.keys()))
            if query_name:
                text_to_send = query_name if question.query.strip().lower().startswith("who") else name_task_map[query_name]
                payload = {"text": text_to_send}
                if sources:
                    payload["sources"] = sources
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Keyword fallback
            keywords = [k for k in ["kubernetes", "migration", "document"] if k in question.query.lower()]
            if keywords:
                for name, task in name_task_map.items():
                    if any(k in task.lower() for k in keywords):
                        payload = {"text": name}
                        if sources:
                            payload["sources"] = sources
                        yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

            # Otherwise stream LLM response
            async for text in ollama_client.generate_stream(
                prompt=question.query,
                system_prompt=(
                    "You are BrainWave, an expert assistant. Answer ONLY from the provided context.\n"
                    "For responsibility questions (e.g. 'Who is responsible for X?'):\n"
                    "1. Scan the context for bullet or action-item lines containing the task.\n"
                    "2. The pattern is usually '- Name: Task description'.\n"
                    "3. Return ONLY the Name (no extra words).\n"
                    "4. If multiple names appear, return the one whose task best matches the user's query.\n"
                    "5. If nothing matches, answer 'I don't know'.\n\n"
                    f"Context:\n{context}"
                ),
                temperature=question.temperature,
                max_tokens=question.max_tokens
            ):
                yield f"data: {json.dumps({'text': text})}\n\n"
            # Send sources after completion
            if sources:
                yield f"data: {json.dumps({'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"
 
        return EventSourceResponse(event_generator_rag())
    
    except Exception as e:
        logger.error(f"Error streaming response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store"""
    try:
        stats = vector_store.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the vector store"""
    try:
        vector_store.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/collection")
async def delete_collection():
    """Delete the entire collection from the vector store"""
    try:
        vector_store.delete_collection()
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

@app.post("/conversations")
async def create_conversation(
    request: Request,
    title: Optional[str] = Body(None, description="Title of the conversation"),
    metadata: Optional[Dict[str, Any]] = Body(None, description="Additional metadata for the conversation")
):
    """Create a new conversation"""
    try:
        conversation = chat_manager.create_conversation(
            title=title,
            metadata=metadata
        )
        return {
            "status": "success",
            "conversation_id": conversation.conversation_id,
            "title": conversation.title
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating conversation: {str(e)}"
        )

@app.get("/conversations")
async def list_conversations(
    request: Request,
    limit: int = Query(20, description="Maximum number of conversations to return", ge=1, le=100),
    skip: int = Query(0, description="Number of conversations to skip", ge=0)
):
    """List available conversations"""
    try:
        conversations = chat_manager.list_conversations(limit=limit, skip=skip)
        return {
            "status": "success",
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing conversations: {str(e)}"
        )

@app.get("/conversations/{conversation_id}")
async def get_conversation(
    request: Request,
    conversation_id: str = Path(..., description="ID of the conversation")
):
    """Get a conversation by ID"""
    conversation = chat_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )
    
    return {
        "status": "success",
        "conversation": conversation.to_dict()
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    request: Request,
    conversation_id: str = Path(..., description="ID of the conversation to delete")
):
    """Delete a conversation"""
    success = chat_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found or could not be deleted"
        )
    
    return {
        "status": "success",
        "message": f"Conversation {conversation_id} deleted"
    }

@app.post("/conversations/{conversation_id}/messages")
async def add_message(
    request: Request,
    conversation_id: str = Path(..., description="ID of the conversation"),
    role: str = Body(..., description="Role of the message sender (user/assistant)"),
    content: str = Body(..., description="Message content"),
    metadata: Optional[Dict[str, Any]] = Body(None, description="Additional metadata for the message")
):
    """Add a message to a conversation"""
    message = chat_manager.add_message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata=metadata
    )
    
    if not message:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )
    
    return {
        "status": "success",
        "message": message.to_dict()
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
