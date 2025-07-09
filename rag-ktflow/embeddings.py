"""
Document processing and embedding module with optimized embedding generation,
validation, and enhanced document structure.
"""
import os
import re
import uuid
import json
import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import markdown
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt

# Add tokenizer support
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")
    
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

# Add PDF support
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install with: pip install PyPDF2")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Document chunk with content, metadata, and embedding"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: str = field(default="")
    word_count: int = field(default=0)
    char_count: int = field(default=0)
    token_count: Optional[int] = field(default=None)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Calculate metrics after initialization"""
        if self.content and not self.word_count:
            self.word_count = len(self.content.split())
        if self.content and not self.char_count:
            self.char_count = len(self.content)
        if not self.hash and self.content:
            self.hash = self.compute_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Remove embedding from dict if present
        if result["embedding"] is not None:
            del result["embedding"]
        return result
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of content for deduplication"""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

class DocumentProcessor:
    """Handles document processing, splitting, and embedding"""
    
    VALID_DOCUMENT_TYPES = ["text", "markdown", "json", "csv", "pdf"]
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        normalize_embeddings: bool = True,
        sentence_level: bool = False,
        tokenizer_name: Optional[str] = None,
        use_token_chunking: bool = False,
        chunk_token_size: int = 256,
    ):
        """
        Initialize document processor
        
        Args:
            model_name: Name of the sentence transformer model
            chunk_size: Maximum size of document chunks in characters or tokens
            chunk_overlap: Overlap between chunks in characters or tokens
            normalize_embeddings: Whether to normalize embeddings (recommended)
            sentence_level: Whether to split by sentences first
            tokenizer_name: Name of the HuggingFace tokenizer to use
            use_token_chunking: Whether to use token-based chunking
            chunk_token_size: Size of text chunks in tokens (when use_token_chunking is True)
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.normalize_embeddings = normalize_embeddings
            self.sentence_level = sentence_level
            self.use_token_chunking = use_token_chunking
            self.chunk_token_size = chunk_token_size
            
            # Initialize tokenizer if token chunking is enabled
            self.tokenizer = None
            if use_token_chunking:
                if TIKTOKEN_AVAILABLE and (not tokenizer_name or tokenizer_name in ["gpt-3.5-turbo", "gpt-4", "cl100k_base"]):
                    # Use tiktoken for OpenAI models
                    encoding_name = "cl100k_base"
                    if tokenizer_name and tokenizer_name not in ["cl100k_base"]:
                        try:
                            encoding_name = tiktoken.encoding_for_model(tokenizer_name).name
                        except Exception:
                            pass
                            
                    logger.info(f"Using tiktoken with encoding: {encoding_name}")
                    self.tokenizer = tiktoken.get_encoding(encoding_name)
                elif TRANSFORMERS_AVAILABLE and tokenizer_name:
                    # Use HuggingFace tokenizer
                    logger.info(f"Loading HuggingFace tokenizer: {tokenizer_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                elif TRANSFORMERS_AVAILABLE:
                    # Default to the same model's tokenizer if available
                    try:
                        logger.info(f"Attempting to load tokenizer for model: {model_name}")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    except Exception as e:
                        logger.warning(f"Could not load tokenizer for {model_name}, falling back to character chunking: {e}")
                        self.use_token_chunking = False
                else:
                    logger.warning("Token chunking enabled but no tokenizer available, falling back to character chunking")
                    self.use_token_chunking = False
            
            logger.info(f"Initialized document processor with model {model_name}")
            logger.info(f"Settings: chunk_size={chunk_size}, overlap={chunk_overlap}, normalize={normalize_embeddings}")
            
            if use_token_chunking and self.tokenizer:
                logger.info(f"Using token-based chunking with tokenizer: {tokenizer_name or model_name}")
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}")
            raise
    
    def process_document(
        self, 
        content: str, 
        source: str, 
        document_type: str = "text", 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process document content into chunks with metadata
        
        Args:
            content: Document content
            source: Source identifier for the document
            document_type: Type of document (text, markdown, json, csv, pdf)
            metadata: Additional metadata to include
            
        Returns:
            List of document chunks with processed content and metadata
        """
        try:
            # Validate document type
            if document_type not in self.VALID_DOCUMENT_TYPES:
                logger.warning(f"Invalid document type: {document_type}. Defaulting to text.")
                document_type = "text"
            
            # Sanitize content
            content = self._sanitize_content(content)
            
            # Convert content based on document type
            processed_content = self._process_content_by_type(content, document_type)
            
            # Split into chunks based on configuration
            if self.sentence_level:
                chunks = self._split_sentences(processed_content)
                logger.info(f"Split document into {len(chunks)} sentence-level chunks")
            elif self.use_token_chunking and self.tokenizer:
                chunks = self._split_by_tokens(processed_content)
                logger.info(f"Split document into {len(chunks)} token-based chunks")
            else:
                chunks = self._split_text(processed_content)
                logger.info(f"Split document into {len(chunks)} character-based chunks")
            
            # Create document chunks
            doc_chunks = []
            chunk_hashes = set()  # For deduplication
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # Create base metadata
                chunk_metadata = {
                    "source": source,
                    "document_type": document_type,
                    "chunk_index": i
                }
                
                # Add custom metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                
                # Create document chunk
                doc_chunk = DocumentChunk(
                    content=chunk,
                    metadata=chunk_metadata
                )
                
                # Add token count if tokenizer is available
                if self.tokenizer:
                    if hasattr(self.tokenizer, "encode"):
                        # tiktoken style
                        doc_chunk.token_count = len(self.tokenizer.encode(chunk))
                    elif hasattr(self.tokenizer, "tokenize"):
                        # HuggingFace style
                        doc_chunk.token_count = len(self.tokenizer.tokenize(chunk))
                
                # Skip duplicate chunks
                if doc_chunk.hash in chunk_hashes:
                    logger.debug(f"Skipping duplicate chunk with hash {doc_chunk.hash[:8]}")
                    continue
                    
                chunk_hashes.add(doc_chunk.hash)
                doc_chunks.append(doc_chunk)
            
            # Generate embeddings
            embeddings = self._batch_embed_chunks(doc_chunks)
            
            # Assign embeddings to chunks
            for i, embedding in enumerate(embeddings):
                if i < len(doc_chunks) and embedding is not None:
                    doc_chunks[i].embedding = embedding.tolist()
            
            return doc_chunks
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return []
    
    def _sanitize_content(self, content: str) -> str:
        """
        Sanitize content by removing control characters
        
        Args:
            content: Raw content
            
        Returns:
            Sanitized content
        """
        if not content:
            return ""
            
        try:
            # Remove control characters except newlines and tabs
            sanitized = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", content)
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing content: {e}")
            return content
    
    def _process_content_by_type(self, content: str, document_type: str) -> str:
        """
        Process content based on document type
        
        Args:
            content: Raw document content
            document_type: Type of document (text, markdown, json, csv, pdf)
            
        Returns:
            Processed plain text content
        """
        if document_type == "text":
            # Plain text, no processing needed
            return content
            
        elif document_type == "markdown":
            # Convert markdown to plain text
            try:
                html = markdown.markdown(content)
                # Simple HTML tag removal
                text = re.sub(r"<[^>]*>", "", html)
                return text
            except Exception as e:
                logger.error(f"Error processing markdown: {e}")
                return content
                
        elif document_type == "json":
            # Extract text from JSON
            try:
                data = json.loads(content)
                return self._extract_text_from_json(data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON content: {e}")
                return content
                
        elif document_type == "csv":
            try:
                # Extract text from CSV
                result = []
                csv_reader = csv.reader(io.StringIO(content))
                header = next(csv_reader, None)
                
                if header:
                    for row in csv_reader:
                        if len(row) == len(header):
                            line = []
                            for i, field in enumerate(row):
                                line.append(f"{header[i]}: {field}")
                            result.append(", ".join(line))
                        else:
                            result.append(", ".join(row))
                else:
                    for row in csv_reader:
                        result.append(", ".join(row))
                        
                return "\n\n".join(result)
            except Exception as e:
                logger.error(f"Error processing CSV: {e}")
                return content
                
        elif document_type == "pdf":
            if PDF_AVAILABLE:
                try:
                    # Extract text from PDF
                    pdf_text = []
                    # For binary PDF content
                    if isinstance(content, bytes):
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    else:
                        # For string content that might be base64 or file path
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content.encode("utf-8", errors="ignore")))
                        
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text() or ""
                        pdf_text.append(page_text)
                    
                    return "\n\n".join(pdf_text)
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}")
                    return content
            else:
                logger.warning("PDF processing requested but PyPDF2 is not installed")
                return content
        else:
            # Plain text, no processing needed
            return content
    
    def _extract_text_from_json(self, data: Any) -> str:
        """
        Extract text from JSON structure
        
        Args:
            data: JSON data structure
            
        Returns:
            Extracted text
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, (int, float, bool)):
            return str(data)
        elif isinstance(data, dict):
            texts = []
            for key, value in data.items():
                # Include key-value pairs for better context
                if isinstance(value, (str, int, float, bool)):
                    texts.append(f"{key}: {value}")
                else:
                    # For nested structures, just extract the text
                    extracted = self._extract_text_from_json(value)
                    if extracted:
                        texts.append(extracted)
            return " ".join(texts)
        elif isinstance(data, list):
            texts = [self._extract_text_from_json(item) for item in data]
            return " ".join(text for text in texts if text)
        else:
            return ""
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks by character
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Get end position for this chunk
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at paragraph/sentence
            if end < len(text):
                # Try to find paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to find sentence break (period followed by space)
                    sentence_break = text.rfind(". ", start, end)
                    if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2
                    else:
                        # Try to find a space to break at
                        space_break = text.rfind(" ", start, end)
                        if space_break != -1 and space_break > start:
                            end = space_break + 1
            
            # Add chunk
            chunk = text[start:min(end, len(text))].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move to next chunk with overlap
            if end >= len(text):
                break
                
            start = end - self.chunk_overlap
            
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, then combine into chunks
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Simple sentence splitting (improved)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Current chunk would be too large, save it and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_len
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_length += sentence_len
                
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Split text by token count
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or not self.tokenizer:
            logger.warning("Token splitting requested but no tokenizer available")
            return self._split_text(text)
        
        try:
            chunks = []
            
            # Handle different tokenizer types
            if hasattr(self.tokenizer, "encode"):
                # tiktoken style
                tokens = self.tokenizer.encode(text)
                
                # Split tokens into chunks
                for i in range(0, len(tokens), self.chunk_token_size - self.chunk_overlap):
                    # Get chunk tokens
                    chunk_tokens = tokens[i:i + self.chunk_token_size]
                    # Decode back to text
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)
            
            elif hasattr(self.tokenizer, "tokenize") and hasattr(self.tokenizer, "convert_tokens_to_string"):
                # HuggingFace style
                tokens = self.tokenizer.tokenize(text)
                
                # Split tokens into chunks
                for i in range(0, len(tokens), self.chunk_token_size - self.chunk_overlap):
                    # Get chunk tokens
                    chunk_tokens = tokens[i:i + self.chunk_token_size]
                    # Convert back to text
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                    chunks.append(chunk_text)
            
            else:
                logger.warning("Unsupported tokenizer interface, falling back to text splitting")
                return self._split_text(text)
                
            return chunks
        except Exception as e:
            logger.error(f"Error in token-based splitting: {e}", exc_info=True)
            # Fall back to regular text splitting
            return self._split_text(text)
    
    @retry(stop=stop_after_attempt(3))
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        if not text:
            logger.warning("Attempted to embed empty text")
            return []
            
        try:
            embedding = self.model.encode(
                text, 
                normalize_embeddings=self.normalize_embeddings
            )
            return embedding.tolist()
        except Exception as e:
            # Log the failed text for debugging
            truncated_text = text[:100] + "..." if len(text) > 100 else text
            logger.error(f"Error embedding text: {e}")
            logger.debug(f"Failed text: {truncated_text}")
            raise  # Will be caught by retry decorator
    
    def _batch_embed_chunks(self, chunks: List[DocumentChunk]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple chunks in batch
        
        Args:
            chunks: Document chunks to embed
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
            
        texts = [chunk.content for chunk in chunks]
        
        try:
            # Batch processing with retry
            @retry(stop=stop_after_attempt(3))
            def batch_encode(batch_texts):
                return self.model.encode(
                    batch_texts,
                    batch_size=32,  # Process in reasonably sized batches
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=len(batch_texts) > 100  # Show progress for large batches
                )
                
            embeddings = batch_encode(texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error batch embedding chunks: {e}")
            
            # Try embedding one by one to identify problematic chunks
            results = []
            for i, text in enumerate(texts):
                try:
                    embedding = self.embed_text(text)
                    results.append(np.array(embedding) if embedding else None)
                except Exception as chunk_e:
                    logger.error(f"Failed to embed chunk {i}: {chunk_e}")
                    logger.debug(f"Problematic chunk (truncated): {text[:100]}...")
                    results.append(None)
            
            return results
