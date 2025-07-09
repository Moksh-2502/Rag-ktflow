"""
Vector store using ChromaDB
"""
import logging
import os
import uuid
from typing import Dict, List, Any, Optional, Union

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store implementation using ChromaDB
    """
    
    def __init__(
        self, 
        persist_directory: str = "chroma_db", 
        collection_name: str = "documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
            embedding_model_name: Model name for SentenceTransformer embeddings
            normalize_embeddings: Whether to normalize embeddings
        """
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_model_name = embedding_model_name
            self.normalize_embeddings = normalize_embeddings
            
            logger.info(f"Initialized vector store with model {embedding_model_name} and collection {collection_name}")
            logger.info(f"ChromaDB persistence directory: {os.path.abspath(persist_directory)}")
            
            # Check collection stats
            stats = self.get_stats()
            logger.info(f"Collection has {stats['count']} documents")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_document_chunks(self, document_chunks: List[Any]) -> str:
        """
        Add document chunks to vector store
        
        Args:
            document_chunks: List of DocumentChunk objects
            
        Returns:
            Document ID for the added chunks
        """
        if not document_chunks:
            logger.warning("No document chunks to add")
            return ""
            
        try:
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Extract data from chunks
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in document_chunks:
                # Create unique ID for chunk
                chunk_id = f"{document_id}_{chunk.id}"
                
                # Prepare metadata with source
                metadata = self._sanitize_metadata(chunk.metadata or {})
                
                # Add common metadata fields
                metadata["document_id"] = document_id
                metadata["source"] = metadata.get("source", "unknown")
                metadata["document_type"] = metadata.get("document_type", "text")
                
                # Add token count metadata if available
                if hasattr(chunk, "token_count") and chunk.token_count is not None:
                    metadata["token_count"] = chunk.token_count
                    
                # Add word and character counts
                if hasattr(chunk, "word_count"):
                    metadata["word_count"] = chunk.word_count
                if hasattr(chunk, "char_count"):
                    metadata["char_count"] = chunk.char_count
                
                # Add hash if available
                if hasattr(chunk, "hash") and chunk.hash:
                    metadata["hash"] = chunk.hash
                
                # Add to lists
                ids.append(chunk_id)
                documents.append(chunk.content)
                metadatas.append(metadata)
                
                # Use pre-computed embedding if available, otherwise will compute later
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
            
            # If all chunks have embeddings, use them directly
            if embeddings and len(embeddings) == len(documents):
                logger.info("Using pre-computed embeddings for document chunks")
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB compute embeddings using its embedding function
                logger.info("Computing embeddings for document chunks via ChromaDB")
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Added {len(documents)} chunks with document ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document chunks: {e}")
            raise
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure it's compatible with ChromaDB
        
        Args:
            metadata: Original metadata
            
        Returns:
            Sanitized metadata
        """
        if not metadata:
            return {}
            
        # ChromaDB only supports str, int, float, bool as metadata values
        result = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value):
                # Convert lists of primitive types to strings
                result[key] = str(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries recursively
                flat_dict = self._flatten_dict(value, parent_key=key)
                result.update(flat_dict)
            else:
                # Convert other types to string
                try:
                    result[key] = str(value)
                except Exception:
                    logger.warning(f"Could not convert metadata value for key '{key}' to string, skipping")
        
        return result
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """
        Flatten nested dictionary for metadata storage
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested dictionary
            
        Returns:
            Flattened dictionary
        """
        items = {}
        
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                items.update(self._flatten_dict(value, new_key))
            elif isinstance(value, (str, int, float, bool)):
                # Store primitive types directly
                items[new_key] = value
            else:
                # Convert other types to string
                try:
                    items[new_key] = str(value)
                except Exception:
                    logger.warning(f"Could not convert nested metadata value for key '{new_key}' to string, skipping")
        
        return items
    
    def query(
        self, 
        query_text: str, 
        top_k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query vector store for similar documents
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            filter: Optional filter for metadata
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of documents with content, metadata and optionally embeddings
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode(
                query_text, 
                normalize_embeddings=self.normalize_embeddings
            ).tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter,
                include=["documents", "metadatas"] + (["embeddings"] if include_embeddings else [])
            )
            
            # Format results
            formatted_results = []
            
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "document_type": results["metadatas"][0][i].get("document_type", "text")
                }
                
                # Include embedding if requested
                if include_embeddings and "embeddings" in results:
                    result["embedding"] = results["embeddings"][0][i]
                    
                formatted_results.append(result)
            
            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document by ID
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            Success status
        """
        try:
            # Delete all chunks with matching document_id in metadata
            self.collection.delete(
                where={"document_id": document_id}
            )
            logger.info(f"Deleted document with ID: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection
        
        Returns:
            Success status
        """
        try:
            self.client.delete_collection(self.collection.name)
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Deleted and recreated collection: {self.collection.name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Statistics dictionary
        """
        try:
            # Get count of documents
            count = self.collection.count()
            
            # Get unique sources
            results = self.collection.get(
                include=["metadatas"],
                limit=10000  # Reasonable limit to avoid memory issues
            )
            
            sources = set()
            document_types = set()
            document_ids = set()
            
            for metadata in results["metadatas"]:
                if "source" in metadata:
                    sources.add(metadata["source"])
                if "document_type" in metadata:
                    document_types.add(metadata["document_type"])
                if "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
            
            return {
                "count": count,
                "sources": list(sources),
                "document_types": list(document_types),
                "unique_documents": len(document_ids),
                "embedding_model": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "count": 0,
                "sources": [],
                "document_types": [],
                "unique_documents": 0,
                "embedding_model": self.embedding_model_name,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the vector store is healthy
        
        Returns:
            Health status dictionary
        """
        try:
            # Try to get collection count
            count = self.collection.count()
            
            return {
                "status": "ok",
                "message": f"Vector store is healthy with {count} documents",
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "error",
                "message": f"Vector store health check failed: {str(e)}",
                "collection_name": getattr(self.collection, "name", "unknown"),
                "embedding_model": self.embedding_model_name
            }
