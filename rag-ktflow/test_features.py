"""
Test script for validating enhanced RAG system features
"""
import os
import logging
import tempfile
import unittest
from typing import Dict, List, Any

import pandas as pd
from embeddings import DocumentProcessor, DocumentChunk
from vectorstore import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRAGFeatures(unittest.TestCase):
    """Test cases for RAG system enhancements"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for vector store
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize vector store with test collection
        self.vector_store = VectorStore(
            persist_directory=self.temp_dir.name,
            collection_name="test_collection",
            embedding_model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize document processor with token chunking enabled
        self.doc_processor = DocumentProcessor(
            embedding_model="all-MiniLM-L6-v2",
            use_token_chunking=True,
            tokenizer_name="cl100k_base",  # OpenAI's tokenizer via tiktoken
            chunk_token_size=256
        )
        
        # Create sample documents
        self._create_sample_documents()
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def _create_sample_documents(self):
        """Create sample documents for testing"""
        # Text sample
        self.text_content = """
        # RAG System Test Document
        
        This is a sample document for testing the enhanced RAG system features.
        
        ## Key Features
        - Token-aware chunking
        - PDF document support
        - CSV document support
        - Improved metadata handling
        - Embedding retry logic
        
        The RAG system should properly process this document and split it into appropriate chunks based on token count rather than character count. This makes the chunking more accurate for LLM context windows.
        
        ### Technical Details
        
        The system uses tiktoken or HuggingFace tokenizers for token-aware chunking, which ensures that chunks respect token boundaries. This is important for optimizing context usage in LLMs.
        
        Error handling has been improved with retry logic for embedding generation, making the system more robust against transient failures.
        """
        
        # CSV sample
        self.csv_content = "name,role,department,years_experience\nJohn Doe,Developer,Engineering,5\nJane Smith,Manager,Product,8\nMike Johnson,Designer,UX,3\nSara Williams,Architect,Engineering,10"
        
        # PDF content (simulated as text for testing)
        self.pdf_content = """
        PDF TEST DOCUMENT
        
        This simulates content extracted from a PDF file.
        
        Page 1
        ------
        The RAG system now supports PDF documents using PyPDF2.
        This feature allows ingesting technical documentation, research papers, and other PDF-based content.
        
        Page 2
        ------
        Extracted text maintains its structure as much as possible.
        The system processes each page and concatenates the content.
        """
        
        # Save CSV to temp file
        self.csv_path = os.path.join(self.temp_dir.name, "test.csv")
        with open(self.csv_path, "w") as f:
            f.write(self.csv_content)
        
        # Save text to temp file
        self.text_path = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.text_path, "w") as f:
            f.write(self.text_content)
        
        # PDF would normally be binary, but we'll use text for testing
        self.pdf_path = os.path.join(self.temp_dir.name, "test.pdf")
        with open(self.pdf_path, "w") as f:
            f.write(self.pdf_content)
    
    def test_token_aware_chunking(self):
        """Test token-aware chunking functionality"""
        # Process with token chunking
        chunks_token = self.doc_processor.process_document(
            content=self.text_content,
            source="test_token.txt",
            document_type="text",
            metadata={"test_type": "token_chunking"}
        )
        
        # Verify chunks have token counts
        for chunk in chunks_token:
            self.assertIsNotNone(chunk.token_count, "Chunk should have token_count metadata")
            self.assertGreater(chunk.token_count, 0, "Token count should be positive")
            self.assertLessEqual(chunk.token_count, 256, "Token count should not exceed chunk_token_size")
        
        # Create a new processor with character chunking
        char_processor = DocumentProcessor(
            embedding_model="all-MiniLM-L6-v2",
            use_token_chunking=False,
            chunk_size=200
        )
        
        # Process with character chunking
        chunks_char = char_processor.process_document(
            content=self.text_content,
            source="test_char.txt",
            document_type="text",
            metadata={"test_type": "char_chunking"}
        )
        
        # Verify chunks differ between methods
        logger.info(f"Token chunking produced {len(chunks_token)} chunks")
        logger.info(f"Character chunking produced {len(chunks_char)} chunks")
        
        # The chunking results should be different
        self.assertNotEqual(
            len(chunks_token), 
            len(chunks_char), 
            "Token and character chunking should produce different results"
        )
    
    def test_csv_document_support(self):
        """Test CSV document processing"""
        # Process CSV content
        chunks = self.doc_processor.process_document(
            content=self.csv_content,
            source="test.csv",
            document_type="csv",
            metadata={"test_type": "csv_support"}
        )
        
        # Verify chunks contain CSV data
        self.assertGreater(len(chunks), 0, "Should have extracted chunks from CSV")
        
        # Check CSV parsing results
        csv_text = chunks[0].content
        self.assertIn("name: John Doe", csv_text, "CSV should be parsed into readable text format")
        self.assertIn("role: Developer", csv_text, "CSV should contain field-value pairs")
        
        # Verify metadata
        self.assertEqual(chunks[0].metadata["document_type"], "csv", "Document type should be CSV")
        self.assertEqual(chunks[0].metadata["source"], "test.csv", "Source should be preserved")
    
    def test_pdf_document_support(self):
        """Test PDF document processing"""
        # We're simulating PDF processing since actual PDF binary parsing requires PyPDF2
        # In a real scenario, this would read the actual PDF file
        
        chunks = self.doc_processor.process_document(
            content=self.pdf_content,
            source="test.pdf",
            document_type="pdf",
            metadata={"test_type": "pdf_support"}
        )
        
        # Verify chunks contain PDF data
        self.assertGreater(len(chunks), 0, "Should have extracted chunks from PDF")
        
        # Check if PDF content is present
        pdf_text = " ".join([chunk.content for chunk in chunks])
        self.assertIn("PDF TEST DOCUMENT", pdf_text, "Should contain PDF title")
        self.assertIn("PyPDF2", pdf_text, "Should contain key PDF content")
        
        # Verify metadata
        self.assertEqual(chunks[0].metadata["document_type"], "pdf", "Document type should be PDF")
        self.assertEqual(chunks[0].metadata["source"], "test.pdf", "Source should be preserved")
    
    def test_embedding_and_vectorstore(self):
        """Test embedding generation and vector store integration"""
        # Process a document
        chunks = self.doc_processor.process_document(
            content=self.text_content,
            source="vector_test.txt",
            document_type="text",
            metadata={"test_type": "vectorstore_integration"}
        )
        
        # Add to vector store
        doc_id = self.vector_store.add_document_chunks(chunks)
        
        # Verify document was added
        self.assertIsNotNone(doc_id, "Document ID should be returned")
        self.assertNotEqual(doc_id, "", "Document ID should not be empty")
        
        # Query the vector store
        results = self.vector_store.query(
            query_text="token chunking",
            top_k=3
        )
        
        # Verify query results
        self.assertGreater(len(results), 0, "Should return query results")
        
        # Test metadata was preserved
        self.assertEqual(results[0]["metadata"]["test_type"], "vectorstore_integration", "Metadata should be preserved")
        self.assertEqual(results[0]["metadata"]["document_type"], "text", "Document type should be preserved")
        
        # Test document stats
        stats = self.vector_store.get_stats()
        self.assertGreater(stats["count"], 0, "Vector store should contain documents")
        self.assertIn("text", stats["document_types"], "Document types should include 'text'")
    
    def test_enhanced_metadata(self):
        """Test enhanced metadata handling"""
        # Create document with rich metadata
        rich_metadata = {
            "author": "Test User",
            "created_at": "2023-06-15",
            "tags": ["test", "metadata", "enhancements"],
            "importance": 5,
            "nested": {
                "level1": {
                    "level2": "nested value"
                }
            }
        }
        
        chunks = self.doc_processor.process_document(
            content="This is a test document with rich metadata.",
            source="metadata_test.txt",
            document_type="text",
            metadata=rich_metadata
        )
        
        # Add to vector store
        doc_id = self.vector_store.add_document_chunks(chunks)
        
        # Query the vector store
        results = self.vector_store.query(
            query_text="rich metadata",
            top_k=1
        )
        
        # Verify metadata was flattened and preserved
        metadata = results[0]["metadata"]
        self.assertEqual(metadata["author"], "Test User", "Basic metadata should be preserved")
        self.assertEqual(metadata["importance"], 5, "Numeric metadata should be preserved")
        self.assertIn("nested.level1.level2", metadata, "Nested metadata should be flattened")
        self.assertEqual(metadata["nested.level1.level2"], "nested value", "Nested values should be preserved")
        
        # Verify token count is present
        self.assertIn("token_count", metadata, "Token count should be in metadata")
        self.assertGreater(metadata["token_count"], 0, "Token count should be positive")

if __name__ == "__main__":
    unittest.main()
