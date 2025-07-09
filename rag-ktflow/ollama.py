"""
Ollama client for interacting with local LLMs
"""
import os
import json
import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, AsyncIterator, Any, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(
        self, 
        model_name: str = "mistral", 
        api_base: str = "http://localhost:11434",
        timeout: float = 120.0,
        gpu_memory_limit: str = None
    ):
        """
        Initialize Ollama client
        
        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for Ollama API
            timeout: Request timeout in seconds
            gpu_memory_limit: GPU memory limit (e.g., "8G", "4G")
        """
        self.model_name = model_name
        self.api_base = api_base
        self.timeout = timeout
        self.gpu_memory_limit = gpu_memory_limit
        
        # Set environment variable for Ollama GPU memory limit if provided
        if self.gpu_memory_limit:
            os.environ["OLLAMA_GPU_LAYERS"] = "35"  # Limit layers offloaded to GPU
            os.environ["OLLAMA_KEEP_ALIVE"] = "10m"  # Reduce model unloading
            logger.info(f"Setting GPU memory optimizations: OLLAMA_GPU_LAYERS=35")
        
        logger.info(f"Initialized Ollama client for model '{model_name}' at {api_base}")
        
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama is available
        
        Returns:
            Dictionary with status information
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_base}/api/tags")
                if response.status_code == 200:
                    return {
                        "status": "ok",
                        "message": "Ollama is available",
                        "models": [model["name"] for model in response.json().get("models", [])]
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Ollama returned unexpected status: {response.status_code}"
                    }
        except Exception as e:
            logger.error(f"Error checking Ollama health: {e}")
            return {
                "status": "error", 
                "message": f"Error connecting to Ollama: {str(e)}"
            }
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout))
    )
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text using Ollama API
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # First try with GPU with extremely optimized settings
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,  # ensure single JSON response for easy parsing
                    "options": {
                        "num_ctx": 2048,        # Larger context window to fit 12 chunks
                        "num_batch": 16,
                        "num_gpu": 1,
                        "seed": 42,
                        "num_thread": 2,
                        "repeat_last_n": 64
                    }
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                    
                logger.debug(f"Sending request to Ollama API with GPU: {payload}")
                response = await client.post(
                    f"{self.api_base}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                raw_text = response.text
                try:
                    result = json.loads(raw_text)
                except json.JSONDecodeError:
                    # Fallback: treat as NDJSON and take the last JSON object
                    lines = [line for line in raw_text.splitlines() if line.strip()]
                    try:
                        result = json.loads(lines[-1])
                    except Exception as e:
                        logger.error(f"Failed to parse Ollama NDJSON response: {e}")
                        raise
                return result.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"HTTP error from Ollama API with GPU: {e}")
            resp = getattr(e, "response", None)
            if resp is not None:
                logger.error(f"Response status code: {resp.status_code}")
                try:
                    error_content = resp.text
                    logger.error(f"Error response content: {error_content}")
                    # Check for out-of-VRAM clue
                    if "unable to allocate CUDA" in error_content or "CUDA" in error_content:
                        logger.info("GPU memory allocation failed, falling back to CPU mode")
                        return await self.generate_cpu_fallback(prompt, system_prompt, temperature, max_tokens)
                except Exception as parse_err:
                    logger.debug(f"Failed parsing error body: {parse_err}")
            else:
                # Timeout / connection errors have no response – treat as temporary and try CPU fallback
                if isinstance(e, httpx.ReadTimeout):
                    logger.warning("Ollama GPU request timed out – falling back to CPU mode")
                    return await self.generate_cpu_fallback(prompt, system_prompt, temperature, max_tokens)
            # As a final fallback try CLI
            logger.info("Falling back to Ollama CLI")
            return await self.generate_cli_fallback(prompt, system_prompt, temperature)
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            raise
            
    async def generate_cpu_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate text using Ollama API in CPU-only mode
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "num_batch": 16,
                        "num_gpu": 1,
                        "seed": 42,
                        "num_thread": 2,
                        "repeat_last_n": 64
                    }
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                    
                logger.debug(f"Sending request to Ollama API with CPU-only: {payload}")
                response = await client.post(
                    f"{self.api_base}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                raw_text = response.text
                try:
                    result = json.loads(raw_text)
                except json.JSONDecodeError:
                    lines = [l for l in raw_text.splitlines() if l.strip()]
                    result = json.loads(lines[-1])
                return result.get("response", "")
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error from Ollama API in CPU mode: {e}")
            if e.response:
                logger.error(f"Response status code: {e.response.status_code}")
                try:
                    error_content = e.response.text
                    logger.error(f"Error response content: {error_content}")
                except:
                    pass
                    
            # If both GPU and CPU API calls fail, try CLI
            logger.info("Falling back to Ollama CLI")
            return await self.generate_cli_fallback(prompt, system_prompt, temperature)
        except Exception as e:
            logger.error(f"Error generating text with Ollama CPU mode: {e}")
            raise
            
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        """
        Generate streaming text from the Ollama model
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Yields:
            Stream of generated text chunks
        """
        # First try GPU mode with extremely optimized settings
        client = None
        try:
            client = httpx.AsyncClient(timeout=self.timeout)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "options": {
                    "num_ctx": 2048,        # Larger context window to fit 12 chunks
                    "num_batch": 16,
                    "num_gpu": 1,
                    "seed": 42,
                    "num_thread": 2,
                    "repeat_last_n": 64
                }
            }
                
            if system_prompt:
                payload["system"] = system_prompt
                
            logger.debug(f"Starting streaming generation with Ollama")
            async with client.stream(
                "POST",
                f"{self.api_base}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    try:
                        if not chunk.strip():
                            continue
                            
                        json_chunk = json.loads(chunk)
                        token = json_chunk.get("response", "")
                        yield token
                        
                        if json_chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON chunk: {chunk}")
                        continue
                        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama API streaming: {e}")
            if e.response:
                logger.error(f"Response status code: {e.response.status_code}")
                try:
                    error_content = e.response.text
                    logger.error(f"Error response content: {error_content}")
                except:
                    pass
            logger.info("Falling back to non-streaming generation")
            response = await self.generate(prompt, system_prompt, temperature, max_tokens)
            yield response
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield f"Error generating response: {str(e)}"
        finally:
            if client:
                await client.aclose()
            
    async def generate_cli_fallback(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Ollama CLI as a fallback
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            temperature: Temperature parameter
            
        Returns:
            Generated text
        """
        cmd = ["ollama", "run", self.model_name, "--temp", str(temperature)]
        
        input_text = prompt
        if system_prompt:
            # Format depends on the model - this works for most Ollama models
            input_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        
        try:
            logger.info("Using Ollama CLI fallback")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=input_text.encode("utf-8"))
            
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                logger.error(f"Error calling Ollama CLI: {error_msg}")
                return f"Error: {error_msg}"
                
            return stdout.decode("utf-8", errors="replace").strip()
        except Exception as e:
            logger.error(f"Error using Ollama CLI fallback: {e}")
            return f"Failed to generate response: {str(e)}"
            
    def generate_cli(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Ollama CLI (synchronous fallback method)
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        cmd = ["ollama", "run", self.model_name]
        
        input_text = prompt
        if system_prompt:
            # Format depends on the model - this works for most Ollama models
            input_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        
        try:
            logger.info("Using Ollama CLI fallback")
            result = subprocess.run(
                cmd,
                input=input_text.encode("utf-8"),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error calling Ollama CLI: {e}")
            return f"Error: {e.stderr.strip()}"
            
    async def check_health(self) -> bool:
        """
        Check if Ollama service is available
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.api_base}/api/tags")
                if response.status_code == 200:
                    # Check if our model is available
                    try:
                        tags = response.json().get("models", [])
                        model_available = any(tag.get("name") == self.model_name for tag in tags)
                        
                        if not model_available:
                            logger.warning(f"Model {self.model_name} not found in available models")
                    except (json.JSONDecodeError, KeyError):
                        # If we can't check the model list, just return that the server is up
                        pass
                        
                    return True
                else:
                    logger.warning(f"Ollama health check failed with status {response.status_code}")
                    return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
            
    def format_rag_prompt(
        self, 
        query: str, 
        context_chunks: List[str],
        max_context_length: int = 4000
    ) -> Dict[str, str]:
        """
        Format prompt for RAG with context
        
        Args:
            query: User query
            context_chunks: List of context chunks
            max_context_length: Maximum context length to include (approximate)
            
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        # Create system prompt for RAG
        system_prompt = (
            "You are a knowledgeable assistant answering questions based on the provided context. "
            "Your answers should be based exclusively on the context information provided. "
            "If the context doesn't contain enough information to answer fully, acknowledge the limitations "
            "and explain what you can based on the context. "
            "Do not make up information or use knowledge outside the provided context. "
            "Provide specific information from the context rather than generic responses. "
            "If asked about topics not in the context, politely state that you don't have information on that topic."
        )
        
        # Format context chunks with numbers and source metadata if available
        formatted_chunks = []
        total_length = 0
        
        for i, chunk in enumerate(context_chunks):
            # Add chunk with header
            chunk_text = f"[Context {i+1}]\n{chunk}"
            
            # Check if adding this chunk would exceed the max context length
            # If so, break - we'll use the chunks we've collected so far
            if total_length + len(chunk_text) > max_context_length and formatted_chunks:
                logger.warning(f"Truncated context to fit {max_context_length} chars, using {len(formatted_chunks)} of {len(context_chunks)} chunks")
                break
                
            formatted_chunks.append(chunk_text)
            total_length += len(chunk_text)
        
        # Format user prompt with query and context
        context_text = "\n\n".join(formatted_chunks)
        user_prompt = f"I need information based on the following context:\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        logger.debug(f"Created RAG prompt with {len(formatted_chunks)} context chunks, total length: {total_length}")
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
