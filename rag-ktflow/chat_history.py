"""
Chat history management for RAG system
"""
import time
import logging
import json
import uuid
import os
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatMessage:
    """Represents a single message in a chat history"""
    
    def __init__(
        self,
        role: str,
        content: str,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a chat message
        
        Args:
            role: Role of the message sender (user/assistant)
            content: Message content
            message_id: Unique ID for the message
            timestamp: Message timestamp
            metadata: Additional metadata for the message
        """
        self.role = role
        self.content = content
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "role": self.role,
            "content": self.content,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary representation"""
        return cls(
            role=data["role"],
            content=data["content"],
            message_id=data["message_id"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


class Conversation:
    """Represents a conversation with multiple messages"""
    
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a conversation
        
        Args:
            conversation_id: Unique ID for the conversation
            title: Title of the conversation
            messages: List of chat messages
            metadata: Additional metadata for the conversation
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.title = title or f"Conversation {self.conversation_id[:8]}"
        self.messages = messages or []
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def add_message(self, message: Union[ChatMessage, Dict[str, Any]]) -> ChatMessage:
        """
        Add a message to the conversation
        
        Args:
            message: Chat message or dictionary
            
        Returns:
            The added message
        """
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        
        self.messages.append(message)
        self.updated_at = time.time()
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary representation"""
        messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        conversation = cls(
            conversation_id=data["conversation_id"],
            title=data.get("title"),
            messages=messages,
            metadata=data.get("metadata", {})
        )
        conversation.created_at = data.get("created_at", time.time())
        conversation.updated_at = data.get("updated_at", time.time())
        return conversation
    
    def get_context_window(self, max_messages: int = 10) -> List[ChatMessage]:
        """
        Get the most recent messages up to max_messages
        
        Args:
            max_messages: Maximum number of messages to include
            
        Returns:
            List of most recent messages
        """
        return self.messages[-max_messages:] if max_messages > 0 else self.messages
    
    def format_for_prompt(self, max_messages: int = 10) -> str:
        """
        Format recent messages for inclusion in an LLM prompt
        
        Args:
            max_messages: Maximum number of messages to include
            
        Returns:
            Formatted conversation history
        """
        recent_messages = self.get_context_window(max_messages)
        formatted = []
        
        for msg in recent_messages:
            role_prefix = "User: " if msg.role.lower() == "user" else "Assistant: "
            formatted.append(f"{role_prefix}{msg.content}")
        
        return "\n\n".join(formatted)


class ChatHistoryManager:
    """Manages conversation histories"""
    
    def __init__(self, storage_dir: str = "chat_history"):
        """
        Initialize chat history manager
        
        Args:
            storage_dir: Directory to store chat histories
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.active_conversations: Dict[str, Conversation] = {}
        logger.info(f"Initialized chat history manager with storage at {os.path.abspath(storage_dir)}")
    
    def create_conversation(
        self, 
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation
        
        Args:
            title: Conversation title
            metadata: Additional metadata
            
        Returns:
            New conversation
        """
        conversation = Conversation(
            title=title,
            metadata=metadata
        )
        self.active_conversations[conversation.conversation_id] = conversation
        self._save_conversation(conversation)
        logger.info(f"Created new conversation: {conversation.conversation_id}")
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation if found, None otherwise
        """
        # Check in-memory cache first
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Try loading from storage
        try:
            conversation = self._load_conversation(conversation_id)
            if conversation:
                self.active_conversations[conversation_id] = conversation
                return conversation
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
        
        logger.warning(f"Conversation {conversation_id} not found")
        return None
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChatMessage]:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Added message if successful, None otherwise
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            logger.warning(f"Cannot add message, conversation {conversation_id} not found")
            return None
        
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata
        )
        conversation.add_message(message)
        self._save_conversation(conversation)
        logger.info(f"Added message to conversation {conversation_id}")
        return message
    
    def list_conversations(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """
        List available conversations
        
        Args:
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip
            
        Returns:
            List of conversation summaries
        """
        conversations = []
        
        try:
            files = os.listdir(self.storage_dir)
            json_files = [f for f in files if f.endswith(".json")]
            
            # Sort by modification time (most recent first)
            sorted_files = sorted(
                json_files,
                key=lambda f: os.path.getmtime(os.path.join(self.storage_dir, f)),
                reverse=True
            )
            
            # Apply pagination
            paginated_files = sorted_files[skip:skip+limit]
            
            for filename in paginated_files:
                try:
                    file_path = os.path.join(self.storage_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                    # Create a summary
                    summary = {
                        "conversation_id": data["conversation_id"],
                        "title": data.get("title", "Untitled"),
                        "message_count": len(data.get("messages", [])),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at")
                    }
                    conversations.append(summary)
                except Exception as e:
                    logger.error(f"Error processing conversation file {filename}: {e}")
            
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
        
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            Success status
        """
        try:
            # Remove from in-memory cache
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
            
            # Remove from storage
            file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            else:
                logger.warning(f"Conversation file for {conversation_id} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    def _save_conversation(self, conversation: Conversation) -> bool:
        """
        Save conversation to storage
        
        Args:
            conversation: Conversation to save
            
        Returns:
            Success status
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{conversation.conversation_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.conversation_id}: {e}")
            return False
    
    def _load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load conversation from storage
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Loaded conversation if successful, None otherwise
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            if not os.path.exists(file_path):
                logger.warning(f"Conversation file for {conversation_id} not found")
                return None
                
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
            return None
