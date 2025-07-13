# Multi-Agentic Conversational AI System with Gemini AI
# Main application file

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from datetime import datetime
import uuid
import json
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database and AI imports
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import asyncio

# Gemini AI imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Document processing imports
import PyPDF2
import pandas as pd
from io import BytesIO, StringIO
import csv


# Configuration
class Settings:
    mongodb_url: str = os.getenv("MONGODB_URL")
    database_name: str = os.getenv("DATABASE_NAME")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    gemini_model: str = "gemini-1.5-flash"
    max_context_length: int = 8000
    debug_mode: bool = True


settings = Settings()

# Check Gemini API key
if not settings.gemini_api_key:
    print("❌ Gemini API key is not set!")
    print("Please:")
    print("1. Get an API key from https://makersuite.google.com/app/apikey")
    print("2. Set it as environment variable: export GEMINI_API_KEY='your_key'")
    print("3. Or create a .env file with: GEMINI_API_KEY=your_key")
else:
    print(f"✅ Gemini API key loaded: {settings.gemini_api_key[:10]}...")
    # Configure Gemini
    genai.configure(api_key=settings.gemini_api_key)

# Database setup
client = AsyncIOMotorClient(settings.mongodb_url)
db = client[settings.database_name]

# Collections
users_collection = db.users
conversations_collection = db.conversations
documents_collection = db.documents
embeddings_collection = db.embeddings

# Initialize embedding model
embedding_model = SentenceTransformer(settings.embedding_model)


# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: str
    company: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = {}


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    timestamp: datetime
    processing_time: float
    rag_sources: List[str]
    conversation_category: Optional[str] = None


class ConversationEntry(BaseModel):
    user_id: str
    session_id: str
    user_message: str
    bot_response: str
    timestamp: datetime
    category: Optional[str] = None
    status: str = "active"


# RAG System
class RAGSystem:
    def __init__(self):
        self.embedding_model = embedding_model

    async def add_document(self, content: str, filename: str, doc_type: str):
        """Add document to knowledge base with embeddings"""
        chunks = self.split_text(content)
        document_id = str(uuid.uuid4())

        # Store document
        doc_entry = {
            "_id": document_id,
            "filename": filename,
            "type": doc_type,
            "content": content,
            "chunks": chunks,
            "created_at": datetime.utcnow()
        }

        await documents_collection.insert_one(doc_entry)

        # Generate and store embeddings
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk).tolist()

            embedding_entry = {
                "document_id": document_id,
                "chunk_index": i,
                "chunk_text": chunk,
                "embedding": embedding,
                "created_at": datetime.utcnow()
            }

            await embeddings_collection.insert_one(embedding_entry)

        return document_id

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    async def retrieve_relevant_docs(self, query: str, top_k: int = 3):
        """Retrieve relevant documents using semantic search"""
        query_embedding = self.embedding_model.encode(query).tolist()

        # Get all embeddings
        embeddings_cursor = embeddings_collection.find({})
        embeddings_list = await embeddings_cursor.to_list(length=None)

        if not embeddings_list:
            return []

        # Calculate similarities
        similarities = []
        for emb_doc in embeddings_list:
            similarity = cosine_similarity(
                [query_embedding],
                [emb_doc['embedding']]
            )[0][0]
            similarities.append((similarity, emb_doc))

        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_docs = similarities[:top_k]

        # Get document contents
        relevant_docs = []
        for similarity, emb_doc in top_docs:
            doc = await documents_collection.find_one({"_id": emb_doc['document_id']})
            if doc:
                relevant_docs.append({
                    "content": emb_doc['chunk_text'],
                    "source": doc['filename'],
                    "similarity": similarity
                })

        return relevant_docs


# CRM System
class CRMSystem:
    async def create_user(self, user_data: UserCreate):
        """Create new user profile"""
        user_id = str(uuid.uuid4())

        user_doc = {
            "_id": user_id,
            "name": user_data.name,
            "email": user_data.email,
            "company": user_data.company,
            "preferences": user_data.preferences,
            "created_at": datetime.utcnow(),
            "last_interaction": datetime.utcnow()
        }

        await users_collection.insert_one(user_doc)
        return user_id

    async def get_user(self, user_id: str):
        """Get user by ID"""
        user = await users_collection.find_one({"_id": user_id})
        return user

    async def update_user(self, user_id: str, update_data: UserUpdate):
        """Update user information"""
        update_dict = {}
        if update_data.name is not None:
            update_dict["name"] = update_data.name
        if update_data.email is not None:
            update_dict["email"] = update_data.email
        if update_data.company is not None:
            update_dict["company"] = update_data.company
        if update_data.preferences is not None:
            update_dict["preferences"] = update_data.preferences

        update_dict["last_interaction"] = datetime.utcnow()

        result = await users_collection.update_one(
            {"_id": user_id},
            {"$set": update_dict}
        )

        return result.modified_count > 0

    async def save_conversation(self, conversation_entry: ConversationEntry):
        """Save conversation to database"""
        conv_doc = {
            "user_id": conversation_entry.user_id,
            "session_id": conversation_entry.session_id,
            "user_message": conversation_entry.user_message,
            "bot_response": conversation_entry.bot_response,
            "timestamp": conversation_entry.timestamp,
            "category": conversation_entry.category,
            "status": conversation_entry.status
        }

        await conversations_collection.insert_one(conv_doc)

    async def get_conversation_history(self, user_id: str, limit: int = 20):
        """Get conversation history for user"""
        conversations = await conversations_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit).to_list(length=None)

        return conversations

    async def categorize_conversation(self, message: str, response: str):
        """Simple categorization logic"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['help', 'support', 'problem', 'issue']):
            return 'support'
        elif any(word in message_lower for word in ['buy', 'purchase', 'price', 'cost']):
            return 'sales'
        elif any(word in message_lower for word in ['information', 'about', 'what', 'how']):
            return 'information'
        else:
            return 'general'


# Gemini AI Service
class GeminiService:
    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model

        # Initialize Gemini model with safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        # Generation configuration
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1000,
        }

        # Fallback responses
        self.fallback_responses = {
            'greeting': [
                "Hello! I'm here to help you with any questions or concerns you may have.",
                "Hi there! Welcome to our service. How can I assist you today?",
                "Greetings! I'm ready to help you with whatever you need."
            ],
            'support': [
                "I understand you need assistance. I'm here to help resolve your issue.",
                "Thank you for reaching out for support. Let me help you with that problem.",
                "I'm here to provide the support you need. Can you tell me more about the issue?"
            ],
            'sales': [
                "I'd be happy to help you with information about our products and pricing.",
                "Let me assist you in finding the right solution for your needs.",
                "I can provide details about our services and help you make the best choice."
            ],
            'information': [
                "I can provide you with the information you're looking for.",
                "Let me share what I know about that topic.",
                "I'd be happy to explain more about our services and offerings."
            ],
            'general': [
                "That's an interesting question. Let me help you with that.",
                "I understand what you're asking. Here's what I can tell you.",
                "Thank you for your question. I'll do my best to provide a helpful response."
            ]
        }

    async def generate_response(self, prompt: str, conversation_history: List[str] = None, rag_context: str = None):
        """Generate response using Gemini AI"""

        if not self.api_key:
            print("No Gemini API key available, using fallback response")
            return self._get_fallback_response(prompt)

        try:
            # Initialize the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )

            # Build the complete prompt with context
            system_prompt = """You are a helpful, knowledgeable AI assistant. You provide accurate, relevant, and helpful responses to user questions. 
You are part of a customer service system, so be professional but friendly. If you have relevant information from documents, use it to provide better answers.
Keep your responses concise but comprehensive."""

            # Build context
            context_parts = []

            if rag_context:
                context_parts.append(f"Relevant information from knowledge base:\n{rag_context}")

            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 6 messages
                context_parts.append(f"Recent conversation:\n" + "\n".join(recent_history))

            context_parts.append(f"User question: {prompt}")

            full_prompt = system_prompt + "\n\n" + "\n\n".join(context_parts)

            # Generate response
            response = await asyncio.to_thread(model.generate_content, full_prompt)

            # Extract response text
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = candidate.content.parts[0].text
                    return response_text.strip()

            # If no valid response, use fallback
            return self._get_fallback_response(prompt)

        except Exception as e:
            print(f"Error generating response with Gemini: {str(e)}")
            return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str):
        """Generate fallback response using simple rules"""
        import random

        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']):
            category = 'greeting'
        elif any(word in prompt_lower for word in ['help', 'support', 'problem', 'issue', 'error', 'trouble', 'fix']):
            category = 'support'
        elif any(word in prompt_lower for word in
                 ['buy', 'purchase', 'price', 'cost', 'plan', 'pricing', 'payment', 'subscribe']):
            category = 'sales'
        elif any(word in prompt_lower for word in
                 ['what', 'how', 'information', 'about', 'tell me', 'explain', 'describe']):
            category = 'information'
        else:
            category = 'general'

        base_response = random.choice(self.fallback_responses[category])

        # Add some context based on the prompt
        if 'account' in prompt_lower:
            base_response += " Regarding your account, I can help you with general information and direct you to the right resources."
        elif 'service' in prompt_lower:
            base_response += " I can provide information about our services and how they might benefit you."
        elif 'technical' in prompt_lower or 'tech' in prompt_lower:
            base_response += " For technical matters, I can provide general guidance and help you find the right solution."

        return base_response


# Initialize systems
rag_system = RAGSystem()
crm_system = CRMSystem()
gemini_service = GeminiService()


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Multi-Agentic Conversational AI System with Gemini...")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Multi-Agentic Conversational AI System with Gemini",
    description="A comprehensive conversational AI system with RAG and CRM integration using Google Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint with RAG and CRM integration"""
    start_time = time.time()

    # Generate session ID if not provided
    if not message.session_id:
        message.session_id = str(uuid.uuid4())

    # Generate user ID if not provided
    if not message.user_id:
        message.user_id = str(uuid.uuid4())

    try:
        # Get user conversation history
        conversation_history = await crm_system.get_conversation_history(
            message.user_id, limit=10
        )

        # Build conversation context
        history_context = []
        for conv in reversed(conversation_history):
            history_context.append(f"User: {conv['user_message']}")
            history_context.append(f"Assistant: {conv['bot_response']}")

        # RAG: Retrieve relevant documents
        relevant_docs = await rag_system.retrieve_relevant_docs(message.message)

        # Build RAG context
        rag_context = ""
        rag_sources = []
        if relevant_docs:
            rag_context = "Relevant information from knowledge base:\n"
            for doc in relevant_docs:
                rag_context += f"- From {doc['source']}: {doc['content']}\n"
                rag_sources.append(doc['source'])

        # Generate response using Gemini
        bot_response = await gemini_service.generate_response(
            message.message,
            history_context,
            rag_context
        )

        # Categorize conversation
        category = await crm_system.categorize_conversation(
            message.message,
            bot_response
        )

        # Save conversation to CRM
        conversation_entry = ConversationEntry(
            user_id=message.user_id,
            session_id=message.session_id,
            user_message=message.message,
            bot_response=bot_response,
            timestamp=datetime.utcnow(),
            category=category,
            status="active"
        )

        await crm_system.save_conversation(conversation_entry)

        # Update user last interaction
        await users_collection.update_one(
            {"_id": message.user_id},
            {"$set": {"last_interaction": datetime.utcnow()}},
            upsert=True
        )

        processing_time = time.time() - start_time

        return ChatResponse(
            response=bot_response,
            user_id=message.user_id,
            session_id=message.session_id,
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            rag_sources=rag_sources,
            conversation_category=category
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_docs")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to populate RAG knowledge base"""
    uploaded_files = []

    for file in files:
        try:
            content = await file.read()
            filename = file.filename

            # Process different file types
            if filename.endswith('.pdf'):
                text_content = extract_text_from_pdf(content)
            elif filename.endswith('.txt'):
                text_content = content.decode('utf-8')
            elif filename.endswith('.csv'):
                text_content = process_csv(content)
            elif filename.endswith('.json'):
                text_content = process_json(content)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {filename}"
                )

            # Add to RAG system
            document_id = await rag_system.add_document(
                text_content,
                filename,
                filename.split('.')[-1]
            )

            uploaded_files.append({
                "filename": filename,
                "document_id": document_id,
                "status": "processed"
            })

        except Exception as e:
            uploaded_files.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    return {"uploaded_files": uploaded_files}


@app.post("/crm/create_user")
async def create_user(user_data: UserCreate):
    """Create new user profile"""
    try:
        user_id = await crm_system.create_user(user_data)
        return {"user_id": user_id, "message": "User created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/crm/update_user/{user_id}")
async def update_user(user_id: str, update_data: UserUpdate):
    """Update user information"""
    try:
        success = await crm_system.update_user(user_id, update_data)
        if success:
            return {"message": "User updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crm/conversations/{user_id}")
async def get_conversations(user_id: str, limit: int = 20):
    """Fetch conversation history for user"""
    try:
        conversations = await crm_system.get_conversation_history(user_id, limit)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_conversation(user_id: Optional[str] = None):
    """Reset conversation memory"""
    try:
        if user_id:
            await conversations_collection.update_many(
                {"user_id": user_id},
                {"$set": {"status": "archived"}}
            )
            return {"message": f"Conversation reset for user {user_id}"}
        else:
            await conversations_collection.update_many(
                {},
                {"$set": {"status": "archived"}}
            )
            return {"message": "All conversations reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "ai_provider": "Google Gemini"
    }


# Utility functions
def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def process_csv(content: bytes) -> str:
    """Process CSV file"""
    df = pd.read_csv(BytesIO(content))
    return df.to_string()


def process_json(content: bytes) -> str:
    """Process JSON file"""
    data = json.loads(content.decode('utf-8'))
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)