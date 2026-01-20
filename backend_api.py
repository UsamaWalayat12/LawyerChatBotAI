"""
Pakistani Legal RAG Assistant - REST API
FastAPI backend that wraps the existing RAG functionality
Optimized for Vercel serverless deployment
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import sys
import json

# Import existing RAG functions
try:
    from chroma_test import (
        retrieve_and_filter,
        call_gemini_chat,
        build_strict_rag_prompt,
        load_history,
        save_history,
        add_to_history,
        generate_pdf_from_history,
        extract_case_metadata,
        GEMINI_MODEL,
        TOP_K,
        RETURN_TOP
    )
except ImportError as e:
    print(f"Warning: Could not import chroma_test: {e}")
    # Fallback values for serverless environment
    GEMINI_MODEL = "models/gemini-2.5-flash"
    TOP_K = 10
    RETURN_TOP = 5
    
    # Dummy functions to prevent NameError
    def retrieve_and_filter(*args, **kwargs): return []
    def call_gemini_chat(prompt, *args, **kwargs): return "RAG is unavailable (Serverless Mode). Please configure a cloud database."
    def build_strict_rag_prompt(*args, **kwargs): return ""
    def load_history(): return []
    def save_history(*args, **kwargs): pass
    def add_to_history(*args, **kwargs): pass
    def generate_pdf_from_history(*args, **kwargs): return None
    def extract_case_metadata(*args, **kwargs): return {}

# Create FastAPI app
app = FastAPI(
    title="Pakistani Legal RAG Assistant API",
    description="REST API for Pakistani legal document analysis and generation",
    version="1.0.0"
)

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Flutter app can connect)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    success: bool
    answer: str
    timestamp: str
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None

class HistoryResponse(BaseModel):
    success: bool
    history: List[Dict[str, Any]]
    total_messages: int

class GeneratePDFRequest(BaseModel):
    mode: str = "template"  # "template" or "full"
    doc_type: Optional[str] = None

class GeneratePDFResponse(BaseModel):
    success: bool
    pdf_url: str
    filename: str

class StatusResponse(BaseModel):
    status: str
    version: str
    model: str
    endpoints: List[str]

# Root endpoint
@app.get("/")
async def root():
    """API root - health check"""
    return {
        "message": "Pakistani Legal RAG Assistant API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Health check
@app.get("/health")
async def health_check():
    """Check if API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Status endpoint
@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get API status and configuration"""
    return {
        "status": "running",
        "version": "1.0.0",
        "model": GEMINI_MODEL,
        "endpoints": [
            "/api/chat",
            "/api/history",
            "/api/generate-pdf",
            "/api/pdf/{filename}",
            "/api/clear-history"
        ]
    }

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a legal query and get AI-generated answer
    
    Args:
        request: ChatRequest with query string
        
    Returns:
        ChatResponse with AI answer
    """
    try:
        query = request.query.strip()
        user_id = request.user_id or "default"
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Load history for this user
        history = load_history(user_id)
        
        # Add user query to history
        add_to_history(history, "user", query, user_id)
        
        # Retrieve evidence
        print(f"[API] Processing query: {query[:100]}...")
        evidences = retrieve_and_filter(query, top_k=TOP_K, return_top=RETURN_TOP)
        print(f"[API] Retrieved {len(evidences)} evidence documents")
        
        if len(evidences) == 0:
            # Log the issue for debugging
            print(f"[API] WARNING: No evidence found for query: {query}")
            
            # Provide a more helpful error message
            answer = (
                "I apologize, but I couldn't find specific legal documents matching your query in the database. "
                "This could mean:\n\n"
                "1. The query might need to be rephrased\n"
                "2. The specific legal topic might not be in the current database\n"
                "3. Try using different keywords or asking about general Pakistani legal topics\n\n"
                "You can try asking about:\n"
                "- Contract law and breach of contract\n"
                "- Property and land transfer procedures\n"
                "- Employment law and labor rights\n"
                "- Legal notices and documentation\n"
                "- Court procedures and judgments"
            )
        else:
            # Build prompt
            prompt = build_strict_rag_prompt(query, evidences)
            
            # Get AI response
            print(f"[API] Calling Gemini AI...")
            answer = call_gemini_chat(prompt, model=GEMINI_MODEL, temperature=0.0, max_tokens=2000)
            print(f"[API] Received AI response ({len(answer)} chars)")
        
        # Add assistant response to history
        add_to_history(history, "assistant", answer, user_id)
        
        # Generate message ID
        message_id = f"msg_{len(history)}"
        
        return {
            "success": True,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "message_id": message_id,
            "conversation_id": request.conversation_id
        }
        
    except Exception as e:
        print(f"[API] Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Get history endpoint
@app.get("/api/history", response_model=HistoryResponse)
async def get_history(user_id: Optional[str] = "default"):
    """
    Get complete chat history for a specific user
    
    Args:
        user_id: User ID to filter history (query parameter)
    
    Returns:
        HistoryResponse with all messages for the user
    """
    try:
        history = load_history(user_id)
        
        return {
            "success": True,
            "history": history,
            "total_messages": len(history)
        }
        
    except Exception as e:
        print(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

# Get conversations endpoint
@app.get("/api/conversations")
async def get_conversations(user_id: Optional[str] = "default"):
    """
    Get list of conversations for a specific user
    
    Args:
        user_id: User ID to filter conversations (query parameter)
    
    Returns:
        List of conversations with metadata
    """
    try:
        # For now, we'll return a simple structure
        # In the future, you can enhance this to group messages by conversation_id
        history = load_history(user_id)
        
        # Create a single conversation entry for all messages
        if history:
            from datetime import datetime
            return {
                "success": True,
                "conversations": [{
                    "id": "default",
                    "title": "Legal Chat",
                    "timestamp": datetime.now().isoformat(),
                    "messageCount": len(history),
                    "lastMessageTime": datetime.now().isoformat()
                }]
            }
        else:
            return {
                "success": True,
                "conversations": []
            }
        
    except Exception as e:
        print(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversations: {str(e)}")

# Clear history endpoint
@app.delete("/api/history/clear")
async def clear_history():
    """Clear all chat history"""
    try:
        history_file = "chat_history.json"
        if os.path.exists(history_file):
            os.remove(history_file)
        
        return {
            "success": True,
            "message": "Chat history cleared successfully"
        }
        
    except Exception as e:
        print(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

# Generate PDF endpoint
@app.post("/api/generate-pdf", response_model=GeneratePDFResponse)
async def generate_pdf(request: GeneratePDFRequest):
    """
    Generate PDF document from chat history
    
    Args:
        request: GeneratePDFRequest with mode and doc_type
        
    Returns:
        GeneratePDFResponse with PDF URL and filename
    """
    try:
        history = load_history()
        
        if not history:
            raise HTTPException(status_code=400, detail="No chat history found. Have a conversation first.")
        
        # Generate PDF
        pdf_filename = generate_pdf_from_history(
            history, 
            doc_type=request.doc_type,
            mode=request.mode
        )
        
        if not pdf_filename:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")
        
        # Return URL to download PDF
        pdf_url = f"/api/pdf/{pdf_filename}"
        
        return {
            "success": True,
            "pdf_url": pdf_url,
            "filename": pdf_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

# Download PDF endpoint
@app.get("/api/pdf/{filename}")
async def download_pdf(filename: str):
    """
    Download generated PDF file
    
    Args:
        filename: Name of the PDF file
        
    Returns:
        FileResponse with PDF file
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Return file
        return FileResponse(
            filename,
            media_type="application/pdf",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading PDF: {str(e)}")

# List PDFs endpoint
@app.get("/api/pdfs")
async def list_pdfs():
    """List all generated PDF files"""
    try:
        pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
        
        return {
            "success": True,
            "pdfs": pdf_files,
            "count": len(pdf_files)
        }
        
    except Exception as e:
        print(f"Error listing PDFs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")

# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "success": True,
        "config": {
            "model": GEMINI_MODEL,
            "top_k": TOP_K,
            "return_top": RETURN_TOP,
            "db_path": "ChromaDB",
            "collection": "pakistan_law"
        }
    }

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "An unexpected error occurred"
        }
    )

# Run server
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment (Railway sets this automatically)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("=" * 60)
    print("🚀 Pakistani Legal RAG Assistant API")
    print("=" * 60)
    print(f"📡 Server starting at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🤖 AI Model: {GEMINI_MODEL}")
    print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print("=" * 60)
    print("\n✅ Server is ready! You can now connect your Flutter app.\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
