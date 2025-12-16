from dotenv import load_dotenv
load_dotenv()

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, StateGraph, START
from langchain_qdrant import FastEmbedSparse
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

from database import QdrantVT
from router import QueryRouter
from tools import ToolSearch
from database.chat_history import ChatHistoryManager

from typing_extensions import TypedDict
from typing import List

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Request, Form

from pprint import pprint
from database.create_docs import extract_text_from_pdf, LawDocumentCreator
import os
import json
import asyncio
import traceback
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Khởi tạo các CONSTANT ###
HUGGINGFACE_MODEL = 'keepitreal/vietnamese-sbert'
FAST_EMBED_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
LLM_MODEL = "sonar-pro"
COLLECTION_NAME = "law_collection"
VN_LAW_PDF_PATH = "VanBanGoc_52.2014.QH13.pdf"


### Khởi tạo ChatHistoryManager ###
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "hoangducdung"),
    "database": os.getenv("DB_NAME", "law_db")
}

chat_history_manager = ChatHistoryManager(**DB_CONFIG)

### Khởi tạo các đối tượng ###
embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)
sparse = FastEmbedSparse(model_name=FAST_EMBED_SPARSE_MODEL, batch_size=4)
collection = COLLECTION_NAME

qdrant_vectors = QdrantVT(collectionName=collection, embeddingModel=embeddings, SparseModel=sparse)

from langchain_openai import ChatOpenAI

llm_nostream = ChatOpenAI(
    api_key="",
    base_url="https://api.perplexity.ai",
    model_name=LLM_MODEL,
    streaming=False,
    temperature=0,
)

llm_stream = ChatOpenAI(
    api_key="",
    base_url="https://api.perplexity.ai",
    model_name=LLM_MODEL,
    streaming=True,
    temperature=0,
)


# QueryRouter phải nhận model non-stream
query_router = QueryRouter(llm=llm_nostream)

tool = ToolSearch()
wiki_tool = tool.wiki
documents = qdrant_vectors.load_documents(pdf_path=VN_LAW_PDF_PATH)
try:
    qdrant_vectors.client.get_collection(collection) 
    retriever = qdrant_vectors.load_vectorstore()
except:
    retriever = qdrant_vectors.init_vectorstore()
    
compressor =  CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY", ""), model="rerank-multilingual-v3.0")
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

### FUNCTIONS ###
class GraphState(TypedDict):
    question: str
    session_id: str
    instruction: str
    generation: str
    documents: List[str]
    answer: str

def retrieve(state):
    print("---RETRIEVE---")
    instruction = state["instruction"]
    question = state["question"]    
    documents = rerank_retriever.invoke(instruction)
    
    text = ""
    for i, result in enumerate(documents):
        try:
            chapter_number = result.metadata.get('chapter_number', 'N/A')
            chapter_title = result.metadata.get('chapter_title', 'N/A')
            article_number = result.metadata.get('article_number', 'N/A')
            
            if chapter_number != 'N/A' and chapter_title != 'N/A' and article_number != 'N/A':
                text += f"{i+1}. {chapter_number} - {chapter_title} - {article_number}\n {result.page_content}\n\n"
            else:
                text += f"{i+1}. {article_number if article_number != 'N/A' else 'Điều không xác định'}\n {result.page_content}\n\n"
                
        except Exception as e:
            text += f"{i+1}. Nội dung pháp lý\n {result.page_content}\n\n"
    
    return {"documents": text, "question": question}

def wiki_search(state):
    print("---WIKI_SEARCH---")
    question = state["question"]
    wiki_results = wiki_tool.invoke({"query": question})
    wiki_results = Document(page_content=wiki_results)
    return {"documents": wiki_results, "question": question}

def route_question(state):
    print("---ROUTE_QUESTION---")
    question = state["question"]
    source = query_router.route_question(question)

    if source.datasource == "wiki_search":
        return "wiki_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"

def preprocess_query(state):
    session_id = state.get('session_id')
    conversation_context = ""

    if session_id:
        try:
            context_messages = chat_history_manager.get_conversation_context(session_id, 5)
            if context_messages:
                conversation_context = "\n--- NGỮ CẢNH CUỘC TRÒ CHUYỆN TRƯỚC ĐÓ ---\n"
                for msg in context_messages[:-1]:
                    role = "👤 Người dùng" if msg['role'] == 'user' else "🤖 Trợ lý"
                    conversation_context += f"{role}: {msg['content'][:200]}\n"
                conversation_context += "--- KẾT THÚC NGỮ CẢNH ---\n\n"
        except Exception:
            conversation_context = ""

    prompt = f"""
    Bạn là chuyên gia Luật Hôn nhân và Gia đình Việt Nam 2014.
    {conversation_context}
    Câu hỏi: {state['question']}
    """
    response = llm_stream.invoke(prompt)
    return {"instruction": response.content}

def chatbot(state):
    # Placeholder for simple chatbot flow if needed, usually we use stream
    return {"answer": "Chatbot response"}

workflow = StateGraph(GraphState)
workflow.add_node("preprocess_query", preprocess_query)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("chatbot", chatbot)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "preprocess_query",
    },
)
workflow.add_edge( "preprocess_query", "retrieve")
workflow.add_edge( "retrieve", "chatbot")
workflow.add_edge( "wiki_search", "chatbot")
workflow.add_edge( "chatbot", END)

graph = workflow.compile()

# MODELS
class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None

class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class QuestionInput(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    question: str

class SessionUpdate(BaseModel):
    title: str

class MessageSearch(BaseModel):
    user_id: str
    query: str

async def gen_streaming_response(inputs: QuestionInput):
    async def send(evt: dict):
        try:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception:
            safe_evt = {k: str(v) for k, v in evt.items()}
            yield f"data: {json.dumps(safe_evt, ensure_ascii=False)}\n\n"

    try:
        session_id = inputs.session_id
        if not session_id:
            session_id = chat_history_manager.create_session(inputs.user_id, "New Chat")
            if not session_id:
                async for chunk in send({"type": "error", "error": "Không thể tạo session"}): yield chunk
                return

        # [FIX] Lưu câu hỏi người dùng vào DB ngay lập tức
        chat_history_manager.add_message(session_id, "user", inputs.question)
        
        # [FIX] Cập nhật title nếu là tin nhắn đầu tiên của session
        try:
            msgs = chat_history_manager.get_session_messages(session_id, 5)
            # Nếu chỉ có 1 tin (vừa lưu), hoặc title vẫn là mặc định
            if len(msgs) <= 1:
                new_title = inputs.question[:50] + "..." if len(inputs.question) > 50 else inputs.question
                chat_history_manager.update_session_title(session_id, new_title)
        except Exception:
            pass

        async for chunk in send({"type": "session_id", "session_id": session_id}): yield chunk

        question_text = inputs.question
        try:
            source = query_router.route_question(question_text)
            datasource = getattr(source, "datasource", "vectorstore")
            async for chunk in send({"type": "route", "source": datasource}): yield chunk
        except Exception:
            datasource = "vectorstore"

        conversation_context = ""
        try:
            context_messages = chat_history_manager.get_conversation_context(session_id, 5)
            if context_messages:
                conversation_context = "\n--- NGỮ CẢNH ---\n"
                for msg in context_messages[:-1]:
                    role = "User" if msg.get('role') == 'user' else "Bot"
                    conversation_context += f"{role}: {msg.get('content', '')[:200]}\n"
        except Exception:
            pass

        documents_text = ""
        if datasource == "wiki_search":
            async for chunk in send({"type": "status", "message": "Đang tìm kiếm trên Wikipedia..."}): yield chunk
            try:
                wiki_result = wiki_tool.invoke({"query": question_text})
                documents_text = Document(page_content=wiki_result).page_content or ""
            except Exception as e:
                async for chunk in send({"type": "error", "error": f"Wiki lỗi: {str(e)}"}): yield chunk
        else:
            async for chunk in send({"type": "status", "message": "Đang tìm kiếm trong CSDL..."}): yield chunk
            
            # Preprocess / Instruction generation
            preprocess_prompt = f"Tạo câu truy vấn tìm kiếm cho: {question_text}\nNgữ cảnh: {conversation_context}"
            instruction = question_text # Fallback default
            try:
                instruction = llm_stream.invoke(preprocess_prompt).content
            except:
                pass

            try:
                documents = rerank_retriever.invoke(instruction)
                cnt = len(documents) if hasattr(documents, "__len__") else 0
                async for chunk in send({"type": "documents_retrieved", "count": cnt}): yield chunk

                buf = []
                for i, result in enumerate(documents or []):
                    md = getattr(result, "metadata", {}) or {}
                    buf.append(f"{i+1}. {md.get('article_number','')} {result.page_content}\n")
                documents_text = "\n".join(buf)
            except Exception as e:
                async for chunk in send({"type": "error", "error": f"Retrieve lỗi: {str(e)}"}): yield chunk

        async for chunk in send({"type": "generating", "message": "Đang trả lời..."}): yield chunk

        final_prompt = f"""
        Trợ lý luật sư chuyên nghiệp. Dựa vào tài liệu sau trả lời câu hỏi.
        Tài liệu: {documents_text}
        Context: {conversation_context}
        Câu hỏi: {question_text}
        """

        full_answer = ""
        try:
            for chunk in llm_stream.stream(final_prompt):
                text = getattr(chunk, "content", "") or ""
                if text:
                    full_answer += text
                    yield f"data: {json.dumps({'type':'content','content': text}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.02)
        except Exception as e:
            async for chunk in send({"type": "error", "error": f"LLM lỗi: {str(e)}"}): yield chunk

        # Save history
        try:
            if full_answer.strip():
                chat_history_manager.add_message(session_id, "assistant", full_answer)
            
            # Update title if needed (simple logic)
            msgs = chat_history_manager.get_session_messages(session_id, 5)
            if len(msgs) <= 2:
                chat_history_manager.update_session_title(session_id, question_text[:50])
                
        except Exception:
            pass

        async for chunk in send({"type": "completed"}): yield chunk

    except Exception as e:
        async for chunk in send({"type": "error", "error": str(e)}): yield chunk

@app.post("/api/chat/stream")
async def chat_stream(inputs: QuestionInput):
    return StreamingResponse(gen_streaming_response(inputs), media_type="text/event-stream")

@app.post("/api/users")
async def create_user(user: UserCreate):
    user_id = chat_history_manager.create_user(user.username, user.email)
    if user_id:
        return {"status": "success", "user_id": user_id}
    raise HTTPException(status_code=400, detail="User creation failed")

@app.get("/api/users/{username}")
async def get_user_by_username(username: str):
    user = chat_history_manager.get_user_by_username(username)
    if user:
        return {"status": "success", "user": user}
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/api/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 50):
    sessions = chat_history_manager.get_user_sessions(user_id, limit)
    return {"status": "success", "sessions": sessions}

@app.post("/api/sessions")
async def create_session(session: SessionCreate):
    session_id = chat_history_manager.create_session(session.user_id, session.title)
    if session_id:
        return {"status": "success", "session_id": session_id}
    raise HTTPException(status_code=400, detail="Session creation failed")

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 100):
    messages = chat_history_manager.get_session_messages(session_id, limit)
    return {"status": "success", "messages": messages}

@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, session_update: SessionUpdate):
    success = chat_history_manager.update_session_title(session_id, session_update.title)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Update failed")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    success = chat_history_manager.delete_session(session_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Delete failed")

@app.get("/api/admin/stats")
async def get_admin_stats():
    stats = chat_history_manager.get_system_stats()
    return {"status": "success", "stats": stats}

@app.get("/api/admin/users")
async def get_admin_users(limit: int = 20):
    users = chat_history_manager.get_all_users_activity(limit)
    return {"status": "success", "users": users}

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    raw_text = extract_text_from_pdf(file_location)
    
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "hoangducdung",
        "database": "law_db"
    }
    doc_creator = LawDocumentCreator(**db_config)
    doc_creator.add_chapters_and_articles_to_db(raw_text)
    documents = doc_creator.create_documents_from_db()
    qdrant_vectors.upsert_documents(documents)
    
    return {"status": "success", "filename": file.filename}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    success = chat_history_manager.delete_user(user_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Delete failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)