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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Request, Form

from pprint import pprint
from database.create_docs import extract_text_from_pdf, LawDocumentCreator
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Cho phép origin của giao diện
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Khởi tạo các CONSTANT ###
#HUGGINGFACE_MODEL = "intfloat/multilingual-e5-base"
HUGGINGFACE_MODEL = 'keepitreal/vietnamese-sbert'
FAST_EMBED_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
# LLM_MODEL = "llama-3.1-8b-instant"
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

###  ###
qdrant_vectors = QdrantVT(collectionName=collection, embeddingModel=embeddings, SparseModel=sparse)
# llm = ChatGroq(model_name=LLM_MODEL, api_key="")
llm = ChatOpenAI(
    api_key="",
    base_url="https://api.perplexity.ai",
    model_name = LLM_MODEL
)
query_router = QueryRouter(llm=llm)
tool = ToolSearch()
wiki_tool = tool.wiki
documents = qdrant_vectors.load_documents(pdf_path=VN_LAW_PDF_PATH)
try:
    qdrant_vectors.client.get_collection(collection) 
    retriever = qdrant_vectors.load_vectorstore()

    
except:
    retriever = qdrant_vectors.init_vectorstore()
    
compressor =  CohereRerank(cohere_api_key="", model="rerank-multilingual-v3.0")
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

### FUNCTIONS ###
class GraphState(TypedDict):
    """
    Trạng thái của đồ thị

    Attributes:
        question: câu hỏi
        session_id: ID của session để lấy context
        instruction: hướng dẫn đã được xử lý
        generation: LLM generation
        documents: danh sách các văn bản
        answer: câu trả lời
    """
    question: str
    session_id: str  # Thêm field này
    instruction: str
    generation: str
    documents: List[str]
    answer: str

def retrieve(state):
    print("---RETRIEVE---")
    instruction = state["instruction"]
    question = state["question"]    
    documents = rerank_retriever.invoke(instruction)
    print(f"Number of documents retrieved: {len(documents)}")
    
    text = ""
    
    for i, result in enumerate(documents):
        try:
            print(f"Document {i} metadata keys: {list(result.metadata.keys())}")
            print(f"Document {i} metadata: {result.metadata}")
            
            # Safely get metadata with default values using .get()
            chapter_number = result.metadata.get('chapter_number', 'N/A')
            chapter_title = result.metadata.get('chapter_title', 'N/A')
            article_number = result.metadata.get('article_number', 'N/A')
            
            print(f"Extracted - Chapter: {chapter_number}, Title: {chapter_title}, Article: {article_number}")
            
            # Format the text with safe metadata access
            if chapter_number != 'N/A' and chapter_title != 'N/A' and article_number != 'N/A':
                # Full metadata available
                text += f"{i+1}. {chapter_number} - {chapter_title} - {article_number}\n {result.page_content}\n\n"
            else:
                # Missing some metadata, use a simpler format
                text += f"{i+1}. {article_number if article_number != 'N/A' else 'Điều không xác định'}\n {result.page_content}\n\n"
                
        except Exception as e:
            print(f"Error processing document {i}: {str(e)}")
            print(f"Document type: {type(result)}")
            print(f"Has metadata attr: {hasattr(result, 'metadata')}")
            if hasattr(result, 'metadata'):
                print(f"Metadata type: {type(result.metadata)}")
            # Fallback to simple format
            text += f"{i+1}. Nội dung pháp lý\n {result.page_content}\n\n"
    
    print("---RETRIEVE COMPLETED---")
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
        print("---ROUTE QUESTION TO WIKI SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO VECTORSTORE---")
        return "vectorstore"

def preprocess_query(state):
    # prompt = f"""
    # Bạn là chuyên gia Luật Hôn nhân và Gia đình Việt Nam 2014. Khi nhận được một tình huống hoặc câu hỏi, hãy:
    # 1.Tóm tắt nội dung chính của tình huống.
    # 2.Trích xuất các từ khóa quan trọng.
    # 3.Liệt kê số điều và tên điều trong luật có thể áp dụng.
    # Chỉ trả lời với 3 mục trên, không giải thích thêm.
    # Ví dụ:
    # Tình huống: "Hai vợ chồng đồng ý ly hôn và đã thỏa thuận xong việc chia tài sản."
    # Trả lời:
    # Tóm tắt: Thuận tình ly hôn, đã thỏa thuận chia tài sản.
    # Từ khóa: Thuận tình ly hôn, chia tài sản.
    # Điều luật: Điều 55. Thuận tình ly hôn.

    # Đây là câu hỏi dành cho bạn: {state['question']}
    # """

    session_id = state.get('session_id')
    conversation_context = ""

    if session_id:
        try:
            # Lấy 5 tin nhắn gần nhất để có context
            context_messages = chat_history_manager.get_conversation_context(session_id, 5)
            
            if context_messages:
                conversation_context = "\n--- NGỮ CẢNH CUỘC TRÒ CHUYỆN TRƯỚC ĐÓ ---\n"
                for msg in context_messages[:-1]:  # Loại bỏ tin nhắn hiện tại
                    role = "👤 Người dùng" if msg['role'] == 'user' else "🤖 Trợ lý"
                    conversation_context += f"{role}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}\n"
                conversation_context += "--- KẾT THÚC NGỮ CẢNH ---\n\n"
        except Exception as e:
            print(f"Lỗi khi lấy context: {e}")
            conversation_context = ""

    prompt = f"""
    Bạn là chuyên gia Luật Hôn nhân và Gia đình Việt Nam 2014. Khi nhận được một tình huống hoặc câu hỏi, hãy:

    Nhiệm vụ của bạn:
    1. **Đọc ngữ cảnh cuộc trò chuyện** (nếu có) để hiểu mạch câu chuyện và các vấn đề đã được thảo luận.
    2. **Phân tích câu hỏi hiện tại** trong bối cảnh của cuộc trò chuyện.
    3. **Tóm tắt nội dung chính** của tình huống (bao gồm cả context trước đó nếu liên quan).
    4. **Trích xuất từ khóa quan trọng** từ tình huống và 5 từ khóa tương tự cùng ý nghĩa.
    5. **Liệt kê các điều luật** có thể áp dụng.

    Chỉ trả lời với 3 mục cuối (tóm tắt, từ khóa, điều luật), không giải thích thêm.

    Ví dụ:
    Tình huống: "Hai vợ chồng đồng ý ly hôn và đã thỏa thuận xong việc chia tài sản."
    Trả lời:
    Tóm tắt: Thuận tình ly hôn, đã thỏa thuận chia tài sản.
    Từ khóa: Thuận tình ly hôn, chia tài sản, ly hôn, thỏa thuận, tài sản.
    Điều luật: Điều 55. Thuận tình ly hôn.

    Đây là ngữ cảnh:
    {conversation_context}
    Đây là câu hỏi dành cho bạn: {state['question']}
    """

    
    response = llm.invoke(prompt)
    
    return {
        "instruction": response.content
    }
def chatbot(state):

    session_id = state.get('session_id')
    conversation_context = ""

    if session_id:
        try:
            # Lấy 5 tin nhắn gần nhất để có context
            context_messages = chat_history_manager.get_conversation_context(session_id, 5)
            
            if context_messages:
                conversation_context = "\n--- NGỮ CẢNH CUỘC TRÒ CHUYỆN TRƯỚC ĐÓ ---\n"
                for msg in context_messages[:-1]:  # Loại bỏ tin nhắn hiện tại
                    role = "👤 Người dùng" if msg['role'] == 'user' else "🤖 Trợ lý"
                    conversation_context += f"{role}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}\n"
                conversation_context += "--- KẾT THÚC NGỮ CẢNH ---\n\n"
        except Exception as e:
            print(f"Lỗi khi lấy context: {e}")
            conversation_context = ""

    # Prompt cải tiến cho tác vụ RAG
    prompt = f"""
    Bạn là một trợ lý ảo thông minh, chuyên sâu về Luật hôn nhân và gia đình Việt Nam. 
    Bạn có khả năng truy cập vào các tài liệu pháp lý liên quan để trả lời câu hỏi một cách chính xác và đầy đủ nhất.

    Dưới đây là tài liệu pháp lý mà bạn có thể tham khảo để trả lời câu hỏi. 
    Vui lòng phân tích context, câu hỏi và sử dụng thông tin từ tài liệu để cung cấp câu trả lời rõ ràng và chính xác.

    Tài liệu pháp lý: {state['documents']}

    {conversation_context}

    Câu hỏi: {state['question']}

    Lưu ý: Đảm bảo câu trả lời dựa trên các quy định và điều khoản trong các tài liệu pháp lý, không thêm thông tin ngoài tài liệu.
    """
    
    # Gửi prompt và câu hỏi vào LLM để nhận câu trả lời
    response = llm.invoke(prompt)
    
    return {
        "answer": [response]
    }


workflow = StateGraph(GraphState)
# Định nghĩa các node
workflow.add_node("preprocess_query", preprocess_query)
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("chatbot", chatbot)  # chatbot
# preprocess_query

# Định nghĩa các cạnh
# Xây dựng đồ thị
# workflow.add_edge(START, "preprocess_query")
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

# Compile
graph = workflow.compile()

try:
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_image)
    print("\nĐã lưu đồ thị workflow vào file 'workflow_graph.png'")
except Exception as e:
    print(f"\nKhông thể tạo đồ thị: {str(e)}")


# endpoints
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

@app.post("/api/users")
async def create_user(user: UserCreate):
    """Tạo user mới"""
    try:
        user_id = chat_history_manager.create_user(user.username, user.email)
        if user_id:
            return {"status": "success", "user_id": user_id}
        else:
            raise HTTPException(status_code=400, detail="Không thể tạo user")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{username}")
async def get_user_by_username(username: str):
    """Lấy thông tin user theo username"""
    try:
        user = chat_history_manager.get_user_by_username(username)
        if user:
            return {"status": "success", "user": user}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 50):
    """Lấy danh sách sessions của user"""
    try:
        sessions = chat_history_manager.get_user_sessions(user_id, limit)
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/sessions")
async def create_session(session: SessionCreate):
    """Tạo session mới"""
    try:
        session_id = chat_history_manager.create_session(session.user_id, session.title)
        if session_id:
            return {"status": "success", "session_id": session_id}
        else:
            raise HTTPException(status_code=400, detail="Không thể tạo session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 100):
    """Lấy messages của session"""
    try:
        messages = chat_history_manager.get_session_messages(session_id, limit)
        return {"status": "success", "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, session_update: SessionUpdate):
    """Cập nhật title session"""
    try:
        success = chat_history_manager.update_session_title(session_id, session_update.title)
        if success:
            return {"status": "success", "message": "Session updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Không thể cập nhật session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa session"""
    try:
        success = chat_history_manager.delete_session(session_id)
        if success:
            return {"status": "success", "message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Không thể xóa session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Chat Endpoints
@app.post("/api/chat")
async def chat(inputs: QuestionInput):
    """Chat với bot và lưu lịch sử"""
    try:
        # Nếu không có session_id, tạo session mới
        session_id = inputs.session_id
        if not session_id:
            session_id = chat_history_manager.create_session(inputs.user_id, "New Chat")
            if not session_id:
                raise HTTPException(status_code=400, detail="Không thể tạo session")

        # Lấy conversation context hiện tại (để có thể sử dụng trong tương lai)
        conversation_context = chat_history_manager.get_conversation_context(session_id, 10)

        # Lưu câu hỏi của user
        user_message_id = chat_history_manager.add_message(
            session_id, "user", inputs.question
        )
        
        # Xử lý câu hỏi với graph
        results = graph.invoke({
            "question": inputs.question,
            "session_id": session_id
        })
        #answer = results.get('answer', '')

        raw_answer = results.get('answer', '')
        if isinstance(raw_answer, list) and len(raw_answer) > 0:
            # Nếu answer là list, lấy phần tử đầu tiên
            answer_obj = raw_answer[0]
            if hasattr(answer_obj, 'content'):
                # Nếu là AIMessage, lấy content
                answer = answer_obj.content
            else:
                answer = str(answer_obj)
        elif hasattr(raw_answer, 'content'):
            # Nếu trực tiếp là AIMessage
            answer = raw_answer.content
        else:
            # Fallback to string conversion
            answer = str(raw_answer)
        
        print(f"[DEBUG] Raw answer type: {type(raw_answer)}")
        print(f"[DEBUG] Processed answer: {answer[:100]}...")
        
        # Lưu câu trả lời của assistant
        assistant_message_id = chat_history_manager.add_message(
            session_id, "assistant", answer
        )

        from datetime import datetime

        context_metadata = {
            "last_question": inputs.question,
            "last_answer": answer[:500] + "..." if len(answer) > 500 else answer,
            "conversation_length": len(conversation_context) + 2,
            "timestamp": datetime.now().isoformat(),
            "user_id": inputs.user_id,
            "processing_results": {
                #"route_used": "vectorstore" if "vectorstore" in str(graph) else "wiki_search",
                "has_documents": "documents" in results,
                "result_keys": list(results.keys()) if isinstance(results, dict) else []
            }
        }
        
        context_saved = chat_history_manager.save_session_context(session_id, context_metadata)
        print(f"Context saved: {context_saved}")

        # Tự động cập nhật title session nếu đây là tin nhắn đầu tiên
        session_messages = chat_history_manager.get_session_messages(session_id, 5)
        if len(session_messages) == 2:  # Chỉ có 2 tin nhắn (user + assistant)
            # Tạo title ngắn gọn từ câu hỏi
            title = inputs.question[:50] + "..." if len(inputs.question) > 50 else inputs.question
            chat_history_manager.update_session_title(session_id, title)
        
        return {
            "status": "success",
            "session_id": session_id,
            "answer": answer,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "context_saved": context_saved,
            "conversation_context_lenght": len(conversation_context),
        }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/context")
async def get_conversation_context(session_id: str, last_n_messages: int = 10):
    """Lấy context conversation"""
    try:
        context = chat_history_manager.get_conversation_context(session_id, last_n_messages)
        return {"status": "success", "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/sessions/{session_id}/save_context")
async def save_session_context_endpoint(session_id: str, context_data: dict):
    """Test endpoint để lưu context"""
    try:
        success = chat_history_manager.save_session_context(session_id, context_data)
        if success:
            return {"status": "success", "message": "Context saved"}
        else:
            return {"status": "error", "message": "Failed to save context"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/sessions/{session_id}/statistics")
async def get_session_statistics(session_id: str):
    """Lấy thống kê session"""
    try:
        stats = chat_history_manager.get_session_statistics(session_id)
        return {"status": "success", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Search Endpoints
@app.post("/api/search/messages")
async def search_messages(search: MessageSearch):
    """Tìm kiếm messages"""
    try:
        messages = chat_history_manager.search_messages(search.user_id, search.query)
        return {"status": "success", "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/chat")
# async def chat(inputs: QuestionInput):
#     # results = []
    
#     # # Giả sử bạn đã có `app.stream()` cho phép lấy các output từ câu hỏi.
#     # for output in graph.invoke({"question": inputs.question}):
#     #     output_result = {}
#     #     for key, value in output.items():
#     #         output_result[key] = value
#     #     results.append(output_result)
#     results = graph.invoke({"question": inputs.question})
#     answer = results.get('answer','')
#     #save_chat_history(inputs.user_id, inputs.question, answer)
#     return {"results": results}

# PDF Upload Endpoint
@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Lưu file tạm thời
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    # 1. Trích xuất text
    raw_text = extract_text_from_pdf(file_location)
    # 2. Khởi tạo class thao tác DB
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "hoangducdung",
        "database": "law_db"
    }
    doc_creator = LawDocumentCreator(**db_config)
    # 3. Lưu vào MySQL
    doc_creator.add_chapters_and_articles_to_db(raw_text)
    # 4. Tạo Document từ MySQL
    documents = doc_creator.create_documents_from_db()
    # 5. Upsert vào Qdrant
    qdrant_vectors.upsert_documents(documents)
    return {"status": "success", "filename": file.filename}

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Law Chat API is running"}