from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os

class QueryRouter():
    """
    Class cho việc định tuyến câu hỏi của người dùng tới nguồn thông tin thích hợp.
    """
    class RouteQuery(BaseModel):
        """
        Dữ liệu đầu vào cho việc định tuyến câu hỏi.
        """
        datasource: Literal["vectorstore", "wiki_search"] = Field(
            ..., description=" Gửi một câu hỏi từ người dùng để chọn định tuyến nó tới wikipedia hoặc vectorstore."
        )

    def __init__(self, llm):
        """
        Khởi tạo QueryRouter với mô hình LLM và API key.

        Args:
            groq_api_key (str): API key cho dịch vụ Groq.
            model_name (str): Tên của mô hình LLM để sử dụng.
        """
        self.llm = llm
        self.structured_llm_router = self.llm.with_structured_output(self.RouteQuery)
        
        self.system_prompt = (
            """
            Bạn là một chuyên gia trong việc định tuyến câu hỏi của người dùng tới nguồn thông tin thích hợp.
            - Nếu câu hỏi liên quan đến Luật Hôn Nhân và Gia Đình Việt Nam 2014, hãy chọn 'vectorstore' làm nguồn tài liệu.
            - Nếu câu hỏi không liên quan đến các chủ đề trong Luật Hôn Nhân và Gia Đình Việt Nam 2014, hãy chọn 'wiki_search' để tìm kiếm thông tin từ Wikipedia.
            """
        )

        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

    def route_question(self, question: str):
        """
        Định tuyến một câu hỏi tới nguồn thông tin thích hợp.

        Args:
            question (str): Câu hỏi của người dùng để định tuyến.

        Returns:
            dict: Kết quả định tuyến.
           
        """
        question_router = self.route_prompt | self.structured_llm_router
        return question_router.invoke({"question": question})