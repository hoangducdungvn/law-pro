import pdfplumber
import re
from langchain.schema import Document

def extract_pdf_documents(pdf_path):
    def extract_text_pdf(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

        return text
    
    chapter_pattern = r"(Chương\s+[IVXLCDM]+)\s+([^\n]+)"
    article_pattern = r"(Điều\s+\d+)\.\s+"

    # Trích xuất văn bản từ PDF
    raw_text = extract_text_pdf(pdf_path)
    chapters = re.split(chapter_pattern, raw_text)
    documents = []

    # Nếu không tìm thấy chương nào, trả về danh sách rỗng
    if len(chapters) <= 1:
        print("Không tìm thấy chương nào.")
        return []

    # Xử lý từng chương
    for i in range(1, len(chapters), 3):
        chapter_number = chapters[i].strip()  # Số chương
        chapter_title = chapters[i+1].strip()  # Tiêu đề chương
        chapter_content = chapters[i+2].strip()  # Nội dung chương

        # Tách các điều trong chương
        articles = re.split(article_pattern, chapter_content)

        for j in range(1, len(articles), 2):
            article_number = articles[j].strip()  # Số điều
            article_content = articles[j+1].strip()  # Nội dung điều

            # Tạo đối tượng Document
            document = Document(
                metadata={
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "article_number": article_number
                },
                page_content=f"{article_number}. {article_content}"
            )
            documents.append(document)

    return documents




