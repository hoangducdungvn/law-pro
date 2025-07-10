from langchain.schema import Document
import MySQLdb
import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_chapters_and_articles(raw_text):
    chapter_pattern = r"(Chương\s+[IVXLCDM]+)\s+([^\n]+)"
    article_pattern = r"(Điều\s+\d+)\.\s+"
    chapters = re.split(chapter_pattern, raw_text)
    result = []
    if len(chapters) <= 1:
        print("Không tìm thấy chương nào.")
        return []
    for i in range(1, len(chapters), 3):
        chapter_number = chapters[i].strip()
        chapter_title = chapters[i+1].strip()
        chapter_content = chapters[i+2].strip()
        articles = re.split(article_pattern, chapter_content)
        article_list = []
        for j in range(1, len(articles), 2):
            article_number = articles[j].strip()
            article_content = articles[j+1].strip()
            article_list.append({
                "article_number": article_number,
                "content": article_content
            })
        result.append({
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "articles": article_list
        })
    return result

class LawDocumentCreator:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect_to_db(self):
        try:
            return MySQLdb.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except MySQLdb.Error as e:
            print(f"Lỗi kết nối MySQL: {e}")
            return None

    def add_chapters_and_articles_to_db(self, raw_text):
        chapters_and_articles = extract_chapters_and_articles(raw_text)
        if not chapters_and_articles:
            print("Không có nội dung để thêm vào cơ sở dữ liệu.")
            return
        db = self.connect_to_db()
        if not db:
            return
        cursor = db.cursor()
        for chapter in chapters_and_articles:
            cursor.execute(
                "INSERT INTO chapters (chapter_number, chapter_title, chapter_content) VALUES (%s, %s, %s)",
                (chapter['chapter_number'], chapter['chapter_title'], chapter['chapter_title'])
            )
            chapter_id = cursor.lastrowid
            for article in chapter['articles']:
                cursor.execute(
                    "INSERT INTO articles (chapter_id, article_number, article_content) VALUES (%s, %s, %s)",
                    (chapter_id, article['article_number'], article['content'])
                )
        db.commit()
        cursor.close()
        db.close()
        print("Đã thêm các chương và điều vào cơ sở dữ liệu.")

    def create_documents_from_db(self):
        db = self.connect_to_db()
        if not db:
            return []
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        try:
            query = """
                SELECT 
                    chapters.chapter_number, 
                    chapters.chapter_title, 
                    articles.article_number, 
                    articles.article_content
                FROM articles
                INNER JOIN chapters ON articles.chapter_id = chapters.id
            """
            cursor.execute(query)
            records = cursor.fetchall()
            documents = []
            for record in records:
                metadata = {
                    "chapter_number": record["chapter_number"],
                    "chapter_title": record["chapter_title"],
                    "article_number": record["article_number"]
                }
                content = record["article_number"] + " " + record["article_content"]
                document = Document(page_content=content, metadata=metadata)
                documents.append(document)
            return documents
        except MySQLdb.Error as e:
            print(f"Lỗi truy vấn MySQL: {e}")
            return []
        finally:
            cursor.close()
            db.close()

# Sử dụng class
# if __name__ == "__main__":
#     db_config = {
#         "host": "localhost",
#         "user": "root",
#         "password": "linhdinhvia@123",
#         "database": "law_db"
#     }
    
#     # Tạo instance của class
#     document_creator = LawDocumentCreator(**db_config)
#     documents = document_creator.create_documents_from_db()
    
#     # Kiểm tra kết quả
#     for doc in documents[:5]:  # In ra 5 document đầu tiên để kiểm tra
#         print(doc)