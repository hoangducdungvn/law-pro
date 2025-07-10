from qdrant_client import QdrantClient
from .document_extract import extract_pdf_documents
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import PointStruct
import pandas as pd

class QdrantVT():
    def __init__(self, qdrant_uri = None, collectionName = None, embeddingModel = None, SparseModel = None, Document = None):
        self.client = QdrantClient(qdrant_uri)
        self.collection = collectionName
        self.embeddings = embeddingModel
        self.sparse = SparseModel
        self.documents = Document
        self.qdrant = None

    def load_documents(self, pdf_path):
        self.documents = extract_pdf_documents(pdf_path)
        return self.documents
    
    def init_vectorstore(self):
        self.qdrant = QdrantVectorStore.from_documents(
            documents = self.documents,
            collection_name = self.collection,
            embedding = self.embeddings,
            url = 'http://localhost:6333',
            sparse_embedding = self.sparse,
            retrieval_mode=RetrievalMode.HYBRID
        )

        retriever = self.qdrant.as_retriever(search_kwangs={"k": 10})
        return retriever
    
    def load_vectorstore(self):
        self.qdrant = QdrantVectorStore(
            client = self.client,
            collection_name = self.collection,
            embedding = self.embeddings,
            sparse_embedding= self.sparse,
            retrieval_mode=RetrievalMode.HYBRID
        )

        retriever = self.qdrant.as_retriever(search_kwangs={"k": 10})
        return retriever
    
    def search(self, query):
        results = self.qdrant.similarity_search(query)
        text = ""
        for i,result in enumerate(results):
            text += f"{i+1}. {result.metadata['chapter_number']} - {result.metadata['chapter_title']} - {result.metadata['article_number']}\n {result.page_content} \n"
        
        return text
    
    def upsert_documents(self, documents, batch_size=2):
        data = []
        for doc in documents:
            data.append({
                "page_content": doc.page_content,
                "chapter_number": doc.metadata["chapter_number"],
                "chapter_title": doc.metadata["chapter_title"],
                "article_number": doc.metadata["article_number"]
            })

        df = pd.DataFrame(data)
        # Tạo embedding cho từng văn bản
        df['embedding'] = df['page_content'].apply(lambda x: self.embeddings.embed_documents(x))
        # Upsert theo batch
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]
            points = [
                PointStruct(
                    id=index,
                    vector=row['embedding'][0],
                    payload=row.drop(labels=['embedding']).to_dict()
                )
                for index, row in batch_df.iterrows()
            ]
            self.client.upsert(collection_name=self.collection, points=points)
        print(f"Đã upsert {len(df)} documents vào Qdrant.")


    



        
        