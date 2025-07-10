from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

class ToolSearch():
    """
    Một lớp để tương tác với các công cụ Arxiv và Wikipedia.

    """

    def __init__(self, arxiv_top_k: int = 1, arxiv_chars_max: int = 512, wiki_top_k: int = 1, wiki_chars_max: int = 512):
        """
        Khởi tạo lớp ToolSearch với cấu hình cho Arxiv và Wikipedia.

        Args:
            arxiv_top_k (int): Số lượng kết quả hàng đầu để lấy từ Arxiv.
            arxiv_chars_max (int): Số ký tự tối đa cho nội dung tài liệu Arxiv.
            wiki_top_k (int): Số lượng kết quả hàng đầu để lấy từ Wikipedia.
            wiki_chars_max (int): Số ký tự tối đa cho nội dung tài liệu Wikipedia.

        """
        self.arxiv_wrapper = ArxivAPIWrapper(top_k_results=arxiv_top_k, doc_content_chars_max=arxiv_chars_max)
        self.arxiv = ArxivQueryRun(api_wrapper=self.arxiv_wrapper)

        self.wiki_wrapper = WikipediaAPIWrapper(top_k_results=wiki_top_k, doc_content_chars_max=wiki_chars_max)
        self.wiki = WikipediaQueryRun(api_wrapper=self.wiki_wrapper)

    def query_arxiv(self, query: str):
        return self.arxiv.run(query)

    def query_wikipedia(self, query: str):
        return self.wiki.run(query)

#Example usage
if __name__ == "__main__":
    retriever = ToolSearch()
    arxiv_result = retriever.query_arxiv("machine learning")
    print("Arxiv Result:", arxiv_result)

    wiki_result = retriever.query_wikipedia("Artificial Intelligence")
    print("Wikipedia Result:", wiki_result)