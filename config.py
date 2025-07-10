import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_MODEL = "intfloat/multilingual-e5-base"
FAST_EMBED_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
LLM_MODEL = "llama-3.1-8b-instant"
COLLECTION_NAME = "law_collection"
VN_LAW_PDF_PATH = "VanBanGoc_52.2014.QH13.pdf"
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")