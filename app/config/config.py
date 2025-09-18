import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID="deepseek-ai/DeepSeek-V3.1-Base"
DB_FAISS_PATH="vectorstore/db_faiss"
DATA_PATH="O:/DSA/Medical_RAG_Chatbot/data/"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

