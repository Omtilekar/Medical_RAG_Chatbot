from langchain_community.llms import HuggingFaceHub
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from langchain_groq import ChatGroq

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
repo_id="deepseek-ai/DeepSeek-V3.1-Base"
logger=get_logger(__name__)

def load_llm():
    try:
        logger.info("Loading LLM model from Hugging Face Hub...")
        
        llm=ChatGroq(
            repo_id=repo_id,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3, 
            max_new_tokens=256,
            return_full_text=False
            )
        logger.info("LLM model loaded successfully.")
        
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        raise CustomException("Error loading LLM model", e)