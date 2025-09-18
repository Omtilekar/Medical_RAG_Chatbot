from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
import os
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading FAISS vector store from path...")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True)
        else:
            logger.error(f"NO vector store found at path")
            
    except Exception as e:
        error_msg = f"Error loading FAISS vector store: {e}"
        logger.error(error_msg)
        
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save to vector store.")
        logger.info("Generating your new vector DB...")
        
        embedding_model = get_embedding_model()
        
        db = FAISS.from_documents(
            text_chunks,
            embedding_model)
        logger.info("Savinging your vector DB...") 
        db.save_local(DB_FAISS_PATH)
        logger.info(f"FAISS vector store saved successfully at path.")
        
        return db
        
    except Exception as e:
        error_msg = f"failed to create FAISS vector store: {e}"
        logger.error(error_msg)
        raise CustomException(error_msg)
    
