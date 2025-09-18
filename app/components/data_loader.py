import os
from app.components.pdf_loader import Load_PDF_Files,create_text_chunks
from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def process_and_store_pdfs():
    try:
        logger.info("Making vector store from PDF files...")
        
        documents = Load_PDF_Files()
        
        text_chunks = create_text_chunks(documents)   
        
        save_vector_store(text_chunks)
        logger.info("Vector store created and saved successfully.")
    
    except Exception as e:
        error_meaasge=CustomException(f"failed to create vector store: {e}")
        logger.error(error_meaasge)
        
if __name__ == "__main__":
    process_and_store_pdfs()