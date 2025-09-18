import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def Load_PDF_Files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"The Data path does not exist: {DATA_PATH}")

        logger.info(f"Loading PDF files from directory: {DATA_PATH}")
        
        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()#All data will be loaded here
        
        if not documents:#If no pdfs are found
            raise CustomException(f"No PDF files found in the specified directory: {DATA_PATH}")
        else:
            logger.info(f"Found {len(documents)} PDF files in the directory.")
            
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        raise CustomException(e)

# Split the documents into smaller chunks
def create_text_chunks(documents): 
    try:
        if not documents:
            raise CustomException("No documents to split.")
        
        logger.info("Splitting documents into smaller chunks.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            raise CustomException("Document splitting resulted in no chunks.")
        else:
            logger.info(f"Documents split into {len(docs)} chunks.")
        
        return docs
    
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise CustomException(e)
