from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from app.components.llm3 import load_llm_pipeline as load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os

logger=get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical questions in 2-3 lines using only the provided context. If the answer is not contained within the context, respond with "I don't know".
Context: {context}

Question: {question}

"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

def create_qa_chain():
    try:
        logger.info("loading vector store for context retrieval...")
        
        db=load_vector_store()
        if db is None:
            raise CustomException("Vector store is not loaded properly.")
        else:
            logger.info("Vector store loaded successfully $$$.")
        
        llm = load_llm()
        if llm is None:
            raise CustomException("LLM model is not loaded properly.")
        else:
            logger.info("LLM model loaded successfully $$$.")
        
        #prompt = set_custom_prompt()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k":1}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )
        
        logger.info("QA chain created successfully.")  
        return qa_chain
    
    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        # 🚨 Explicitly return None on failure
        return None

