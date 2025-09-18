import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.config.config import HF_TOKEN # Assuming HF_TOKEN is still needed for private models
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_community.llms import HuggingFacePipeline

# Define the repository ID for the Hugging Face model
repo_id = "openai-community/gpt2"
logger = get_logger(__name__)

def load_llm_pipeline():
    
    try:
        logger.info(f"Loading tokenizer from Hugging Face Hub: {repo_id}...")
        # Load the tokenizer associated with the model
        # The token might be required if the model is private or gated.
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
        logger.info("Tokenizer loaded successfully.")

        logger.info(f"Loading model from Hugging Face Hub: {repo_id}...")
        # Load the pre-trained model for causal language modeling
        # Using torch.bfloat16 for better performance on compatible hardware (e.g., Ampere GPUs)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto" # Automatically select the device (GPU if available)
        )
        logger.info("Model loaded successfully.")

        logger.info("Creating text-generation pipeline...")
        # Create a pipeline for text generation. This is a high-level API
        # for easy inference.
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.3,
            max_new_tokens=512,
            return_full_text=False
        )
        logger.info("Text-generation pipeline created successfully.")
        
        # Wrap pipeline into LangChain-compatible LLM
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        logger.info("HuggingFacePipeline wrapper created successfully.")

        return llm
        

    except Exception as e:
        logger.error(f"Error loading model or creating pipeline: {e}")
        # Re-raise as a custom exception for consistent error handling in the app
        raise CustomException("Error loading LLM model or creating pipeline", e)

# Example of how you might define the placeholder modules for context
# You would have these defined properly in your application structure.

#