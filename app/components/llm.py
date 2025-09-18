from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

repo_id = "deepseek-ai/DeepSeek-V3.1-Base"
logger = get_logger(__name__)

def load_llm():
    try:
        logger.info("Loading LLM model from Hugging Face Hub...")

        # Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            token=HF_TOKEN,
            device_map="auto",   # puts model on GPU if available
        )

        # Create pipeline
        llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.3,
        )

        logger.info("LLM model loaded successfully.")
        return llm

    except Exception as e:
        logger.error(f"Error loading LLM model: {e}")
        raise CustomException("Error loading LLM model", e)
