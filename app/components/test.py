# test_llm.py
from app.components.llm3 import load_llm_pipeline
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN


print("Repo ID:", HUGGINGFACE_REPO_ID)
print("HF Token loaded:", HF_TOKEN is not None)

    # Load the LLM
llm = load_llm_pipeline()
print("LLM loaded:", llm)

