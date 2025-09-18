# test_token.py
import app.config.config as config  # adjust the import path if needed

# Print the token for debugging (only in dev)
print("HF_TOKEN loaded:", config.HF_TOKEN)

# Safer check without printing the actual token
if config.HF_TOKEN:
    print("HF_TOKEN is loaded successfully!")
else:
    print("HF_TOKEN not found. Check your .env or environment variables.")
