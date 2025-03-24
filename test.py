from transformers import AutoTokenizer, AutoModelForCausalLM
hf_token = "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"
# Load the tokenizer
print("Loading tokenizer...")
# Load model directly
#  # Replace this with your actual Hugging Face token
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B",use_access_token=hf_token)
print("Model Loaded")