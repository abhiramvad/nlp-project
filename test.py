import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer efficiently
model_name = "meta-llama/Llama-3.2-3B"  # You can replace this with any large model

# Option 1: Use mixed precision (FP16) for efficiency on supported GPUs
# If using a supported GPU, the model will automatically use FP16 for mixed precision.
#model = AutoModelForSequenceClassification.from_pretrained(model_name, revision="main", torch_dtype=torch.float16 if device == "cuda" else torch.float32)

# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to appropriate device (GPU or CPU)
#model.to(device)

# Option 2: Use device_map for model parallelism (if multiple GPUs are available)
# If you have multiple GPUs, you can use `device_map="auto"` to distribute the model across GPUs
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

# Optionally, you can load the model in smaller parts if it's too large for the memory:
# For example, using the following line to distribute the model
# model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map={"transformer.h.0": "cuda:0", "transformer.h.1": "cuda:1"})

# Example inference with some dummy text
inputs = tokenizer("This is a test sentence", return_tensors="pt").to(device)
outputs = model(**inputs)

# Get the model's prediction
logits = outputs.logits
prediction = torch.argmax(logits, dim=-1)

print(f"Prediction: {prediction.item()}")
