from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification , Trainer, TrainingArguments
dataset = load_dataset("open-web-math/open-web-math")
print(len(dataset['train']))