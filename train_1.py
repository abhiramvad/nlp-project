from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_dataset("open-web-math/open-web-math")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B",use_auth_token=True)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",device="auto")

print("Model loaded successfully!")

import torch
from transformers import Trainer, TrainingArguments

split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

#small_train_dataset = train_dataset.select(range(100))
#small_eval_dataset = eval_dataset.select(range(50))

# Assign a padding token to the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the eos_token as the pad_token

# Tokenize the small sample dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for loss computation
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama3-math-pretrain-small",
    evaluation_strategy="steps",
    eval_steps=10, 
    save_steps=10, 
    per_device_train_batch_size=1, 
    num_train_epochs=1,  # Train for 1 epoch
    logging_dir="./logs",
    logging_steps=5,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=10,
    fp16=torch.cuda.is_available(),  # Use mixed precision if supported
    report_to="wandb",  # Disable reporting to external services
)
device = torch.device("cuda")
# Initialize the Trainer
trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train the model
print("Starting training on small sample...")
trainer.train()
print("Training complete!")