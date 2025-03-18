import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("open-web-math/open-web-math")

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-1b")

    # Tokenize the dataset
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["url", "date", "metadata"])

    # Split the dataset into train and validation sets
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # Load the model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-1b")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./llama3-math-pretrain",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),  # Use mixed precision if supported
        report_to="none",  # Disable reporting to external services
    )

    # Initialize the Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving the model...")
    model.save_pretrained("./llama3-math-pretrain")
    tokenizer.save_pretrained("./llama3-math-pretrain")

    print("Training complete!")

if __name__ == "__main__":
    main()