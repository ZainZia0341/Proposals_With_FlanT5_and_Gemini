from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pre-trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

# Load your dataset
dataset = load_dataset("csv", data_files="proposals_dataset.csv")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["job_description"]
    targets = examples["proposal"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"]
)

# Fine-tune the model
trainer.train()
