import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Load the Excel file with phishing/benign URLs
df = pd.read_excel('./data/data_bal - 20000.xlsx')

# Split the data into training and testing sets
train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
test_df = df.drop(train_df.index)                # 20% for testing

# Convert the pandas dataframe to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the pre-trained DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Move the model to GPU
model.to(device)
print("Model is on device:", next(model.parameters()).device)

# Tokenize and preprocess the URLs
def preprocess_function(examples):
    tokenized_examples = tokenizer(examples['URLs'], truncation=True, padding='max_length', max_length=64)
    tokenized_examples['label'] = examples['Labels']  # Adjust for 'Labels' column in the Excel file
    return tokenized_examples

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.remove_columns(["URLs"])
test_dataset = test_dataset.remove_columns(["URLs"])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,  # Increase if memory allows
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=1,
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
    fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if CUDA is available
    learning_rate=1e-5,
    max_grad_norm=1.0,
    report_to="none",  # Avoid logging errors
)


# Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print("Trainer is using device:", trainer.args.device)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./model/phishing_model')
tokenizer.save_pretrained('./model/phishing_model')
