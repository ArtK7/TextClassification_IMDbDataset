# =====================
# Import Libraries
# =====================
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from colorama import Fore, Style  # For colored output from tqdm import tqdm  # Progress bar for tokenization

# =====================
# 1. Load IMDb Dataset
# =====================
print(Fore.MAGENTA + "Step 1: Loading IMDb dataset..." + Style.RESET_ALL)
dataset = load_dataset("imdb")

# =====================
# 2. Preprocess the Data
# =====================
print(Fore.MAGENTA + "Step 2: Tokenizing the dataset..." + Style.RESET_ALL)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset with a progress bar
tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, desc="Tokenizing Data")

# =====================
# 3. Dynamic Balancing - Total Size = 250
# =====================
print(Fore.MAGENTA + "Step 3: Shuffling and dynamically balancing the dataset..." + Style.RESET_ALL)
test_data = tokenized_datasets["test"].shuffle(seed=42)

# Separate by labels
positive_samples = [ex for ex in test_data if ex['label'] == 1]
negative_samples = [ex for ex in test_data if ex['label'] == 0]

# Ensure balanced proportions based on the smaller class size and total limit
sample_size = 250 // 2  # Split evenly for dynamic balance
balanced_data = random.sample(positive_samples, sample_size) + random.sample(negative_samples, sample_size)
random.shuffle(balanced_data)  # Shuffle again to make it more random

print(Fore.WHITE + f"Balanced dataset size: {len(balanced_data)}" + Style.RESET_ALL)

# =====================
# 4. Load Pre-trained Fine-Tuned DistilBERT Model
# =====================
print(Fore.MAGENTA + "Step 4: Loading pre-trained fine-tuned DistilBERT model..." + Style.RESET_ALL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)

# =====================
# 5. Process the dataset
# =====================
print(Fore.MAGENTA + "Step 5: Making predictions on the full dataset..." + Style.RESET_ALL)
inputs = tokenizer(
    [ex["text"] for ex in balanced_data],
    padding=True, truncation=True, return_tensors="pt"
).to(device)

true_labels = torch.tensor([ex["label"] for ex in balanced_data]).to(device)

# Get the predictions
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

# =====================
# 6. Count the sum of Positive and Negative Reviews
# =====================
positive_count = (predictions == 1).sum().item()
negative_count = (predictions == 0).sum().item()

print(Fore.GREEN + f"Positive Reviews: {positive_count}" + Style.RESET_ALL)
print(Fore.RED + f"Negative Reviews: {negative_count}" + Style.RESET_ALL)

# =====================
# 7. Evaluate Performance Metrics
# =====================
print(Fore.MAGENTA + "Step 7: Evaluating performance metrics..." + Style.RESET_ALL)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels.cpu().tolist(), predictions.cpu().tolist(), average='binary', zero_division=0
)
accuracy = accuracy_score(true_labels.cpu().tolist(), predictions.cpu().tolist())

# Print evaluation results
print(Fore.WHITE + "\nEvaluation Results:" + Style.RESET_ALL)
print(f"{Fore.WHITE}{'Metric':<12}{'Value':<10}{Style.RESET_ALL}")
print(f"{Fore.WHITE}{'-' * 22}{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'Accuracy':<12}{accuracy:.2f}{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'Precision':<12}{precision:.2f}{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'Recall':<12}{recall:.2f}{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}{'F1 Score':<12}{f1:.2f}{Style.RESET_ALL}")