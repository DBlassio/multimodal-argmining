"""
text_LLM_classifier.py
------------------------------------------------------------
Fine-tune and evaluate a text classification model using 
Hugging Face Transformers with configuration loaded from JSON.
------------------------------------------------------------
"""
print("="*60)
print(f"Text LLM Classifier Training Script")
print("="*60)

# ===============================
# Libraries
# ===============================
print("Loading libraries...")
import os
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ===============================
# Load Configuration
# ===============================
print("Loading configuration...")
config_path = os.path.join(os.path.dirname(__file__), "text_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

MODEL_NAME = config["model_parameters"]["model_name"]
MAX_LEN = config["model_parameters"]["max_length"]
SEED = config["misc"]["seed"]
AUGMENTED = config["data_settings"]["use_augmented_data"]

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===============================
# Environment Setup
# ===============================
print("Setting up environment...")
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    path = config["paths"]["data"]["base_path_colab"]
else:
    path = config["paths"]["data"]["base_path_local"]

os.chdir(path)
print(f"Working directory: {path}")

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# Load Dataset
# ===============================
print("Loading dataset...")
if AUGMENTED:
    train_path = os.path.join(path, config['paths']['data']['train_dataset_augmented'])
else:
    train_path = os.path.join(path, config['paths']['data']['train_dataset'])

dev_path = os.path.join(path, config['paths']['data']['dev_dataset'])
test_path = os.path.join(path, config['paths']['data']['test_dataset'])

df_train = pd.read_csv(train_path)
df_dev = pd.read_csv(dev_path)
df_test = pd.read_csv(test_path)

label2id = {"oppose": 0, "support": 1}

def clean_text(text):
    """Basic cleaning for text inputs."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    return text

for df in [df_train, df_dev, df_test]:
    #df["tweet_text"] = df["tweet_text"].apply(clean_text)
    df["label"] = df["stance"].map(label2id)

dataset_train = Dataset.from_pandas(df_train[["tweet_text", "label"]])
dataset_dev = Dataset.from_pandas(df_dev[["tweet_text", "label"]])
dataset_test = Dataset.from_pandas(df_test[["tweet_text", "label"]])


# ===============================
# Tokenization
# ===============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
print(f"Tokenizer loaded: {MODEL_NAME}")

def tokenize_dataset(dataset):
    """Tokenize dataset for model input."""
    def tokenize_batch(batch):
        return tokenizer(
            batch["tweet_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

train_dataset_tok = tokenize_dataset(dataset_train)
dev_dataset_tok = tokenize_dataset(dataset_dev)
test_dataset_tok = tokenize_dataset(dataset_test)
print("Tokenization complete.")


# ===============================
# Model Setup
# ===============================
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2,
                                                                hidden_dropout_prob=config["model_parameters"]["dropout"],
                                                                attention_probs_dropout_prob=config["model_parameters"]["dropout"]).to(device)
print(f"Model loaded: {MODEL_NAME}")
# ===============================
# Metrics
# ===============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "f1": f1_macro,
        "precision": precision,
        "recall": recall
    }


# ===============================
# Trainer Configuration
# ===============================
print("Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=config["paths"]["output_dir"],
    eval_strategy=config["training_parameters"]["eval_strategy"],
    save_strategy=config["training_parameters"]["save_strategy"],
    logging_strategy=config["training_parameters"]["logging_strategy"],
    learning_rate=config["training_parameters"]["learning_rate"],
    per_device_train_batch_size=config["training_parameters"]["batch_size"],
    per_device_eval_batch_size=config["training_parameters"]["batch_size"],
    num_train_epochs=config["training_parameters"]["num_train_epochs"],
    weight_decay=config["training_parameters"]["weight_decay"],
    warmup_ratio=config["training_parameters"]["warmup_ratio"],
    load_best_model_at_end=config["training_parameters"]["load_best_model_at_end"],
    report_to=config["training_parameters"]["report_to"],
    metric_for_best_model=config["training_parameters"]["metric_for_best_model"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tok,
    eval_dataset=dev_dataset_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ===============================
# Training
# ===============================
print("Starting training with the following parameters: ")

print(f"\nModel Name: {MODEL_NAME}")
print(f"Max Sequence Length: {MAX_LEN}")
print(f"Use Augmented Data: {AUGMENTED}")
print(f"Random Seed: {SEED}")
print("Training Parameters:")
for param in ["learning_rate", "weight_decay", "batch_size", "num_train_epochs", "warmup_ratio",
              "eval_strategy", "save_strategy", "logging_strategy", "load_best_model_at_end",
              "metric_for_best_model", "report_to"]:
    value = config["training_parameters"].get(param)
    print(f"  {param}: {value}")
trainer.train()


# ===============================
# Evaluation
# ===============================
print("\nEvaluating model on test dataset...")
predictions = trainer.predict(test_dataset_tok)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1_macro = f1_score(y_true, y_pred, average="macro")

print(f"\nResults for {MODEL_NAME}:")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1_macro:.4f}")

# Save metrics to CSV
results_df = pd.DataFrame([{
    "model": MODEL_NAME,
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_macro,
    "augmented": AUGMENTED,
    "seed": SEED
}])
save_path = os.path.join(config["paths"]["output_dir"], "results_summary.csv")
results_df.to_csv(save_path, index=False)
print(f"\nMetrics saved to: {save_path}")


# ===============================
# Save Model for Inference
# ===============================
saved_model_path = os.path.join(config["paths"]["saved_model_path"], f"text_classifier_{MODEL_NAME}")
os.makedirs(saved_model_path, exist_ok=True)
model.save_pretrained(saved_model_path)
tokenizer.save_pretrained(saved_model_path)
print(f"\nModel and tokenizer saved to: {saved_model_path}")


# ===============================
# Confusion Matrix
# ===============================
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["oppose", "support"],
    yticklabels=["oppose", "support"]
)
plt.title(f"Confusion Matrix - {MODEL_NAME}\nF1: {f1_macro:.4f}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
safe_model_name = MODEL_NAME.replace("/", "_").replace("\\", "_")
save_path = os.path.join(config["paths"]["output_dir"], f"confusion_matrix_{safe_model_name}.jpg")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Confusion matrix saved to: {save_path}")

