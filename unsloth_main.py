import pandas as pd
from datasets import Dataset
import torch
import os
import sys

# Unsloth and TRL imports
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# For secure token input
from getpass import getpass

hf_token = "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"

if not hf_token:
    print("Error: Hugging Face token not found. Please provide a valid token.")
    sys.exit(1)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 2: Load and Prepare Data
try:
    df = pd.read_csv('labeled_persuasion_transcripts.csv').iloc[:3000]
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'labeled_persuasion_transcripts.csv' not found in the current directory.")
    sys.exit(1)

df.head()

def convert_label(y):
    """
    Convert continuous belief change scores to binary labels.
    - 0: No belief change (belief change <= 10)
    - 1: Belief change (belief change > 10)
    """
    return 1 if y > 10 else 0

df['label'] = df['y'].apply(convert_label)
print("Labels converted to binary.")

def prepare_conversation(row):
    """
    Format each row into a conversation suitable for Unsloth.
    Separates the prompt and target for accurate loss computation.
    """
    conversations = [
        {"role": "user", "content": f"{row['X']}"},
        {"role": "assistant", "content": f"{row['label']}"}
    ]
    return {"conversations": conversations}

# Apply the formatting to the dataframe
data = df.apply(prepare_conversation, axis=1)
dataset = Dataset.from_pandas(pd.DataFrame(data.tolist()))
print("Dataset prepared with conversations. Sample entry:")
print(dataset[0])

from unsloth.chat_templates import get_chat_template

# Define model parameters
max_seq_length = 2048  # Adjust based on your data
dtype = None  # Auto-detect; use torch.float16 if desired
load_in_4bit = True  # Use 4-bit quantization to save memory

# List of supported 4-bit models by Unsloth
fourbit_models = [
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    # Add other models if needed
]

# Choose the appropriate model from the list
chosen_model = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

# Load the model and tokenizer using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=chosen_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token,  # Use this if accessing gated models
)

print("Model and tokenizer loaded successfully.")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank; higher means more capacity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,  # Can adjust for regularization
    bias="none",  # Typically "none" for optimized performance
    use_gradient_checkpointing="unsloth",  # Optimizes memory usage
    random_state=3407,
    use_rslora=False,  # Set to True if using Rank Stabilized LoRA
    loftq_config=None,  # Only if using LoftQ
)

print("LoRA adapters added to the model.")

from unsloth.chat_templates import get_chat_template, standardize_sharegpt

# Apply the appropriate chat template
chat_template = "llama-3.1"  # Choose based on your model's requirements

tokenizer = get_chat_template(
    tokenizer,
    chat_template=chat_template,
)

def format_conversations(examples):
    """
    Convert conversations into a single text field as per the chat template.
    Separates prompt and target for proper label masking.
    """
    convos = examples["conversations"]
    texts = []
    prompts = []
    for convo in convos:
        # Generate the formatted text
        formatted_text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted_text)
        
        # Extract the prompt (user part)
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{convo[0]['content']}\n<|eot_id|>\n"
        prompts.append(prompt)
    
    return {"text": texts, "prompt": prompts}

# Apply the formatting to the dataset
dataset = dataset.map(format_conversations, batched=True)
print("Conversations formatted according to the chat template.")
print("Sample formatted text:")
print(dataset[0]["text"])

# Tokenization function for the training
def tokenize_function(example):
    """
    Tokenize the input text and set labels to -100 for prompt tokens.
    Only the target tokens (0 or 1) are used for loss computation.
    """
    # Tokenize the full text (prompt + target)
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    
    # Tokenize the prompt to find its length
    tokenized_prompt = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    prompt_length = len(tokenized_prompt["input_ids"])
    
    # Create labels by cloning input IDs
    labels = tokenized["input_ids"].copy()
    
    # Set labels for prompt tokens to -100 to ignore them in loss computation
    labels[:prompt_length] = [-100] * prompt_length
    
    # Assign labels to the tokenized data
    tokenized["labels"] = labels
    
    return tokenized

# Apply tokenization
print("Tokenizing the dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["conversations", "text", "prompt"]
)
print("Tokenization complete.")

# Split the dataset into training and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
print("Dataset split into training and validation sets.")

# Define a data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

# Define training arguments with checkpointing
training_args = TrainingArguments(
    output_dir='./fine_tuned_model_binary',  # Directory to save checkpoints and model
    overwrite_output_dir=True,
    num_train_epochs=1,               # Adjust as needed
    per_device_train_batch_size=4,    # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",             # Save checkpoints every `save_steps`
    save_steps=200,
    save_total_limit=5,                # Keep last 5 checkpoints
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),    # Use mixed precision if available
    report_to="none",                  # Disable reporting to WandB or other services
    load_best_model_at_end=True,       # Load the best model at the end based on evaluation metric
    metric_for_best_model="f1",        # Use F1 score to determine the best model
    greater_is_better=True,
)

print("Training arguments defined.")

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=data_collator,
    dataset_num_proc=2,
    packing=False,  # Can speed up training for short sequences
    args=training_args,
)

print("Trainer initialized.")

# Apply train_on_responses_only to mask user prompts
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n"
)

print("Applied train_on_responses_only to mask user prompts.")

# Define compute_metrics function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for binary classification.
    """
    logits, labels = eval_pred
    # Apply sigmoid to logits to get probabilities
    probs = torch.sigmoid(torch.tensor(logits)).squeeze()
    preds = (probs > 0.5).int().numpy()
    
    # Flatten labels and preds
    preds = preds.flatten()
    labels = labels.flatten()
    
    # Remove labels that are -100
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Assign compute_metrics to the trainer
trainer.compute_metrics = compute_metrics

print("Compute_metrics function assigned to trainer.")

# Train the model
print("Starting training...")
trainer.train()
print("Training complete.")

# Evaluate the model
print("Evaluating the model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the fine-tuned LoRA adapters
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("LoRA adapters saved successfully.")

# Example Inference Function
from unsloth import FastLanguageModel

# Load the fine-tuned model and tokenizer with Unsloth
model_path = './lora_model'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native faster inference
FastLanguageModel.for_inference(model)

def predict_belief_change(transcript):
    """
    Predict belief change based on the dialogue transcript.
    Returns 0 (no change) or 1 (change).
    """
    # Format the conversation
    messages = [
        {"role": "user", "content": f"{transcript}"}
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to(device)

    # Generate the response (0 or 1)
    output = model.generate(
        input_ids=input_text,
        max_new_tokens=1,
        do_sample=False,                # Deterministic output
        temperature=0.0,                # Avoid randomness
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated token
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    # Extract the prediction
    prediction_str = generated_text.split('Belief Change (0 for no change, 1 for change):')[-1].strip()

    # Convert to integer
    if prediction_str.startswith('0'):
        return 0
    elif prediction_str.startswith('1'):
        return 1
    else:
        # Handle unexpected outputs
        print(f"Unexpected prediction: '{prediction_str}'. Defaulting to 0.")
        return 0

# Example usage
transcript = """
Dialogue:
Persuader: Have you considered that the evidence might suggest otherwise?
Target: I'm not sure, maybe I should look into it.

Based on the dialogue between a persuader and a target, predict whether the target's belief changed as a result of the conversation.

Belief Change (0 for no change, 1 for change):
"""

prediction = predict_belief_change(transcript)
print(f"Predicted Belief Change: {prediction}")