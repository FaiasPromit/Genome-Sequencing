import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import os

import config
from utils import get_logger
from dataset import SVDataset
from model import DNABERTSVClassifier

logger = get_logger(__name__)

def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    return avg_loss, metrics

def main():
    logger.info(f"Using device: {config.DEVICE}")
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # 1. Load Dataset
    full_dataset = SVDataset(
        tsv_path=config.PROCESSED_DATA_TSV,
        tokenizer_name=config.MODEL_NAME,
        max_length=config.MAX_TOKEN_LENGTH
    )
    
    # 2. Split dataset
    val_size = int(len(full_dataset) * config.VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # 3. Initialize Model, Optimizer, Loss, Scheduler
    model = DNABERTSVClassifier(config.MODEL_NAME).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 4. Training loop
    best_f1 = 0
    for epoch in range(config.EPOCHS):
        logger.info(f"--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE, scheduler)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        val_loss, metrics = eval_model(model, val_loader, loss_fn, config.DEVICE)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Metrics: {metrics}")
        
        # Save the best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved to {config.MODEL_SAVE_PATH} with F1-score: {best_f1:.4f}")

    logger.info("Training complete.")

if __name__ == "__main__":
    main()