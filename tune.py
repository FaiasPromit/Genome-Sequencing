import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import config
from utils import get_logger
from dataset import SVDataset
from model import DNABERTSVClassifier
from train import train_epoch, eval_model  # Reuse training functions

logger = get_logger(__name__)

def tune_trainable(tune_config):
    """Trainable function for Ray Tune."""
    
    # --- Data Loading ---
    full_dataset = SVDataset(
        tsv_path=config.PROCESSED_DATA_TSV,
        tokenizer_name=config.MODEL_NAME,
        max_length=config.MAX_TOKEN_LENGTH
    )
    val_size = int(len(full_dataset) * config.VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=int(tune_config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(tune_config["batch_size"]))
    
    # --- Model Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DNABERTSVClassifier(config.MODEL_NAME).to(device)
    
    optimizer = AdamW(model.parameters(), lr=tune_config["lr"], weight_decay=tune_config["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(config.EPOCHS):
        # We don't need the scheduler for tuning to keep it simple
        train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler=None) # Pass None for scheduler
        
        val_loss, metrics = eval_model(model, val_loader, loss_fn, device)
        
        # Report metrics to Ray Tune
        tune.report(
            loss=val_loss,
            accuracy=metrics['accuracy'],
            f1=metrics['f1']
        )

def main():
    logger.info("Starting hyperparameter tuning with Ray Tune.")
    
    # --- Define Search Space ---
    search_space = {
        "lr": tune.loguniform(1e-5, 5e-5),
        "batch_size": tune.choice([4, 8, 16]),
        "weight_decay": tune.uniform(0.0, 0.1)
    }

    # --- Define Search Algorithm and Scheduler ---
    search_alg = HyperOptSearch(metric="f1", mode="max")
    scheduler = ASHAScheduler(metric="f1", mode="max", grace_period=1, reduction_factor=2)

    # --- Run Tuner ---
    tuner = tune.Tuner(
        tune.with_resources(tune_trainable, {"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=10, # Number of different hyperparameter combinations to try
            search_alg=search_alg,
            scheduler=scheduler,
        ),
        param_space=search_space,
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result("f1", "max")
    
    logger.info("Best trial config: {}".format(best_result.config))
    logger.info("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    logger.info("Best trial final validation F1: {}".format(best_result.metrics["f1"]))

if __name__ == "__main__":
    main()