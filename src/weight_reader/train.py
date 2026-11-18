"""Reproducible CPU training with validation-loss checkpoint selection."""
import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import WeightReaderMLP


def train_model(train_split, validation_split, options, seed):
    torch.manual_seed(seed)
    model = WeightReaderMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(*train_split), batch_size=options['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_loss, best_epoch, best_state, stale = float('inf'), 0, None, 0
    start = time.monotonic()
    for epoch in range(1, options['max_epochs'] + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(validation_split[0])
            val_loss = criterion(logits, validation_split[1]).item()
        if val_loss < best_loss:
            best_loss, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= options['early_stopping_patience_epochs']:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(validation_split[0])
        accuracy = ((logits > 0) == validation_split[1].bool()).float().mean().item()
    return model, {'seed': seed, 'best_epoch': best_epoch, 'epochs_run': epoch,
                   'validation_loss': best_loss, 'validation_accuracy': accuracy,
                   'seconds': time.monotonic() - start}
