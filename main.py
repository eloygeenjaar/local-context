import os
import sys
import torch
import importlib
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lib.utils import get_default_config, generate_version_name
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('medium')
    config = get_default_config(sys.argv)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    version = generate_version_name(config)
    data_module = importlib.import_module('lib.data')
    dataset_type = getattr(data_module, config['dataset'])
    train_dataset = dataset_type(
        'train', config['seed'], config['window_size'], config['window_step'])
    valid_dataset = dataset_type(
        'valid', config['seed'], config['window_size'], config['window_step'])
    data_size = train_dataset.data_size
    config['input_size'] = train_dataset.data_size
    model_module = importlib.import_module('lib.model')
    model_type = getattr(model_module, config['model'])
    model = model_type(
        input_size=data_size,
        local_size=config['local_size'],
        context_size=config['context_size'],
        num_layers=config['num_layers'],
        spatial_hidden_size=config['spatial_hidden_size'],
        temporal_hidden_size=config['temporal_hidden_size'],
        beta=config['beta'],
        gamma=config['gamma'],
        theta=config['theta'],
        lr=config['lr'],
        seed=config['seed'])
    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=version, name="lightning_logs")
    csv_logger = CSVLogger(
        save_dir=os.getcwd(), version=version, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(
        filename="best", save_last=False, monitor="va_loss")
    early_stopping = EarlyStopping(
        monitor="va_loss", patience=50, mode="min")
    trainer = pl.Trainer(
        max_epochs=750, logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, early_stopping], devices=1)
    print(len(train_dataset), len(valid_dataset))
    train_loader = DataLoader(
        train_dataset, num_workers=5, pin_memory=True,
        batch_size=config["batch_size"], shuffle=True,
        persistent_workers=True, prefetch_factor=5, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, num_workers=5, pin_memory=True,
        batch_size=2048, shuffle=False,
        persistent_workers=True, prefetch_factor=5, drop_last=False)
    trainer.fit(model, train_loader, valid_loader)
