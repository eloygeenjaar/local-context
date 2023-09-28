import os
import torch
import importlib
import numpy as np
from torch import optim, nn
from lib.utils import get_default_config
from lib.data import Simulation, AirQuality
from lib.model import GLR
from lib.utils import mean_corr_coef as mcc
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    config = get_default_config([''])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    version = f'{config["model_name"]}_mae'
    train_dataset = AirQuality('train', config["seed"])
    valid_dataset = AirQuality('valid', config["seed"])
    model = GLR(config["input_size"], config["time_length"], config["local_size"], config["global_size"],
                window_size=config["window_size"], kernel=list(config["kernel"]), beta=config["beta"],
                 length_scale=config["length_scale"], kernel_scales=config["kernel_scales"])
    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), version=version, name="lightning_logs")
    csv_logger = CSVLogger(save_dir=os.getcwd(), version=version, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(filename="best", save_last=False, monitor="va_elbo")
    early_stopping = EarlyStopping(monitor="va_elbo", patience=10, mode="min")
    trainer = pl.Trainer(max_epochs=100, logger=[tb_logger, csv_logger],
                         callbacks=[checkpoint_callback, early_stopping])
    train_loader = DataLoader(train_dataset, num_workers=5, pin_memory=True,
                              batch_size=config["batch_size"], shuffle=True,
                              persistent_workers=True, prefetch_factor=5, drop_last=True)
    valid_loader = DataLoader(valid_dataset, num_workers=5, pin_memory=True,
                              batch_size=config["batch_size"], shuffle=False,
                              persistent_workers=True, prefetch_factor=5, drop_last=True)
    trainer.fit(model, train_loader, valid_loader)
