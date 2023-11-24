import os
import sys
import torch
import importlib
import numpy as np
import lightning.pytorch as pl
from lib.utils import get_default_config
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lib.data import Sprite

if __name__ == "__main__":
    torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('medium')
    config = get_default_config(sys.argv)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    version = f'1124_conv_no_windows_m{config["model"]}_d{config["dataset"]}_g{config["gamma"]}_s{config["seed"]}_n{config["normalization"]}_s{config["local_size"]}_g{config["global_size"]}_f{config["fold_ix"]}'
    data_module = importlib.import_module('lib.data')
    dataset_type = getattr(data_module, config['dataset'])
    # The last two arguments are only used for fBIRN
    train_dataset = dataset_type('train', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'])
    valid_dataset = dataset_type('valid', config['normalization'], config["seed"], config['num_folds'], config['fold_ix'])
    window_size, mask_windows, lr, num_timesteps = train_dataset.window_size, train_dataset.mask_windows, train_dataset.learning_rate, train_dataset.num_timesteps
    assert train_dataset.data_size == valid_dataset.data_size
    # config['input_size'] = train_dataset.data_size
    # config['input_size'] = 64 #for no conv
    config['input_size'] = 128 #for conv 
    model_module = importlib.import_module('lib.model')
    model_type = getattr(model_module, config['model'])
    print(config['input_size'])
    print("INPUTT")
    model = model_type(config["input_size"], config["local_size"], config["global_size"], num_timesteps,
                       window_size=window_size, beta=config["beta"], gamma=config["gamma"],
                       mask_windows=mask_windows, lr=lr, seed=config['seed'])
    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), version=version, name="lightning_logs")
    csv_logger = CSVLogger(save_dir=os.getcwd(), version=version, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(filename="best", save_last=False, monitor="va_elbo")
    early_stopping = EarlyStopping(monitor="va_elbo", patience=20, mode="min")
    trainer = pl.Trainer(max_epochs=200, logger=[tb_logger, csv_logger],
                         callbacks=[checkpoint_callback, early_stopping], devices=1)

    train_loader = DataLoader(train_dataset, num_workers=3, pin_memory=True,
                              batch_size=config["batch_size"], shuffle=True,
                              persistent_workers=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset, num_workers=3, pin_memory=True,
                              batch_size=config["batch_size"], shuffle=False,
                              persistent_workers=True, drop_last=False)


    trainer.fit(model, train_loader, valid_loader)
