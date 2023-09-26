import os
import torch
import importlib
import numpy as np
from torch import optim, nn
from lib.utils import get_default_config
from lib.data import DiscreteData
from lib.model import (VAE, iVAE, HiVAE)
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
    version = f'{config["model_name"]}'
    train_dataset = SimulatedData('train', num_samples=config["num_samples"],
                                 num_segments=config["num_segments"], latent_dim=config["data_latent_dim"],
                                 input_size=config["input_size"], num_data_layers=config["num_data_layers"],
                                 seed=config["seed"], prior=config["prior"], data_act=config["data_act"],
                                 uncentered=config["uncentered"], noisy=config["noisy"])
    valid_dataset = DiscreteData('valid', num_samples=config["num_samples"],
                                 num_segments=config["num_segments"], latent_dim=config["data_latent_dim"],
                                 input_size=config["input_size"], num_data_layers=config["num_data_layers"],
                                 seed=config["seed"], prior=config["prior"], data_act=config["data_act"],
                                 uncentered=config["uncentered"], noisy=config["noisy"])
    size_dataset = len(train_dataset)
    #model_type = importlib.import_module(f'.lib.model.{config["model_name"]}')
    #print(model_type)
    model = HiVAE(input_size=config["input_size"], latent_dim=config["latent_dim"],
                 aux_dim=config["num_segments"], size_dataset=size_dataset, num_layers=config["num_layers"],
                 activation=config["activation"], hidden_dim=config["hidden_dim"],
                 hyperparam_tuple=(config["a"], config["b"], config["c"], config["d"]))
    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), version=version, name="lightning_logs")
    csv_logger = CSVLogger(save_dir=os.getcwd(), version=version, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(filename="best", save_last=False, monitor="va_elbo")
    early_stopping = EarlyStopping(monitor="va_elbo", patience=50, mode="min")
    trainer = pl.Trainer(max_epochs=20, logger=[tb_logger, csv_logger],
                         callbacks=[checkpoint_callback, early_stopping])
    train_loader = DataLoader(train_dataset, num_workers=5, pin_memory=True,
                              batch_size=config["batch_size"], shuffle=True,
                              persistent_workers=True, prefetch_factor=5, drop_last=True)
    valid_loader = DataLoader(valid_dataset, num_workers=5, pin_memory=True,
                              batch_size=config["batch_size"], shuffle=False,
                              persistent_workers=True, prefetch_factor=5, drop_last=True)
    trainer.fit(model, train_loader, valid_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        X, U = train_dataset.X.to(device), train_dataset.U.device
        f, g, v, z, l = model(train_dataset.X, train_dataset.U)

    print(mcc(train_dataset.S.numpy(), z.cpu().detach().numpy()))