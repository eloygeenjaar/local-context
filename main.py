import os
import ray
import sys
import torch
import importlib
import numpy as np
import lightning.pytorch as pl
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
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
    search_space = {
        "num_layers": tune.choice([1, 2, 3, 4]),
        "spatial_hidden_size": tune.choice([32, 64, 128, 256]),
        "temporal_hidden_size": tune.choice([128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "beta": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.loguniform(1e-5, 1e-3),
        "theta": tune.loguniform(1e-5, 1e-3)
    }
    version = generate_version_name(config)
    data_module = importlib.import_module('lib.data')
    dataset_type = getattr(data_module, config['dataset'])
    train_dataset = dataset_type(
        'train', config['seed'], config['window_size'], config['window_step'])
    valid_dataset = dataset_type(
        'valid', config['seed'], config['window_size'], config['window_step'])
    config['data_size'] = train_dataset.data_size
    config['input_size'] = train_dataset.data_size
    model_module = importlib.import_module('lib.model')
    model_type = getattr(model_module, config['model'])
    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=version, name="lightning_logs")
    csv_logger = CSVLogger(
        save_dir=os.getcwd(), version=version, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(
        filename="best", save_last=False, monitor="va_loss")
    early_stopping = EarlyStopping(
        monitor="va_loss", patience=50, mode="min")
    train_loader = DataLoader(
        train_dataset, num_workers=5, pin_memory=True,
        batch_size=config["batch_size"], shuffle=True,
        persistent_workers=True, prefetch_factor=5, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, num_workers=5, pin_memory=True,
        batch_size=2048, shuffle=False,
        persistent_workers=True, prefetch_factor=5, drop_last=False)

    def train_tune(search_space, epochs=750):
        model = model_type(config, search_space)
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=[tb_logger, csv_logger])
        trainer.fit(model, train_loader, valid_loader)

    # If testing
    smoke_test = True
    perturbation_interval = 5
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        metric="val_loss",
        mode="min",
        hyperparam_mutations=search_space,
    )

    tuner = tune.Tuner(
        train_tune,
        run_config=train.RunConfig(
            name=version,
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": 750},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute="mean_accuracy",
                num_to_keep=4,
            ),
            local_dir='/Users/egeenjaar/OneDrive - Georgia Institute of Technology/local-global/ray_results'
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=4,
        ),
        param_space=search_space,
    )

    results_grid = tuner.fit()
