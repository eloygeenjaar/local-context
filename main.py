import os
import ray
import sys
import torch
import importlib
import numpy as np
import lightning.pytorch as pl
from ray import train, tune
from lib.data import DataModule
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayTrainReportCallback
from torch.utils.data import DataLoader
from ray.tune.schedulers.pb2 import PB2
from lightning.pytorch.callbacks import ModelCheckpoint
from lib.utils import get_default_config, generate_version_name
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray.train import lightning, ScalingConfig, CheckpointConfig
from ray.tune.search import Repeater, optuna


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    config = get_default_config(sys.argv)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    search_space = {"train_loop_config": {
        "num_layers": tune.choice([1, 2, 3, 4]),
        "spatial_hidden_size": tune.choice([32, 64, 128, 256]),
        "temporal_hidden_size": tune.choice([128, 256, 512]),
        "lr": tune.loguniform(1e-4, 2e-3),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "beta": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.loguniform(1e-5, 1e-3),
        "theta": tune.loguniform(1e-7, 1e-3) if 'Cont' in config['model'] else tune.choice([0])}
    }
    version = generate_version_name(config)
    data_module = importlib.import_module('lib.data')
    dataset_type = getattr(data_module, config['dataset'])
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

    epochs = 750
    viz = False
    def train_tune(search_space):
        torch.set_float32_matmul_precision('medium')
        dm = DataModule(config, dataset_type, config["batch_size"])
        model = model_type(config, search_space, viz)
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping, RayTrainReportCallback()],
            strategy=ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
            plugins=[ray.train.lightning.RayLightningEnvironment()]
        )
        trainer = ray.train.lightning.prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    perturbation_interval = 10
    scheduler = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        metric="va_loss",
        mode="min",
        hyperparam_bounds={
            "train_loop_config": {
            "num_layers": [1, 4],
            "spatial_hidden_size": [32, 256],
            "temporal_hidden_size": [128, 512],
            "lr": [1e-4, 2e-3],
            "batch_size": [32, 256],
            "beta": [1e-5, 1e-3],
            "gamma": [1e-5, 1e-3],
            "theta": [1e-7, 1e-3] if 'Cont' in config['model'] else [0, 0]}},
    )

    ray_trainer = TorchTrainer(
        train_tune,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True,
                                     resources_per_worker={"CPU": 5, "GPU": 1}),
        run_config=ray.train.RunConfig(
            local_dir='/data/users1/egeenjaar/local-global/ray_results',
            name=version,
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": 750},
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=4,
                checkpoint_score_attribute="va_loss",
                checkpoint_score_order="min",
        ))
    )
    # Repeat across 5 seeds since training on fMRI data
    # has high variance, so results need to be averaged
    # across seeds
    # TODO: verify if this may be better,
    # but this does require a different scheduler
    #search_alg = optuna.OptunaSearch(metric="va_loss", mode="min")
    #re_search_alg = Repeater(search_alg, repeat=5)
    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            #search_alg = re_search_alg,
            num_samples=8,
        ),
        param_space=search_space,
    )

    results_grid = tuner.fit()
