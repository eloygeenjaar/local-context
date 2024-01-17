import os
import ray
import sys
import time
import json
import torch
import random
import importlib
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from ray import train, tune
from lib.data import DataModule
from ray.tune import CLIReporter
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import (
    FailureConfig, ScalingConfig, CheckpointConfig)
from ray.train.lightning import RayTrainReportCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lib.utils import (
    get_default_config, generate_version_name,
    get_hyperparameters)


if __name__ == "__main__":
    test = False
    viz = False  # Visualize during training
    max_epochs = 750 if not test else 100
    scheduler_epochs = 250 if not test else 50
    torch.backends.cudnn.deterministic = True
    config = get_default_config(sys.argv)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    version = generate_version_name(config)
    data_module = importlib.import_module('lib.data')
    dataset_type = getattr(data_module, config['dataset'])
    model_module = importlib.import_module('lib.model')
    model_type = getattr(model_module, config['model'])
    early_stopping = EarlyStopping(
        monitor="va_loss", mode="min", patience=50)

    def train_tune(hyperparameters):
        torch.set_float32_matmul_precision('medium')
        dm = DataModule(config, dataset_type, config["batch_size"])
        model = model_type(config, hyperparameters, viz)
        model = ray.train.torch.prepare_model(model)
        kwargs_tr = {
            "devices": "auto",
            "accelerator": "auto",
            "callbacks": [RayTrainReportCallback()],
            "strategy": ray.train.lightning.RayDDPStrategy(find_unused_parameters=True),
            "plugins": [ray.train.lightning.RayLightningEnvironment()],
            "enable_progress_bar": False,
            "enable_checkpointing": False
        }
        trainer = pl.Trainer(**kwargs_tr)
        
        kwargs_fit = {
            "model": model,
            "datamodule": dm,
        }
        trainer = ray.train.lightning.prepare_trainer(trainer)

        # Resume from checkpoint if checkpoint exists
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as ckpt_dir:
                ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
                kwargs_fit['ckpt_path'] = ckpt_path

        trainer.fit(**kwargs_fit)

    # Initialize the scheduler
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='va_loss',
        mode='min',
        max_t=scheduler_epochs,
        grace_period=50 if not test else 1,
        reduction_factor=2,
        brackets=1,
    )

    # Get the hyperparameter ranges etc.
    hyperparameters = get_hyperparameters(config)

    # Initialize the algorithm with which we select future hyperparameters
    # to run the model with
    algo = OptunaSearch(metric="va_loss", mode="min")

    # A reporter that makes it easy to keep track of the status for each of
    # the models
    reporter = CLIReporter(
        parameter_columns=hyperparameters["train_loop_config"].keys(),
        metric_columns=["tr_loss", "tr_mse", "tr_acc", "va_loss", "va_mse", "va_acc"],
    )

    # Initialize the TorchTrainer with the above training function
    ray_trainer = TorchTrainer(
        train_loop_per_worker=train_tune,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True,
                                     resources_per_worker={"CPU": 4, "GPU": 1}),
        run_config=ray.train.RunConfig(
            local_dir='/data/users1/egeenjaar/local-global/ray_results',
            name=version,
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": scheduler_epochs},
            checkpoint_config=CheckpointConfig(num_to_keep=1,
                checkpoint_score_attribute="va_loss",
                checkpoint_score_order="min"
            ),
            progress_reporter=reporter,
            failure_config=FailureConfig(max_failures=2)
       ))
    
    # The tuner will try to find the best hyperparameter configuration
    # using the search algorithm and the scheduler, which decides
    # whether a model with certain hyperparameters is better than 
    # models with other hyperparameters. Try 100 different
    # hyperparameter combinations
    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            num_samples=100 if not test else 2,
            search_alg=algo,
            scheduler=scheduler
        ),
        param_space=hyperparameters
    )
    result_grid = tuner.fit()

    # Wait for workers to be done
    time.sleep(5)

    # Get the best result from the tuner
    best_result = result_grid.get_best_result( 
        metric="va_loss", mode="min")
    # Get the checkpoint at which this model is saved
    best_checkpoint = best_result.checkpoint

    # Get the hyperparameters of the model
    params_p = Path(best_checkpoint.path).parent / 'params.json'
    with params_p.open('r') as f:
        params = json.load(f)
    
    # Delete all the worse models' checkpoints to save disk space
    for result in result_grid:
        if not result == best_result:
            checkpoint = Path(result.checkpoint.path) / 'checkpoint.ckpt'
            checkpoint.unlink()

    # Initialize another trainer instance to finish training the best
    # model
    final_trainer = TorchTrainer(
        train_loop_per_worker=train_tune,
        train_loop_config=params['train_loop_config'],
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True,
                                     resources_per_worker={"CPU": 4, "GPU": 1}),
        run_config=ray.train.RunConfig(
            local_dir='/data/users1/egeenjaar/local-global/ray_results',
            name=version,
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": max_epochs},
            checkpoint_config=CheckpointConfig(num_to_keep=1,
                checkpoint_score_attribute="va_loss",
                checkpoint_score_order="min"),
            progress_reporter=reporter,
    ),
        resume_from_checkpoint=best_result.checkpoint
    )
    result = final_trainer.fit()

    # Wait for workers to be done
    time.sleep(5)
    best_checkpoint = result.best_checkpoints[0][0]

    # Move all the final checkpoints to the parent folder
    # for easy access after training
    original_cp = Path(best_checkpoint.path)
    c_p = original_cp / 'checkpoint.ckpt'
    nc_p = original_cp.parent.parent / 'final.ckpt'
    c_p.replace(nc_p)
    r_p = original_cp.parent / 'result.json'
    nr_p = original_cp.parent.parent / 'result.json'
    r_p.replace(nr_p)
    p_p = original_cp.parent / 'params.json'
    np_p = original_cp.parent.parent / 'params.json'
    p_p.replace(np_p)
