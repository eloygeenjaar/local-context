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
from lib.utils import (
    get_default_config, generate_version_name,
    get_search_space, get_hyperparam_bounds)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray.train import lightning, ScalingConfig, CheckpointConfig
from ray.tune.search import Repeater, optuna
import random

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


if __name__ == "__main__":

    
    storage_path = '/data/users1/dkim195/local-global/ray_results'
    # exp_name = 'mLVAE_dICAfBIRN_s6767_s2_g0'

    experiments_list = get_immediate_subdirectories(storage_path)
    count = 0
    countfail = 0
    for exp_name in experiments_list:
        experiment_path = os.path.join(storage_path, exp_name)
        
        try:
            results = ray.tune.ExperimentAnalysis(experiment_path, default_metric = 'va_mse', default_mode = "min")
            with open("ray_result_analysis_check_ckpt.txt", "a") as f:
                print(f"Top 3 Configs from {experiment_path}...", file = f)
                
                for _ in range(3):
                    print("va_loss: ", results.best_result['va_loss'], "; epoch: ", results.best_result['epoch'], "; ", results.best_config, file = f)
                    print(results.best_checkpoint)
                    bestTrial = results.best_trial
                    results.trials.remove(bestTrial)
            count += 1
        except Exception:
            print("")
            countfail += 1

    print(count)
    print(countfail)