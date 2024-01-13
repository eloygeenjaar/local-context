import os
import ray
import sys
import torch
import shutil
import importlib
import numpy as np
import lightning.pytorch as pl
from ray import train, tune
from lib.data import DataModule
from ray.tune import CLIReporter
from ray.train.torch import TorchConfig, TorchTrainer
from ray.train import Checkpoint, FailureConfig
#from lightning.pytorch.callbacks import Callback
from ray.train.lightning import RayTrainReportCallback
from torch.utils.data import DataLoader
#from ray.tune.schedulers.pb2 import PB2
from tempfile import TemporaryDirectory
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from lightning.pytorch.callbacks import ModelCheckpoint
from lib.utils import (
    get_default_config, generate_version_name,
    get_search_space, get_start_config)
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ray.train import lightning, ScalingConfig, CheckpointConfig
from ray.tune.search import Repeater, optuna
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter


class MyRayTrainReportCallback(RayTrainReportCallback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        print(trainer.current_epoch)
        should_checkpoint = (trainer.current_epoch + 1) % 10 == 0
        
        # Creates a checkpoint dir with fixed name
        tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
        os.makedirs(tmpdir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        if should_checkpoint:
            # Save checkpoint to local
            ckpt_path = os.path.join(tmpdir, self.CHECKPOINT_NAME)
            trainer.save_checkpoint(ckpt_path, weights_only=False)

            # Report to train session
            checkpoint = Checkpoint.from_directory(tmpdir)
            train.report(metrics=metrics, checkpoint=checkpoint)
        else:
            train.report(metrics=metrics)

        # Add a barrier to ensure all workers finished reporting here
        torch.distributed.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)

if __name__ == "__main__":
    test = False
    epochs = 1000
    perturbation_interval = 20 if not test else 10
    torch.backends.cudnn.deterministic = True
    config = get_default_config(sys.argv)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
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
    search_space = get_search_space(config)
    
    viz = False
    def train_tune(search_space):
        torch.set_float32_matmul_precision('medium')
        
        dm = DataModule(config, dataset_type, config["batch_size"])
        # TODO: change config (second) back to search_space
        model = model_type(config, search_space, viz)
        kwargs_tr = {
            "devices": "auto",
            "accelerator": "auto",
            "max_epochs": epochs,
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
        trainer.fit(**kwargs_fit)

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='va_loss',
        mode='min',
        max_t=epochs,
        grace_period=100,
        reduction_factor=3,
        brackets=1,
    )

    algo = OptunaSearch(get_start_config(config), metric="va_loss", mode="min")

    reporter = CLIReporter(
        parameter_columns=search_space["train_loop_config"].keys(),
        metric_columns=["tr_loss", "tr_mse", "tr_acc", "va_loss", "va_mse", "va_acc"],
    )

    ray_trainer = TorchTrainer(
        train_loop_per_worker=train_tune,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True,
                                     resources_per_worker={"CPU": 4, "GPU": 1}),
        run_config=ray.train.RunConfig(
            local_dir='/data/users1/egeenjaar/local-global/ray_results',
            name=version,
            # Stop when we've reached a threshold accuracy, or a maximum
            # training_iteration, whichever comes first
            stop={"training_iteration": epochs},
            checkpoint_config=CheckpointConfig(num_to_keep=1,
                checkpoint_score_attribute="va_loss",
                checkpoint_score_order="min"
            ),
            progress_reporter=reporter,
            failure_config=FailureConfig(max_failures=2)
       ))
    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            #scheduler=scheduler,
            num_samples=100 if not test else 2,
            search_alg=algo,
            scheduler=scheduler
        )
    )
    results_grid = tuner.fit()
