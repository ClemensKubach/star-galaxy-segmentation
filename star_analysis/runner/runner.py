import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Callable

import optuna
import torch
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from optuna import Trial
from torch.utils.data import DataLoader

from star_analysis.runner.run import Run, RunConfig, TrainerConfig
from star_analysis.utils.callbacks import PlottingCallback
from star_analysis.utils.constants import CHECKPOINT_DIR, DATAFILES_ROOT, LOGGING_DIR, MODEL_DIR




class Runner:

    def __init__(
            self,
            data_dir: str = DATAFILES_ROOT,
            project_name: str = "test-project",
            num_workers: int = os.cpu_count(),
    ):
        self.__data_dir = data_dir
        self.__project_name = project_name

        self.runs: dict[int, Run | None] = {}
        self.logger = self._setup_logger()

        self.__num_workers = num_workers

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def project_name(self):
        return self.__project_name

    @property
    def num_workers(self):
        return self.__num_workers

    def add_run(self, run: Run) -> int:
        run_id = len(self.runs)
        self.runs[run_id] = run
        return run_id

    def del_run(self, run_id: int):
        self.runs[run_id] = None

    def get_last_valid_run(self) -> Run | None:
        last_valid_run_id = len(self.runs)
        run = None
        while run is None:
            last_valid_run_id -= 1
            if last_valid_run_id < 0:
                return None
            run = self.runs[last_valid_run_id]
        return run

    def train(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None
    ):
        run = self._check_for_test_run(run, trainer_config)

        run.config.trainer.fit(
            model=run.model,
            datamodule=run.data_module
        )

    def tune(
            self,
            tuning_runs: Iterable[Run] | None = None,
            tuning_trainer_config: TrainerConfig | None = None,
            mode: str = 'iterative',
    ):
        if tuning_runs is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")
            tuning_runs = [run]

        for run in tuning_runs:
            if not run.built:
                run.build(
                    data_dir=self.data_dir,
                    num_workers=self.num_workers,
                    trainer_config=tuning_trainer_config
                )
            else:
                print(f"Run {run.name} already built. Reusing existing build.")

        if mode == 'iterative':
            for run in tuning_runs:
                self.train(
                    run=run,
                    trainer_config=tuning_trainer_config
                )
        else:
            raise NotImplementedError("Dynamically searching parameter space is not yet implemented")
            study = optuna.create_study()
            study.optimize(
                self._get_tuning_objective(tune_runs, config),
                n_trials=config.n_trials,
                timeout=config.timeout,
                n_jobs=config.n_jobs,
                show_progress_bar=config.show_progress_bar
            )
            return study

    def test(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None
    ):
        run = self._check_for_test_run(run, trainer_config)

        run.trainer.test(
            model=run.model,
            datamodule=run.data_module,
            ckpt_path="best"
        )

    def predict(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None,
            data_loader: DataLoader | None = None
    ):
        run = self._check_for_test_run(run, trainer_config)

        if data_loader is None:
            datamodule = run.data_module
        else:
            datamodule = None

        run.trainer.predict(
            model=run.model,
            dataloaders=data_loader,
            datamodule=datamodule
        )

    def save_model(self, run: Run | None = None):
        if run is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if run.trained:
            torch.save(run.model, os.path.join(MODEL_DIR, f'model-{run.name}.pt'))

    # TODO
    def load_model(self, name: str = None, path: str = None, mode: str = 'model') -> LightningModule:
        raise NotImplementedError("Loading models is not yet implemented")
        if name:
            checkpoint = torch.load(os.path.join(MODEL_DIR, name))
        elif path:
            checkpoint = torch.load(path)
        else:
            raise ValueError(f"Either name or path must be specified")

        if mode == 'ckpt':
            self.model_module.load_state_dict(checkpoint['model'])
            self.model_module.optimizer.load_state_dict(checkpoint['optimizer'])
        elif mode == 'model':
            self.model_module = checkpoint
        return self.model_module

    def close(self):
        raise NotImplementedError(f"Close not implemented yet")

    def _setup_logger(self):
        return TensorBoardLogger(name=self.project_name, save_dir=LOGGING_DIR)

    def _check_for_simple_run(
            self,
            run: Run | None,
            trainer_config: TrainerConfig | None
    )-> Run:
        if run is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")

        if not run.built:
            run.build(
                data_dir=self.data_dir,
                num_workers=self.num_workers,
                trainer_config=trainer_config
            )
        else:
            print("Run already built. Reusing existing build.")
        return run

    def _check_for_test_run(
            self,
            run: Run | None,
            trainer_config: TrainerConfig | None
    ) -> Run:
        if run is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")

        if not run.trained:
            raise ValueError("Run has not been trained yet. Please train first.")

        if not run.built:
            run.build(
                data_dir=self.data_dir,
                num_workers=self.num_workers,
                trainer_config=trainer_config
            )
        else:
            print("Run already built. Reusing existing build.")
        return run

    def _get_tuning_objective(self, runs: Iterable[Run] | None, config: TrainerConfig):
        # TODO implement this using optuna
        def objective(trial: Trial):
            self.train(
                run=None,
                config=config
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
