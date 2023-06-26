import copy
import os
from dataclasses import replace
from enum import auto
from typing import Iterable, Callable

import optuna
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import Trial, Study
from strenum import StrEnum
from torch.utils.data import DataLoader

from star_analysis.model.types import ModelTypes
from star_analysis.runner.run import Run, TrainerConfig, OptunaTuneTrainerConfig, RunConfig
from star_analysis.utils.constants import DATAFILES_ROOT, LOGGING_DIR, MODEL_DIR


class TuningModes(StrEnum):
    ITERATIVE = auto()
    PARALLEL = auto()


class LoadingModes(StrEnum):
    CHECKPOINT = auto()
    PT_MODEL = auto()


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

    def prebuild_run(self, run: Run):
        return run.prebuild(
            data_dir=self.data_dir,
            num_workers=self.num_workers
        )

    def rebuild_run(self, run: Run, trainer_config: TrainerConfig):
        return run.rebuild(
            data_dir=self.data_dir,
            num_workers=self.num_workers,
            trainer_config=trainer_config
        )

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
        if trainer_config is None:
            trainer_config = TrainerConfig(
                logger=self.logger
            )

        run = self._check_for_simple_run(run, trainer_config)
        run.fit()

    def tune(
            self,
            mode: TuningModes = TuningModes.ITERATIVE,
            trainer_config: TrainerConfig | OptunaTuneTrainerConfig | None = None,
            runs: Iterable[Run] | None = None,
            optuna_objective: Callable = None
    ) -> list[dict[str, float]] | Study:
        if trainer_config is None:
            trainer_config = TrainerConfig(
                logger=self.logger,
                limit_train_batches=None,
                limit_val_batches=None,
                max_epochs=10
            )

        if mode == TuningModes.ITERATIVE:
            return self._iterative_tune(
                runs=runs,
                trainer_config=trainer_config,
            )
        elif mode == TuningModes.PARALLEL:
            return self._optuna_tune(
                objective=optuna_objective,
                trainer_config=trainer_config,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

    def test(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None
    ) -> list[dict[str, float]]:
        """Returns a list of dicts containing the metrics for each test run."""
        if trainer_config is None:
            trainer_config = TrainerConfig(
                logger=self.logger,
                limit_train_batches=None,
                limit_val_batches=None,
                max_epochs=1,
                devices=1,
                log_every_n_steps=1
            )

        run = self._check_for_test_run(run, trainer_config)

        return run.trainer.test(
            model=run.model,
            datamodule=run.data_module,
            ckpt_path=None,  # "best"
        )

    def predict(
            self,
            run: Run | None = None,
            trainer_config: TrainerConfig | None = None,
            data_loader: DataLoader | None = None
    ) -> list[torch.Tensor]:
        run = self._check_for_test_run(run, trainer_config)

        if data_loader is None:
            datamodule = run.data_module
        else:
            datamodule = None

        return run.trainer.predict(
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
        else:
            raise ValueError("Model not trained yet. Please train it first.")

    def load_model(
            self, mode: LoadingModes = LoadingModes.PT_MODEL,
            filename: str = None,
            path: str = None
    ) -> LightningModule:
        print("Loading is not yet integrated.")
        if filename:
            checkpoint = torch.load(os.path.join(MODEL_DIR, filename))
        elif path:
            checkpoint = torch.load(path)
        else:
            raise ValueError(f"Either name or path must be specified")

        # TODO create run for loaded model
        model = None
        if mode == 'ckpt':
            model.load_state_dict(checkpoint['model'])
            model.optimizer.load_state_dict(checkpoint['optimizer'])
        elif mode == 'model':
            model = checkpoint

        return model

    def close(self):
        raise NotImplementedError(f"Close not implemented yet")

    def _setup_logger(self):
        return TensorBoardLogger(name=self.project_name, save_dir=LOGGING_DIR)

    def _check_for_simple_run(
            self,
            run: Run | None,
            trainer_config: TrainerConfig | None
    ) -> Run:
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

    def _iterative_tune(self, runs: Iterable[Run] | None, trainer_config: TrainerConfig) -> list[dict[str, float]]:
        if runs is None:
            run = self.get_last_valid_run()
            if run is None:
                raise ValueError("No run found in runner. Please add one.")
            runs = [run]

        for run in runs:
            if not run.built:
                run.build(
                    data_dir=self.data_dir,
                    num_workers=self.num_workers,
                    trainer_config=trainer_config
                )
            else:
                print(f"Run {run.name} already built. Reusing existing build.")

        metrics = []
        for run in runs:
            self.train(
                run=run,
                trainer_config=trainer_config
            )
            metrics.append(
                self.test(
                    run=run,
                    trainer_config=None
                )[0]
            )
        return metrics

    def _optuna_tune(self, objective: Callable, trainer_config: OptunaTuneTrainerConfig) -> Study:
        study = optuna.create_study()
        study.optimize(
            objective,
            n_trials=trainer_config.num_trials,
            timeout=trainer_config.timeout,
            n_jobs=trainer_config.num_jobs,
            show_progress_bar=trainer_config.show_progress_bar,
            gc_after_trial=trainer_config.gc_after_trial,
        )
        return study

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
