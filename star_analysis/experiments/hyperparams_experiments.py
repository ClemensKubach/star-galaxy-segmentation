import copy
import os
os.chdir("/home/kubach/project_sync/star_analysis")
print(os.getcwd())

from star_analysis.model.types import ModelTypes
from star_analysis.runner.sdss_runner import SdssRunner
from star_analysis.data.augmentations import Augmentations
from star_analysis.runner.sdss_runner import SdssRunConfig, SdssModelConfig
from star_analysis.model.neural_networks.losses.types import LossType
from star_analysis.runner.run import Run

runner = SdssRunner(project_name="sdss-hparams")

from functools import partial
import copy
from star_analysis.runner.run import OptunaTuneTrainerConfig
from optuna import Trial


def objective(trial: Trial, runner: SdssRunner, default_run_config: SdssRunConfig, trainer_config: OptunaTuneTrainerConfig):
    lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    arch = trial.suggest_categorical("architecture", [ModelTypes.UNET, ModelTypes.DLV3])

    run_config = copy.deepcopy(default_run_config)
    if arch == ModelTypes.UNET:
        run_config.model_type = ModelTypes.UNET
    elif arch == ModelTypes.DLV3:
        run_config.model_type = ModelTypes.DLV3
    run_config.lr = lr

    run = Run(run_config)
    runner.add_run(run)
    runner.train(
        run=run,
        trainer_config=trainer_config
    )
    result = runner.test(
        run=run,
        trainer_config=None
    )[0]
    return result[f'{run.id}/test_f1']


default_run_config = SdssRunConfig(
    model_config=SdssModelConfig(
        learning_rate=1e-4,
        batch_size=32,
        model_type=ModelTypes.UNET,
        loss_type=LossType.DICE
    ),
    augmentation=Augmentations.NONE,
    shuffle_train=True
)
trainer_config = OptunaTuneTrainerConfig(
    logger=runner.logger,
    max_epochs=15,
    timeout=60 * 60 * 3,
    direction="maximize",
    num_jobs=1
)

tuning_objective = partial(
    objective,
    runner=runner,
    default_run_config=default_run_config,
    trainer_config=trainer_config
)

from star_analysis.runner.runner import TuningModes

study = runner.tune(
    mode=TuningModes.PARALLEL,
    trainer_config=trainer_config,
    optuna_objective=tuning_objective,
)

print("Best params:", study.best_params)
