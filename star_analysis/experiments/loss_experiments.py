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

runner = SdssRunner(project_name="sdss-losses")

run_config0 = SdssRunConfig(
    model_config=SdssModelConfig(
        learning_rate=1e-4,
        batch_size=80,
        model_type=ModelTypes.UNET,
        loss_type=LossType.MSE
    ),
    augmentation=Augmentations.NONE,
    shuffle_train=True
)

run_config1 = copy.deepcopy(run_config0)
run_config1.model_config.loss_type = LossType.DICE

run_config2 = copy.deepcopy(run_config0)
run_config2.model_config.loss_type = LossType.DA_DICE

run_config3 = copy.deepcopy(run_config0)
run_config3.model_config.loss_type = LossType.FOCAL

run_config4 = copy.deepcopy(run_config0)
run_config4.model_config.loss_type = LossType.DA_FOCAL

configs = [run_config0, run_config1, run_config2, run_config3, run_config4]
runs = [Run(config) for config in configs]

for run in runs:
    runner.add_run(run)

from star_analysis.runner.runner import TuningModes
from star_analysis.runner.run import TrainerConfig

results = runner.tune(
    mode=TuningModes.ITERATIVE,
    runs=runs,
    trainer_config=TrainerConfig(
        logger=runner.logger,
        max_epochs=10,
    )
)
run_results = zip(runs, results)

print("Results:", results)

best_run, result_best = max(run_results, key=lambda x: x[1][f'{x[0].id}/test_f1'])
print(f"Best run, {best_run.name}, achieved {best_run[f'{best_run.id}/test_f1']} test_f1")

for i, (run, result) in enumerate(run_results):
    print(f"Run {run.id}, {run.name}, achieved {result[f'{run.id}/test_f1']} test_f1")

runner.save_model(best_run.model)

print(best_run.config.model_config)
