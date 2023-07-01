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
from star_analysis.runner.runner import TuningModes
from star_analysis.runner.run import TrainerConfig

runner = SdssRunner(project_name="sdss-augs")

run_config0 = SdssRunConfig(
    model_config=SdssModelConfig(
        learning_rate=1e-4,
        batch_size=80,
        model_type=ModelTypes.UNET,
        loss_type=LossType.DICE
    ),
    augmentation=Augmentations.NONE,
    shuffle_train=True
)

run_config1 = copy.deepcopy(run_config0)
run_config1.augmentation = Augmentations.FLIP

run_config2 = copy.deepcopy(run_config0)
run_config2.augmentation = Augmentations.ROTATE

run_config3 = copy.deepcopy(run_config0)
run_config3.augmentation = Augmentations.ROTATE_FLIP

run_config4 = copy.deepcopy(run_config0)
run_config4.augmentation = Augmentations.BALANCE_CLASSES

configs = [run_config0, run_config1, run_config2, run_config3, run_config4]
runs = [Run(config) for config in configs]

for run in runs:
    runner.add_run(run)

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

best_run, result_best = max(run_results, key=lambda x: x[1][f'{best_run.id}/test_f1'])
print(f"Best run, {best_run.name}, achieved {result_best[f'{best_run.id}/test_f1']} test_f1")

for i, (run, result) in enumerate(run_results):
    print(f"Run {i}, {run.name}, achieved {result[f'{i}/test_f1']} test_f1")

runner.save_model(best_run.model)