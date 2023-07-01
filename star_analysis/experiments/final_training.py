from star_analysis.runner.run import Run, TrainerConfig
from star_analysis.runner.sdss_runner import SdssRunner, SdssRunConfig, SdssModelConfig
from star_analysis.model.types import ModelTypes
from star_analysis.model.neural_networks.losses.types import LossType
from star_analysis.data.augmentations import Augmentations


runner = SdssRunner(project_name="sdss-final")

run = Run(
    SdssRunConfig(
        model_config=SdssModelConfig(
            learning_rate=0.0005,
            batch_size=90,
            model_type=ModelTypes.UNET,
            loss_type=LossType.DA_DICE
        ),
        augmentation=Augmentations.NONE,
        shuffle_train=True
    )
)
runner.add_run(run)

runner.train(
    run=run,
    trainer_config=TrainerConfig(
        logger=runner.logger,
        max_epochs=300,
        limit_train_batches=None,
        limit_val_batches=None,
        log_every_n_steps=50
    )
)
runner.save_model(run)

result_dict = runner.test(
    run=run,
    trainer_config=None,
)[0]
print(result_dict)
