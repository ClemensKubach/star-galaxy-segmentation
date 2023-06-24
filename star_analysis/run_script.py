from star_analysis.runner.run import Run, TrainerConfig
from star_analysis.runner.sdss_runner import SdssRunner, SdssRunConfig, SdssModelConfig
from star_analysis.model.types import ModelTypes
from star_analysis.model.neural_networks.losses.types import LossType
from star_analysis.data.augmentations import Augmentations

if __name__ == '__main__':
    runner = SdssRunner()

    run = Run(
        SdssRunConfig(
            model_config=SdssModelConfig(
                learning_rate=1e-4,
                batch_size=32,
                model_type=ModelTypes.UNET,
                loss_type=LossType.DICE
            ),
            augmentation=Augmentations.NONE,
            shuffle_train=True
        )
    )
    runner.add_run(run)

    runner.train(
        run=run,
        trainer_config=TrainerConfig(
            logger=None,
            max_epochs=1,
        )
    )
    runner.save_model(run)
