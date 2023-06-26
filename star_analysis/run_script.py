from star_analysis.runner.run import Run, TrainerConfig
from star_analysis.runner.sdss_runner import SdssRunner, SdssRunConfig, SdssModelConfig
from star_analysis.model.types import ModelTypes
from star_analysis.model.neural_networks.losses.types import LossType
from star_analysis.data.augmentations import Augmentations


def execute():
    runner = SdssRunner()

    run = Run(
        SdssRunConfig(
            model_config=SdssModelConfig(
                learning_rate=1e-3,
                batch_size=80,
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
            logger=runner.logger,
            max_epochs=1,
            limit_train_batches=10,
            limit_val_batches=10,
            log_every_n_steps=1
        )
    )
    runner.save_model(run)

    result_dict = runner.test(
        run=run,
        trainer_config=None,
    )
    print(result_dict)


if __name__ == '__main__':
    execute()
