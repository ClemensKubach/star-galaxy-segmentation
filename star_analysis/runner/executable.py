from abc import abstractmethod, ABC

from lightning import LightningModule, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger

from star_analysis.model.types import ModelTypes
from star_analysis.utils.constants import LOGGING_DIR


class Executable(ABC):
    """
    This can be executed via cli using app.py. This should later contain the best model configurations.
    It should also contain further functions for the long-running training tasks on remote machines.
    """

    def __init__(
            self,
            data_dir: str,
            model_type: ModelTypes,
            project_name: str,
            batch_size: int,
            learning_rate_init: float
    ):
        self.data_dir = data_dir
        self.model_type = model_type
        self.project_name = project_name
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init

        self.data_module = None
        self.model = None

    def init(self):
        self.data_module = self._setup_data()
        self.model = self._setup_model()
        self._setup_logger()

    def _setup_logger(self):
        self.logger = TensorBoardLogger(name=self.project_name, save_dir=LOGGING_DIR)

    def _setup_model(self) -> LightningModule | None:
        match self.model_type:
            case ModelTypes.FCN:
                raise NotImplementedError(f"FCN not implemented yet")
            case ModelTypes.UNET:
                raise NotImplementedError(f"UNET not implemented yet")
            case ModelTypes.CUSTOM:
                return self.model
            case _:
                raise ValueError(f"Unknown model type {self.model_type}")

    def save_model(self):
        raise NotImplementedError(f"Save model not implemented yet")

    def load_model(self):
        raise NotImplementedError(f"Load model not implemented yet")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        raise NotImplementedError(f"Close not implemented yet")

    @abstractmethod
    def _setup_data(self) -> LightningDataModule:
        raise NotImplementedError(f"Data module must be implemented")

    def set_custom_model(self, model: LightningModule):
        self.model = model

    @abstractmethod
    def train(self):
        raise NotImplementedError(f"Train not implemented")

    @abstractmethod
    def eval(self):
        raise NotImplementedError(f"Eval not implemented")

    @abstractmethod
    def predict(self):
        raise NotImplementedError(f"Predict not implemented")
