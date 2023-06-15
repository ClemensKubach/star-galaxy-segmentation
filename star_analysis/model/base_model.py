import numpy as np


class BaseModel:
    def train(self):
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        pass