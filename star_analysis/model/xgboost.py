from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.model.base_model import BaseModel
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class XGBTrainingStatistics:
    accuracy: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    time: Optional[timedelta] = None


class XGBModel(BaseModel):
    def __init__(self):
        self.__classifier = GradientBoostingClassifier()
        self.__dataprovider = SDSSDataProvider(
            ImageDownloader('/Users/leopinetzki/data')
        )
        self.__dataprovider.prepare(SDSSDataProvider.FIXED_VALIDATION_RUN)

    def __load_data_parallel(self, ids: list[int]):
        with ProcessPoolExecutor(max_workers=4) as handler:
            futures = handler.map(self.__dataprovider.__getitem__, ids)

        return [i for i in futures]

    def train(self):
        start_time = datetime.now()

        logger.info("loading data")

        train_data = self.__load_data_parallel(list(range(1)))
        test_data = self.__load_data_parallel(list(range(101, 102)))

        train_stack = np.concatenate(
            [i[0].reshape(-1, train_data[0][0].shape[-1]) for i in train_data])
        label_stack = np.concatenate(
            [i[1].reshape(-1, train_data[0][1].shape[-1]) for i in train_data])
        indices = np.random.choice(range(len(train_stack)), size=int(len(
            train_stack) * 0.1), replace=False)
        train_stack = train_stack[indices]
        label_stack = label_stack[indices]

        label_stack = self.__add_label_dim(label_stack)

        logger.info("training")

        self.__classifier.fit(
            train_stack, label_stack.nonzero()[1])

        test_data_stack = np.concatenate(
            [i[0].reshape(-1, train_data[0][0].shape[-1]) for i in test_data])
        test_label_stack = np.concatenate(
            [i[1].reshape(-1, train_data[0][1].shape[-1]) for i in test_data])
        test_label_stack = self.__add_label_dim(test_label_stack)

        logger.info("evaluating")

        predictions = self.__classifier.predict(test_data_stack)

        return self.__evaluate(predictions, test_label_stack, start_time)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return super().predict(data)

    def __add_label_dim(self, labels: np.ndarray) -> np.ndarray:
        no_class = np.zeros((labels.shape[0], 1))
        mask = (~labels.astype(bool)).all(axis=-1)
        no_class[mask, :] = 1
        label_stack = np.concatenate((labels, no_class), axis=-1)
        label_stack[label_stack[:, 0] == 1, 1] = 0
        label_stack[label_stack[:, 0] == 1, 2] = 0

        return label_stack

    def __evaluate(self, predictions: np.ndarray,
                   labels: np.ndarray, start_time: datetime):

        labels_dense = labels.nonzero()[1]
        statistics = XGBTrainingStatistics(accuracy=accuracy_score(labels_dense, predictions), precision=precision_score(
            labels_dense, predictions, average='macro'), recall=recall_score(labels_dense, predictions, average='macro'), time=datetime.now() - start_time)

        return statistics


if __name__ == "__main__":
    print(XGBModel().train())
