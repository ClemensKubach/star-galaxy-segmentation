from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Optional

import numpy as np

from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
from star_analysis.model.base_model import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


@dataclass
class SVMTrainingStatistics:
    accuracy: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    time: Optional[timedelta] = None

    @classmethod
    def create_mean_from_list(cls, mean_metrics: list[SVMTrainingStatistics]) -> SVMTrainingStatistics:
        accuracies = np.asarray([metrics.accuracy for metrics in mean_metrics])
        precision = np.asarray([metrics.precision for metrics in mean_metrics])
        recall = np.asarray([metrics.recall for metrics in mean_metrics])

        return cls(accuracy=np.mean(accuracies, axis=0), precision=np.mean(precision, axis=0), recall=np.mean(recall, axis=0))


class SVMModel(BaseModel):
    def __init__(self, threshold: int = 127):
        self.__predictor = SVC()
        self.__dataprovider = SDSSDataProvider(
            ImageDownloader('/Users/leopinetzki/data'))
        self.__threshold = threshold

    def __load_data_parallel(self, ids: list[int]):
        with ProcessPoolExecutor(max_workers=4) as handler:
            futures = handler.map(self.__dataprovider.__getitem__, ids)

        return [i for i in futures]

    def train(self) -> SVMTrainingStatistics:
        start_time = datetime.now()

        logger.info("Loading data for training")

        self.__dataprovider.prepare("1000")
        train_data = self.__load_data_parallel(list(range(10)))
        test_data = self.__load_data_parallel(list(range(101, 104)))

        train_datapoints = self.__select_important_pixels(train_data)
        test_datapoints = self.__select_important_pixels(test_data)

        train_stack = np.concatenate([i[1] for i in train_datapoints])
        label_stack = np.concatenate(
            [i[2] for i in train_datapoints])
        no_class = np.zeros((label_stack.shape[0], 1))
        mask = (~label_stack.astype(bool)).all(axis=-1)
        no_class[mask, :] = 1
        label_stack = np.concatenate((label_stack, no_class), axis=-1)

        logger.info("starting training")

        self.__predictor = self.__predictor.fit(
            train_stack / 256, label_stack.nonzero()[1])

        test_data_stack = np.concatenate([i[1] for i in test_datapoints])

        logger.info("evaluating")

        predictions = self.__predictor.predict(test_data_stack)

        return self.__evaluate(predictions, test_datapoints, test_data, start_time=start_time)

    def __evaluate(self, predictions: np.ndarray,
                   raw_data: list[tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]],
                   test_data: list[np.ndarray], start_time: datetime):
        seen_predictions = 0

        all_metrics = []
        for combined, image in zip(test_data, raw_data):
            label_image = self.__rebuild_label_image(image[0][0], image[0][1],
                                                     predictions[seen_predictions:len(
                                                         image[1]) + seen_predictions],
                                                     (*combined[0].shape[:-1], self.__dataprovider.num_labels + 1))

            no_class = np.zeros((*combined[1].shape[:-1], 1))
            mask = (~combined[1].astype(bool)).all(axis=-1)
            no_class[mask] = 1

            metrics = self.__calculate_metrics(
                label_image, np.concatenate((combined[1], no_class), axis=-1))
            all_metrics.append(metrics)

        statistics = SVMTrainingStatistics.create_mean_from_list(all_metrics)
        statistics.time = datetime.now() - start_time

        return statistics

    def __calculate_metrics(self, reconstructed_image: np.ndarray, true_image: np.ndarray) -> SVMTrainingStatistics:
        assert reconstructed_image.shape == true_image.shape

        true_flat = true_image.flatten()
        reconstructed_flat = reconstructed_image.flatten()
        recall = recall_score(true_flat, reconstructed_flat)
        precision = precision_score(true_flat, reconstructed_flat)
        accuracy = accuracy_score(true_flat, reconstructed_flat)

        return SVMTrainingStatistics(accuracy=accuracy, precision=precision, recall=recall)

    def __rebuild_label_image(self, xs: np.ndarray, ys: np.ndarray, predictions: np.ndarray,
                              label_shape: tuple[int]) -> np.ndarray:
        label_images = np.zeros(label_shape)
        onehot_predictions = np.eye(label_shape[-1])[predictions]
        label_images[xs, ys] = onehot_predictions

        return label_images

    def __select_important_pixels(self, data: list[tuple[np.ndarray]]) -> list[
            tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]]:
        points = []

        for image, label in data:
            points.append(self.__get_datapoints_from_image(image, label))

        return points

    def __get_datapoints_from_image(self, image: np.ndarray, labels: np.ndarray) -> tuple[
            tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        xs, ys = ((image > self.__threshold).any(axis=-1)).nonzero()

        return (xs, ys), image[xs, ys], labels[xs, ys]


if __name__ == "__main__":
    SVMModel().train()
