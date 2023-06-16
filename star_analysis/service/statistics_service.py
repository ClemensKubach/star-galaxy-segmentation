from tqdm.contrib.concurrent import process_map
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
import numpy as np
from astropy.io import fits
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class StatisticsService:
    def __init__(self, run: str = SDSSDataProvider.FIXED_VALIDATION_RUN) -> None:
        self.__data_provider = SDSSDataProvider(
            ImageDownloader('/Users/leopinetzki/data'))
        self.__data_provider.prepare(run)

    def get_channel_mean_variance(self, calculate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not calculate:
            return (np.array([0.01831526, 0.06806776, 0.03508206, 0.00999219, 0.09221864]),
                    np.array([0.63541394, 1.1390159, 0.84219295, 0.7349062, 3.5991151]))

        results = process_map(self._get_image_mean_std,
                              range(len(self.__data_provider)))

        return np.mean([i[0] for i in results if i[0] is not None], axis=0), np.mean([i[1] for i in results if i[1] is not None], axis=0)

    def _get_image_mean_std(self, id: int) -> tuple[np.ndarray, np.ndarray]:
        image, _ = self.__data_provider[id]

        if image is None:
            return None, None

        return image.mean(axis=(0, 1)), image.std(axis=(0, 1))

    def get_number_examples_per_class(self, calculate: bool = False) -> np.ndarray:
        if not calculate:
            return np.array([287404, 538043])

        label_files = self.__data_provider.get_label_files()

        data = np.zeros((len(label_files), ))
        for i, cls_labels in enumerate(label_files):
            results = process_map(self._get_label_count, cls_labels)
            data[i] = sum(results)

        return data

    def _get_label_count(self, file: str) -> int:
        data = fits.open(file)[1]

        return len(data.data)

    def _get_covariance(self, id: int) -> np.ndarray:
        image, _ = self.__data_provider[id]
        image = image - self.get_channel_mean_variance()[0][None, None, :]

        return np.cov(image.reshape(image.shape[-1], -1))

    def get_color_correlation(self, calculate: bool = False) -> np.ndarray:
        if not calculate:
            return np.array([[1.64213383e+00, 3.98713894e-03, 1.59239000e-02, 6.90654871e-03,
                              6.71242042e-04],
                             [3.98713894e-03, 5.19692494e-01, 2.18571724e-02, 5.55899891e-03,
                              4.81904216e-04],
                             [1.59239000e-02, 2.18571724e-02, 1.07855631e+01, 5.69401040e-02,
                              2.29937496e-03],
                             [6.90654871e-03, 5.55899891e-03, 5.69401040e-02, 5.06676116e+00,
                              1.18142112e-03],
                             [6.71242042e-04, 4.81904216e-04, 2.29937496e-03, 1.18142112e-03,
                              7.43233091e-02]])

        results = process_map(self._get_covariance,
                              range(len(self.__data_provider)))

        stds = self.get_channel_mean_variance()[1]
        stds0 = np.repeat(stds[:, None], stds.shape[0], axis=-1)
        stds1 = np.repeat(stds[None, :], stds.shape[0], axis=0)

        return np.mean(results, axis=0) / (stds0 * stds1)

    def __plot_class(self, data: np.ndarray, color: str, title: str):
        sns.lmplot(x='ra', y='dec', data=data, palette=[color],
                   scatter_kws={'s': 2}, fit_reg=False, height=4, aspect=2)
        plt.ylabel('dec')
        plt.xlabel('Equitorial coordinates')
        plt.title(title)
        plt.show()

    def plot_class_distribution(self):
        label_files = self.__data_provider.get_label_files()
        label_fits = [[fits.open(file)[1] for file in label_class]
                      for label_class in label_files]

        coordinates_for_class = [[] for _ in range(len(label_files))]
        for i, class_files in enumerate(label_fits):
            for file in class_files:
                coordinates_for_class[i].extend(
                    zip(file.data['RA'], file.data['DEC']))

        self.__plot_class(
            pd.DataFrame(coordinates_for_class[0], columns=['ra', 'dec']), color='red', title="Galaxy")
        self.__plot_class(
            pd.DataFrame(coordinates_for_class[1], columns=['ra', 'dec']), color='blue', title="Star")

    def _get_mean_std_per_class(self, id: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        image, labels = self.__data_provider[id]

        if image is None:
            return None, None

        means = []
        stds = []
        for label_map in labels.T.astype(bool):
            mean = np.mean(image[label_map.T], axis=0)
            mean[np.isnan(mean)] = 0
            means.append(mean)

            var = np.std(image[label_map.T], axis=0)
            var[np.isnan(var)] = 0
            stds.append(var)

        return means, stds

    def get_distribution_per_class(self, calculate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not calculate:
            return []

        results = process_map(self._get_mean_std_per_class,
                              range(len(self.__data_provider)))

        return np.mean([i[0] for i in results if i[0] is not None], axis=0), np.mean([i[1] for i in results if i[1] is not None], axis=0)
