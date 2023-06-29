import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
import numpy as np
from astropy.io import fits
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class StatisticsService:
    def __init__(self, run: str = SDSSDataProvider.FIXED_VALIDATION_RUN, calculate: bool = False) -> None:
        if calculate:
            self.__data_provider = SDSSDataProvider()
            self.__data_provider.prepare(run)

        self.__calculate = calculate

    def get_channel_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.__calculate:
            return (np.array([0.01829018, 0.06763462, 0.03478437, 0.00994593, 0.09194765]),
                    np.array([0.6351227, 1.1362617, 0.8386613, 0.7339489, 3.5971174]))

        results = process_map(self._get_image_mean_std,
                              range(len(self.__data_provider)))

        return np.mean([i[0] for i in results if i[0] is not None], axis=0), np.mean([i[1] for i in results if i[1] is not None], axis=0)

    def _get_image_mean_std(self, id: int) -> tuple[np.ndarray, np.ndarray]:
        image, _ = self.__data_provider[id]

        if image is None:
            return None, None

        return image.mean(axis=(0, 1)), image.std(axis=(0, 1))

    def get_number_examples_per_class(self) -> np.ndarray:
        if not self.__calculate:
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

    def get_color_covariance(self) -> np.ndarray:
        if not self.__calculate:
            return np.array([[2.89121563e+00, -1.05176982e-04, -1.89354782e-02,
                              -1.03383729e-03, -2.07394063e-04],
                             [-1.05176982e-04,  2.73075300e+00, -2.68645749e-02,
                              -2.14423000e-03, -1.08504346e-03],
                             [-1.89354782e-02, -2.68645749e-02,  3.40216588e+01,
                              -1.06141428e-02, -8.34362314e-03],
                             [-1.03383729e-03, -2.14423000e-03, -1.06141428e-02,
                              3.26035539e+00, -2.64182748e-04],
                             [-2.07394063e-04, -1.08504346e-03, -8.34362314e-03,
                              -2.64182748e-04,  3.30714957e+00]])

        results = []
        try:
            for image, _ in tqdm(self.__data_provider):
                results.append(np.cov(image.reshape(image.shape[-1], -1)))
        except KeyError:
            pass

        return np.mean(results, axis=0)

    def get_color_covariance_per_class(self) -> np.ndarray:
        if not self.__calculate:
            return (np.array([[6.80124327e-01,  5.63158108e-04, -1.15268030e-05,
                               -3.30873009e-04, -2.82045660e-04],
                              [5.63158108e-04,  5.23857142e+00, -1.34290099e-03,
                               -4.58022780e-04, -7.29065036e-04],
                              [-1.15268030e-05, -1.34290099e-03,  2.12589264e+00,
                               -1.12678654e-03, -3.15711925e-04],
                              [-3.30873009e-04, -4.58022780e-04, -1.12678654e-03,
                               1.90096378e+00, -1.75878757e-04],
                              [-2.82045660e-04, -7.29065036e-04, -3.15711925e-04,
                               -1.75878757e-04,  2.55962936e+00]]),
                    np.array([[2.17620018e+00, -7.38086617e-04,  1.11209774e-03,
                               -1.09927403e-03, -7.78044286e-04],
                              [-7.38086617e-04,  2.18214456e+00, -2.71485652e-04,
                               -4.53496217e-04, -3.57630746e-04],
                              [1.11209774e-03, -2.71485652e-04,  3.19908481e+00,
                               -3.91866787e-04, -6.89360352e-04],
                              [-1.09927403e-03, -4.53496217e-04, -3.91866787e-04,
                               4.53817152e+00, -7.37141842e-04],
                              [-7.78044286e-04, -3.57630746e-04, -6.89360352e-04,
                               -7.37141842e-04,  6.15172588e+00]]))

        cov1_ = []
        cov2_ = []
        try:
            for image, label in tqdm(self.__data_provider):
                reshaped = image.reshape(
                    image.shape[-1], -1)
                label_reshaped = label.reshape(
                    label.shape[-1], -1).astype(bool)
                cov1 = np.cov(reshaped[:, label_reshaped[0]])
                cov2 = np.cov(reshaped[:, label_reshaped[1]])
                if label_reshaped[0].any():
                    cov1_.append(cov1)
                if label_reshaped[1].any():
                    cov2_.append(cov2)
        except KeyError:
            pass

        return np.mean(cov1_, axis=0), np.mean(cov2_, axis=0)

    def get_color_correlation(self) -> np.ndarray:
        if not self.__calculate:
            return np.array([[1.00000000e+00, -4.17509405e-05, -2.05024948e-04,
                              -2.41507421e-04, -4.93587182e-05],
                             [-4.17509405e-05,  1.00000000e+00, -9.25423299e-05,
                              -2.02164901e-04, -2.39734696e-04],
                             [-2.05024948e-04, -9.25423299e-05,  1.00000000e+00,
                              -9.81700911e-05, -1.52132613e-04],
                             [-2.41507421e-04, -2.02164901e-04, -9.81700911e-05,
                              1.00000000e+00, -2.36536908e-05],
                             [-4.93587182e-05, -2.39734696e-04, -1.52132613e-04,
                              -2.36536908e-05,  1.00000000e+00]])

        results = []
        for image, _ in tqdm(self.__data_provider):
            results.append(np.corrcoef(image.reshape(image.shape[-1], -1)))

        return np.mean(results, axis=0)

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

    def get_distribution_per_class(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.__calculate:
            return (np.array([[0.3555972, 0.66457057, 0.41107112, 0.5257245, 1.7446687],
                              [1.7689778, 2.3413954, 1.8904239, 0.7942855, 4.4665427]]),
                    np.array([[1.9958429,  2.7264001,  1.9724108,  4.534722, 11.216864],
                              [7.0836196,  7.499741,  6.6610627,  6.519651, 21.924402]]))

        results = process_map(self._get_mean_std_per_class,
                              range(len(self.__data_provider)))

        return np.mean([i[0] for i in results if i[0] is not None], axis=0), np.mean([i[1] for i in results if i[1] is not None], axis=0)
