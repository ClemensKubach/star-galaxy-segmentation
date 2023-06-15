from tqdm.contrib.concurrent import process_map
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
import numpy as np
from astropy.io import fits


class StatisticsService:
    def __init__(self, run: str = "1000") -> None:
        self.__data_provider = SDSSDataProvider(
            ImageDownloader('/Users/leopinetzki/data'))
        self.__data_provider.prepare(run)

    def get_channel_mean_variance(self, calculate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not calculate:
            return (np.array([0.06806371, 0.20728613, 0.464, 0.01954113, 0.393049]),
                    np.array([1.3366362, 2.3025422, 1.7791781, 0.80536276, 6.722137]))

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
            return np.array([155788, 1019846])

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
            covariance = np.array([[38.39584431,  2.17935757,  0.98275128,  0.98868845,  2.09980341],
                                   [2.17935757, 81.56378697,  2.21487135,
                                       0.94742019,  0.95599492],
                                   [0.98275128,  2.21487135, 29.7828834,
                                       2.01658154,  0.88864766],
                                   [0.98868845,  0.94742019,  2.01658154,
                                       52.01123953,  2.00083838],
                                   [2.09980341,  0.95599492,  0.88864766,  2.00083838, 26.48450165]])

            stds = self.get_channel_mean_variance()[1]
            stds0 = np.repeat(stds[:, None], stds.shape[0], axis=-1)
            stds1 = np.repeat(stds[None, :], stds.shape[0], axis=0)

            return covariance / (stds0 * stds1)

        results = process_map(self._get_correlation,
                              range(len(self.__data_provider)))

        stds = self.get_channel_mean_variance()[1]
        stds0 = np.repeat(stds[:, None], stds.shape[0], axis=-1)
        stds1 = np.repeat(stds[None, :], stds.shape[0], axis=0)

        return np.mean(results, axis=0) / (stds0 * stds1)
