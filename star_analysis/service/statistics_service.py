from tqdm.contrib.concurrent import process_map
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider
import numpy as np


class StatisticsService:
    def __init__(self) -> None:
        self.__data_provider = SDSSDataProvider(
            ImageDownloader('/Users/leopinetzki/data'))
        self.__data_provider.prepare("1000")

    def get_channel_mean_variance(self, calculate: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not calculate:
            return (np.ndarray([0.06806371, 0.20728613, 0.464, 0.01954113, 0.393049]),
                    np.ndarray([1.3366362, 2.3025422, 1.7791781, 0.80536276, 6.722137]))

        results = process_map(self._get_image_mean_std,
                              range(len(self.__data_provider)))

        return np.mean([i[0] for i in results if i[0] is not None], axis=0), np.mean([i[1] for i in results if i[1] is not None], axis=0)

    def _get_image_mean_std(self, id: int) -> tuple[np.ndarray, np.ndarray]:
        image, _ = self.__data_provider[id]

        if image is None:
            return None, None

        return image.mean(axis=(0, 1)), image.std(axis=(0, 1))
