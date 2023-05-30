from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import logging
import os
from typing import Optional
import requests
import bs4

logger = logging.getLogger(__name__)


class ImageDownloader:
    URL = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/"
    LABEL_URL = "https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/"

    def __init__(self, limit: Optional[int] = None, max_workers: int = 4) -> None:
        self.__limit = limit
        self.__max_workers = max_workers

        self.__loaded_images = 0

    def download(self, to: str, only_labels: bool = False) -> tuple[list[str], list[str]]:
        image_path = os.path.join(to, 'images')
        label_path = os.path.join(to, 'labels')

        os.makedirs(image_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        loaded_images = []
        if not only_labels:
            loaded_images = self.__get_images(
                ImageDownloader.URL, image_path)

        self.__loaded_images = 0
        loaded_labels = self.__get_images(
            ImageDownloader.LABEL_URL, label_path)

        return loaded_images, loaded_labels

    def __combine_url(self, base: str, extension: str) -> str:
        return f"{base}{'/' if not base.endswith('/') and not extension.startswith('/') else ''}{extension}"

    def __is_file(self, name: str) -> bool:
        return name.endswith('.bz2') or name.endswith('.gz')

    def __get_images(self, url: str, folder: str) -> list[str]:
        if self.__limit is not None and self.__loaded_images >= self.__limit:
            return []

        logger.info(f"Getting images from {url}")
        base_table = requests.get(url)

        html_data = bs4.BeautifulSoup(base_table.content)
        table = html_data.find('table', {'id': 'list'}).find('tbody')

        table_links = [row['href']
                       for row in table.find_all('a') if row['href'] != '../'
                       ]

        loaded_images = self.__download_urls(
            [self.__combine_url(url, i) for i in table_links if self.__is_file(i)], folder)

        now_final_images = []
        for non_final_link in [i for i in table_links if not self.__is_file(i)]:
            now_final_images.extend(self.__get_images(
                self.__combine_url(url, non_final_link), folder)
            )

        return now_final_images + loaded_images

    def __download_urls(self, urls: list[str], folder: str) -> list[str]:
        local_files = []

        if not urls:
            return local_files

        if self.__limit is not None:
            urls = urls[:self.__limit - self.__loaded_images]

        with ProcessPoolExecutor(max_workers=self.__max_workers) as handler:
            futures = handler.map(self._stream_download, urls, repeat(folder))

        local_files.extend(futures)

        return local_files

    def _stream_download(self, url: str, folder) -> str:
        local_filename = os.path.join(folder, url.split('/')[-1])

        if os.path.exists(local_filename):
            return local_filename

        logger.info(f"Downloading {url}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        self.__loaded_images += 1
        return local_filename
