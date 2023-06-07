from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import logging
import os
from typing import Optional
import requests
import bs4
import re

logger = logging.getLogger(__name__)


class ImageDownloader:
    URL = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/"
    LABEL_URL = "https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/"

    def __init__(self, to: str, run: Optional[str] = None, only_labels: bool = False, max_workers: int = 4, fast_check: bool = True) -> None:
        self.__max_workers = max_workers
        self.__to = to
        self.run = run
        self.__only_labels = only_labels
        self.__fast_check = fast_check

    def __prepare(self) -> tuple[str, str]:
        image_path = os.path.join(self.__to, 'images')
        label_path = os.path.join(self.__to, 'labels')

        os.makedirs(image_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        return image_path, label_path

    def download(self) -> tuple[list[str], list[str]]:
        image_path, label_path = self.__prepare()

        loaded_images = []
        if not self.__only_labels:
            loaded_images = self.__get_images(
                self.__combine_url(ImageDownloader.URL, self.run), image_path)

        loaded_labels = self.__get_images(
            ImageDownloader.LABEL_URL, label_path, re.compile(f"-0*{self.run}-.*-(gal|star)\.") if self.run is not None else re.compile(f"-(gal|star)\."))

        return loaded_images, loaded_labels

    def download_exact(self, run: str, camcol: str, field: str):
        image_path, label_path = self.__prepare()

        loaded_images = self.__get_images(
            self.__combine_url(ImageDownloader.URL, run), image_path, pattern=re.compile(f"[ugriz]-0*{run}-{camcol}-{field}"))

        loaded_labels = self.__get_images(
            ImageDownloader.LABEL_URL, label_path, re.compile(
                f"-0*{run}-{camcol}-(gal|star)\.")
        )

        return loaded_images, loaded_labels

    def __combine_url(self, base: str, extension: Optional[str]) -> str:
        if extension is None:
            return base

        return f"{base}{'/' if not base.endswith('/') and not extension.startswith('/') else ''}{extension}"

    def __is_file(self, name: str) -> bool:
        return name.endswith('.bz2') or name.endswith('.gz')

    def __ignore(self, name: str) -> bool:
        return '.' in name or not '/' in name

    def __has_pattern_and_matches(self, name: str, pattern: Optional[re.Pattern]):
        if pattern is None:
            return True

        if '/' in name:
            return True

        return bool(pattern.search(name))

    def __get_images(self, url: str, folder: str, pattern: Optional[re.Pattern] = None) -> list[str]:
        logger.info(f"Getting images from {url}")
        base_table = requests.get(url)

        try:
            base_table.raise_for_status()
        except:
            logger.warning(
                f"Got {base_table.status_code} for {url}:\n {base_table.content}"
            )
            return []

        html_data = bs4.BeautifulSoup(base_table.content, features="lxml")
        table = html_data.find('table', {'id': 'list'}).find('tbody')

        table_links = [row['href']
                       for row in table.find_all('a') if row['href'] != '../'
                       and self.__has_pattern_and_matches(row['href'], pattern)
                       ]

        loaded_images = self.__download_urls(
            [self.__combine_url(url, i) for i in table_links if self.__is_file(i)], folder)

        now_final_images = []
        for non_final_link in [i for i in table_links if not self.__is_file(i) and not self.__ignore(i)]:
            now_final_images.extend(self.__get_images(
                self.__combine_url(url, non_final_link), folder, pattern=pattern)
            )

        return now_final_images + loaded_images

    def __download_urls(self, urls: list[str], folder: str) -> list[str]:
        local_files = []

        if not urls:
            return local_files

        with ProcessPoolExecutor(max_workers=self.__max_workers) as handler:
            futures = handler.map(self._stream_download, urls, repeat(folder))

        local_files.extend((i for i in futures if i is not None))

        return local_files

    def _stream_download(self, url: str, folder) -> Optional[str]:
        file_name = url.split('/')[-1]
        file_folder = file_name.split('-')[2] + '/'
        os.makedirs(os.path.join(folder, file_folder), exist_ok=True)
        local_filename = os.path.join(folder, file_folder, file_name)

        if self.__fast_check and os.path.exists(local_filename):
            return local_filename

        with requests.get(url, stream=True) as r:
            try:
                r.raise_for_status()
            except:
                logger.warning(f"could not download {url}")
                return None

            file_size = int(r.headers['Content-Length'])
            if os.path.exists(local_filename) and os.path.getsize(local_filename) == file_size:
                return local_filename

            logger.info(f"Downloading {url}")

            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename
