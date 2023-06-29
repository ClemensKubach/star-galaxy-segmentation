from __future__ import annotations
from dataclasses import dataclass
import os
import pathlib
from typing import Optional, Union

from tqdm import tqdm

from star_analysis.service.alignment import AlignmentService
from astropy.io import fits
import logging

import numpy as np
from star_analysis.dataprovider.image_downloader import ImageDownloader
from star_analysis.utils.constants import DATAFILES_ROOT

logger = logging.getLogger(__name__)


@dataclass
class ImageFile:
    spectrum: str
    run: str
    camcol: str
    field: str
    fullname: str

    @classmethod
    def from_str(cls: ImageFile, name: str) -> ImageFile:
        data = name.split('/')[-1].split('.')[0].split('-')[1:]

        return cls(spectrum=data[0], run=data[1], camcol=data[2], field=data[3], fullname=name)

    def get_keys(self) -> tuple[str, str, str]:
        return self.run, self.camcol, self.field

    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.run < other.run:
                return True
            if self.camcol < other.camcol:
                return True
            if self.field < other.field:
                return True

            return self.spectrum < other.spectrum

        return super().__lt__(other)

    def __gt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.run > other.run:
                return True
            if self.camcol > other.camcol:
                return True
            if self.field > other.field:
                return True

            return self.spectrum > other.spectrum

        return super().__gt__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.spectrum == other.spectrum and self.run == other.run and self.camcol == other.camcol and self.field == other.field

        return super().__eq__(other)


@dataclass
class LabelFile:
    run: str
    camcol: str
    type_: str
    fullname: str

    @classmethod
    def from_str(cls: LabelFile, name: str) -> LabelFile:
        data = name.split('/')[-1].split('.')[0].split('-')[1:]

        return cls(run=data[0], camcol=data[1], type_=data[2], fullname=name)

    def get_keys(self) -> tuple[str, str]:
        return self.run, self.camcol

    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.run < other.run:
                return True
            if self.camcol < other.camcol:
                return True

            return self.type_ < other.type_

        return super().__lt__(other)

    def __gt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.run > other.run:
                return True
            if self.camcol > other.camcol:
                return True

            return self.type_ > other.type_

        return super().__gt__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.type_ == other.type_ and self.run == other.run and self.camcol == other.camcol

        return super().__eq__(other)


class SDSSDataProvider:
    FIXED_VALIDATION_FIELD = "0080"
    FIXED_VALIDATION_CAMCOL = "6"
    FIXED_VALIDATION_RUN = "8162"
    SINGLETON_DOWNLOADER = ImageDownloader(DATAFILES_ROOT, max_workers=os.cpu_count(), run=FIXED_VALIDATION_RUN)

    def __init__(
            self,
            downloader: ImageDownloader | None = None,
            alignment_service: Optional[AlignmentService] = None,
            include_train_set: bool = True,
            include_test_set: bool = False,
            force_realign: bool = False,
            save_new_alignments: bool = True,
            use_mmap: bool = True
    ):
        self.__downloader = downloader if downloader else SDSSDataProvider.SINGLETON_DOWNLOADER
        self.alignment_service = alignment_service
        self.include_train_set = include_train_set
        self.include_test_set = include_test_set
        self.force_realign = force_realign
        self.save_new_alignments = save_new_alignments
        self.use_mmap = use_mmap

        self.__data_files = {}
        self.__label_files = {}

        self.__data_as_list: list[tuple[list[str], list[str]]] = []
        self.__indexed_data: dict[int, tuple[list[str], list[str]]] = {}

        self.__fixed_validation_files = self.__downloader.download_exact(
            run=SDSSDataProvider.FIXED_VALIDATION_RUN,
            camcol=SDSSDataProvider.FIXED_VALIDATION_CAMCOL,
            field=SDSSDataProvider.FIXED_VALIDATION_FIELD
        )

    @property
    def runs(self) -> list[str]:
        return list(self.__data_files.keys())

    @property
    def camcols(self) -> list[str]:
        camcols = {
            camcol for run in self.runs for camcol in self.__data_files[run]}

        return list(camcols)

    @property
    def fields(self) -> list[str]:
        frame_seqs = {
            frame_seq for run in self.runs for camcol in self.__data_files[run] for frame_seq in
            self.__data_files[run][camcol]
        }
        return list(frame_seqs)

    @property
    def num_labels(self) -> int:
        if self.alignment_service is None:
            self.__create_alignment_service()
        return self.alignment_service.num_labels

    def prepare(self, run: Optional[str] = None):
        if run is not None:
            self.__downloader.run = run

        data_files, label_files = self.__downloader.load()

        self.__group_files(data_files, label_files)

    def prepare_exact(self, run: str, camcol: str, field: str, force_download: bool = False):
        data_files, label_files = self.__downloader.download_exact(
            run=run, camcol=camcol, field=field, force=force_download)

        self.__group_files(data_files, label_files)

    def __group_files(self, images: list[str], labels: list[str]):
        image_objs = sorted([ImageFile.from_str(name) for name in images])
        label_objs = sorted([LabelFile.from_str(name) for name in labels])

        self.__data_files = self.__create_object_map(image_objs)
        self.__label_files = self.__create_object_map(label_objs)

        for run, run_data in self.__data_files.items():
            for camcol, camcol_data in run_data.items():
                for field, field_data in camcol_data.items():
                    if field.lstrip('0') == SDSSDataProvider.FIXED_VALIDATION_FIELD.lstrip('0') \
                            and camcol.lstrip('0') == SDSSDataProvider.FIXED_VALIDATION_CAMCOL.lstrip('0') \
                            and run.lstrip('0') == SDSSDataProvider.FIXED_VALIDATION_RUN.lstrip('0'):
                        if not self.include_test_set:
                            continue
                    else:
                        if not self.include_train_set:
                            continue
                    self.__data_as_list.append(
                        (field_data, self.__label_files[run][camcol])
                    )
        print(f"Found {len(self.__data_as_list)} data/label pairs")
        self.__indexed_data = dict(enumerate(self.__data_as_list))

    def __create_object_map(self, objects: list[Union[ImageFile, LabelFile]]) -> dict:
        files = {}
        for obj in objects:
            keys = obj.get_keys()

            dict_level = files
            for i, key in enumerate(keys):
                is_last_level = i == (len(keys) - 1)

                if key not in dict_level:
                    dict_level[key] = {} if not is_last_level else []

                if is_last_level:
                    dict_level[key].append(obj.fullname)
                    dict_level[key] = sorted(dict_level[key])
                else:
                    dict_level = dict_level[key]

        return files

    def get_run(self, run: str) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
        return self.__data_files[run], self.__label_files[run]

    def get_run_camcol(self, run: str, camcol: str) -> tuple[dict[str, list[str]], list[str]]:
        return self.__data_files[run][camcol], self.__label_files[run][camcol]

    def __create_alignment_service(self):
        first_labels = [fits.open(file)[1].data['OBJC_TYPE'][0]
                        for file in self.__fixed_validation_files[1]]
        self.alignment_service = AlignmentService(
            {j: i for i, j in enumerate(set(first_labels))})

        return self.alignment_service

    def repair(self, run: Optional[str] = None):
        def __do_repair(index: int, force_realign: bool):
            image_obj = ImageFile.from_str(self.__indexed_data[index][0][0])
            self.__downloader.download_exact(
                run=image_obj.run.lstrip("0"),
                camcol=image_obj.camcol,
                field=image_obj.field,
                force=True
            )
            self.force_realign = True
            _x, _y = self._get_aligned_by_data_index(index)
            if _x is None or _y is None:
                raise ValueError("Could not repair files. x or y is None.")
            else:
                logger.info(f"Successfully repaired files for run {image_obj.run}, camcol {image_obj.camcol}, field {image_obj.field}.")
            self.force_realign = force_realign

        def __handle_repair_error(index: int, ex: Exception):
            image_obj = ImageFile.from_str(self.__indexed_data[index][0][0])
            logger.warning(f"Unexpected error occurred while repairing files for run {image_obj.run}, camcol {image_obj.camcol}, field {image_obj.field}. Skipping.")
            logger.warning(ex)

        self.prepare(run=run)
        for idx in tqdm(range(len(self.__indexed_data))):
            try:
                x, y = self._get_aligned_by_data_index(idx)
                assert x is not None and y is not None
            except OSError or EOFError or AssertionError:
                try:
                    __do_repair(idx, self.force_realign)
                except Exception as e:
                    __handle_repair_error(idx, e)
            except Exception as e:
                __handle_repair_error(idx, e)

    def _get_aligned(self, run: str, camcol: str, field: str) -> tuple[np.ndarray, np.ndarray]:
        aligned_path = os.path.join(DATAFILES_ROOT, 'aligned')
        if not os.path.exists(aligned_path):
            os.makedirs(aligned_path)
        x_filepath = os.path.join(aligned_path, f"aligned_{run}-{camcol}-{field}_x.npy")
        y_filepath = os.path.join(aligned_path, f"aligned_{run}-{camcol}-{field}_y.npy")

        x_aligned_file = pathlib.Path(x_filepath)
        y_aligned_file = pathlib.Path(y_filepath)

        if self.force_realign or not x_aligned_file.exists() or not y_aligned_file.exists():
            if self.alignment_service is None:
                self.alignment_service = self.__create_alignment_service()

            x, y = self.alignment_service.align(self.__data_files[run][camcol][field], self.__label_files[run][camcol])
            if self.save_new_alignments:
                np.save(x_filepath, x)
                np.save(y_filepath, y)
        else:
            if self.use_mmap:
                mmap_mode = 'r'
            else:
                mmap_mode = None
            x = np.load(x_filepath, mmap_mode=mmap_mode)
            y = np.load(y_filepath, mmap_mode=mmap_mode)
        return x, y

    def _get_aligned_by_data_index(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        i = ImageFile.from_str(self.__indexed_data[index][0][0])
        return self._get_aligned(i.run, i.camcol, i.field)

    def get_provided_validation_set(self) -> tuple[np.ndarray, np.ndarray]:
        i = ImageFile.from_str(self.__fixed_validation_files[0][0])
        return self._get_aligned(i.run, i.camcol, i.field)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            return self._get_aligned_by_data_index(item)
        except Exception as e:
            image_obj = ImageFile.from_str(self.__indexed_data[item][0][0])
            logger.debug(e)
            logger.warning(
                f"Skipping {image_obj.run}, {image_obj.camcol} {image_obj.field}")
            return None

    def __next__(self) -> tuple[np.ndarray, np.ndarray] | None:
        for idx, images, labels in enumerate(self.__data_as_list):
            try:
                yield self._get_aligned_by_data_index(idx)
            except (OSError, EOFError):
                image_obj = ImageFile.from_str(images[0])
                logger.warning(
                    f"Skipping {image_obj.run}, {image_obj.camcol} {image_obj.field}")
                return None

    def get_label_files(self) -> tuple[list[str]]:
        data = [set() for _ in range(len(self.__data_as_list[0][1]))]

        for _, labels in self.__data_as_list:
            for i, file in enumerate(labels):
                data[i].add(file)

        return tuple(data)

    def __len__(self) -> int:
        return len(self.__indexed_data)
