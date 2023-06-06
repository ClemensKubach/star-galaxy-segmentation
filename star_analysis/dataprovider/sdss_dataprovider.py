from __future__ import annotations
from dataclasses import dataclass
import os
import pathlib
from typing import Optional, Union
from star_analysis.service.alignment import AlignmentService
from astropy.io import fits

import numpy as np
from star_analysis.dataprovider.image_downloader import ImageDownloader


@dataclass
class ImageFile:
    spectrum: str
    run: str
    camcol: str
    frame_sequence: str
    fullname: str

    @classmethod
    def from_str(cls: ImageFile, name: str) -> ImageFile:
        data = name.split('/')[-1].split('.')[0].split('-')[1:]

        return cls(spectrum=data[0], run=data[1], camcol=data[2], frame_sequence=data[3], fullname=name)

    def get_keys(self) -> tuple[str]:
        return self.run, self.camcol, self.frame_sequence

    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.spectrum < other.spectrum

        return super().__lt__(other)

    def __gt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.spectrum > other.spectrum

        return super().__gt__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.spectrum == other.spectrum

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

    def get_keys(self) -> tuple[str]:
        return self.run, self.camcol

    def __lt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.type_ < other.type_

        return super().__lt__(other)

    def __gt__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.type_ > other.type_

        return super().__gt__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.type_ == other.type_

        return super().__eq__(other)


class SDSSDataProvider:
    def __init__(self, downloader: ImageDownloader = ImageDownloader(os.path.join((pathlib.Path(__file__).parent.parent.resolve(), 'data'))), alignment_service: Optional[AlignmentService] = None):
        self.__downloader = downloader
        self.__alignment_service = alignment_service

        self.__data_files = {}
        self.__label_files = {}

    @property
    def runs(self) -> list[str]:
        return list(self.__data_files.keys())

    @property
    def camcols(self) -> list[str]:
        camcols = {
            camcol for run in self.runs for camcol in self.__data_files[run]}

        return list(camcols)

    @property
    def frame_sequences(self) -> list[str]:
        frame_seqs = {
            frame_seq for run in self.runs for camcol in self.__data_files[run] for frame_seq in self.__data_files[run][camcol]}

        return list(frame_seqs)

    def prepare(self, batch: Optional[str] = None):
        if batch is not None:
            self.__downloader.batch = batch

        data_files, label_files = self.__downloader.download()

        self.__group_files(data_files, label_files)

    def __group_files(self, images: list[str], labels: list[str]):
        image_objs = [ImageFile.from_str(name) for name in images]
        label_objs = [LabelFile.from_str(name) for name in labels]

        self.__data_files = self.__create_object_map(image_objs)
        self.__label_files = self.__create_object_map(label_objs)

    def __create_object_map(self, objects: list[Union[ImageFile, LabelFile]]) -> dict:
        files = {}
        for obj in objects:
            keys = obj.get_keys()

            dict_level = files
            for i, key in enumerate(keys):
                is_last_level = i != len(keys)

                if key not in dict_level:
                    dict_level[key] = {} if not is_last_level else []

                if is_last_level:
                    dict_level[key].append(obj.fullname)
                    dict_level[key] = sorted(dict_level[key])

        return files

    def get_run(self, run: str) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
        return self.__data_files[run], self.__label_files[run]

    def get_run_camcol(self, run: str, camcol: str) -> tuple[dict[str, list[str]], list[str]]:
        return self.__data_files[run][camcol], self.__label_files[run][camcol]

    def __create_alignment_service(self):
        first_run = list(self.__label_files.keys())[0]
        first_camcol = list(self.__label_files[first_run].keys())[0]

        first_labels = [fits.open(file)[1].data['OBJC_TYPE']
                        for file in self.__label_files[first_run][first_camcol]]
        self.__alignment_service = AlignmentService(
            {j: i for i, j in enumerate(set(first_labels))})

    def get_aligned(self, run: str, camcol: str, frame_seq: str) -> np.ndarray:
        if self.__alignment_service is None:
            self.__alignment_service = self.__create_alignment_service()

        return self.__alignment_service.align(self.__data_files[run][camcol][frame_seq], self.__label_files[run][camcol])
