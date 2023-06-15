from typing import Union, Optional
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU, BinTableHDU
import numpy as np
import logging
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import warnings

warnings.simplefilter('ignore')


logger = logging.getLogger(__name__)


class AlignmentService:
    def __init__(self, label_encoder: dict[int, int]) -> None:
        self.label_encoder = label_encoder

    @property
    def num_labels(self) -> int:
        return len(self.label_encoder)

    def __load_files(self, files: list[Union[str, HDUList]]) -> list[HDUList]:
        return [(fits.open(file) if isinstance(file, str) else file) for file in files]

    def __do_alignement(self, hdu_frames: list[HDUList], size: Optional[tuple] = None) -> list[Cutout2D]:
        results = [hdu_frames[0][0]]

        for i, other in enumerate(hdu_frames[1:]):
            _, other_cutout = self.__align_image(
                results[0], other[0], size=size)

            results.append(other_cutout)

        return results

    def align(self, files: list[Union[str, HDUList]], labels: list[Union[str, HDUList]]) -> tuple[np.ndarray, np.ndarray]:
        hdu_frames = self.__load_files(files)
        label_frames = [tables[1] for tables in self.__load_files(
            labels)
        ]

        cutouts = self.__do_alignement(hdu_frames)

        stacked_image = np.stack(
            [cutout_frame.data for cutout_frame in cutouts]).T

        label_map = self.__create_label_map(
            WCS(cutouts[0]), hdu_frames[0][3], label_frames, stacked_image.shape[:-1])

        return stacked_image, label_map

    def get_demanded_label_vectors(self, label_map: np.ndarray) -> list[np.ndarray]:
        vectors = []
        for data in label_map.T:
            xs, ys = data.nonzero()

            vectors.append(np.stack((xs, ys)).T)

        return vectors

    def __align_image(self, reference: Union[PrimaryHDU, Cutout2D], other: PrimaryHDU, size: Optional[tuple] = None) -> tuple[Cutout2D, Cutout2D]:
        reference_wcs = (WCS(reference.header)
                         if not isinstance(reference, Cutout2D)
                         else reference.wcs)
        other_wcs = (WCS(other.header)
                     if not isinstance(other, Cutout2D)
                     else other.wcs)

        reference_coords = SkyCoord(
            ra=reference_wcs.wcs.crval[0]*u.deg, dec=reference_wcs.wcs.crval[1]*u.deg)
        other_coords = SkyCoord(
            ra=other_wcs.wcs.crval[0]*u.deg, dec=other_wcs.wcs.crval[1]*u.deg)

        cutout_1 = Cutout2D(reference.data, other_coords,
                            size=reference.data.shape if size is None else size, wcs=reference_wcs,
                            mode='partial', fill_value=0)
        cutout_2 = Cutout2D(other.data, reference_coords,
                            size=reference.data.shape if size is None else size, wcs=other_wcs,
                            mode='partial', fill_value=0)

        return cutout_1, cutout_2

    def __create_label_map(self, wcs: WCS, orig_image_frame_data: BinTableHDU, label_tables: list[BinTableHDU], base_size: tuple) -> np.ndarray:
        label_base = np.zeros((*base_size, len(self.label_encoder)))
        for label_table in label_tables:
            for run, type_, field, camcol,  ra, dec in zip(label_table.data['RUN'], label_table.data['OBJC_TYPE'], label_table.data['FIELD'], label_table.data['CAMCOL'], label_table.data['RA'], label_table.data['DEC']):
                if field != orig_image_frame_data.data['FIELD'] or camcol != orig_image_frame_data.data['CAMCOL'] or run != orig_image_frame_data.data['RUN']:
                    continue

                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                pixels = np.round(coord.to_pixel(
                    wcs=wcs, origin=1), 0).astype(int)
                if pixels[0] >= label_base.shape[0] or pixels[1] >= label_base.shape[1]:
                    continue
                label_base[pixels[0], pixels[1],
                           self.label_encoder[type_]] = 1

        return label_base
