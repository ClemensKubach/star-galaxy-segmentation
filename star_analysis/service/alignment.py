from typing import Union
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
        self.__label_encoder = label_encoder

    def __load_files(self, files: list[Union[str, HDUList]]) -> list[HDUList]:
        return [(fits.open(file) if isinstance(file, str) else file) for file in files]

    def align(self, files: list[Union[str, HDUList]], labels: list[Union[str, HDUList]]) -> np.ndarray:
        hdu_frames = self.__load_files(files)
        label_frames = [tables[1] for tables in self.__load_files(
            labels)
        ]

        reference = hdu_frames[0][0]
        results = [reference]
        for other in hdu_frames[1:]:
            _, other_cutout = self.__align_image(reference, other[0])
            results.append(other_cutout)

        min_dims = np.min([result.shape for result in results], axis=0)
        stacked_image = np.stack(
            [cutout_frame.data[:min_dims[0], :min_dims[1]] for cutout_frame in results]).T

        label_map = self.__create_label_map(
            WCS(reference), hdu_frames[0][3], label_frames, stacked_image.shape[:-1])

        return stacked_image, label_map

    def __align_image(self, reference: Union[PrimaryHDU, Cutout2D], other: PrimaryHDU) -> tuple[Cutout2D, Cutout2D]:
        reference_wcs = WCS(reference.header) if not isinstance(reference,
                                                                Cutout2D) else reference.wcs
        other_wcs = WCS(other.header)

        reference_coords = SkyCoord(
            ra=reference_wcs.wcs.crval[0]*u.deg, dec=reference_wcs.wcs.crval[1]*u.deg)
        other_coords = SkyCoord(
            ra=other_wcs.wcs.crval[0]*u.deg, dec=other_wcs.wcs.crval[1]*u.deg)

        cutout_1 = Cutout2D(reference.data, other_coords,
                            size=reference.data.shape, wcs=reference_wcs)
        cutout_2 = Cutout2D(other.data, reference_coords,
                            size=reference.data.shape, wcs=other_wcs)

        return cutout_1, cutout_2

    def __create_label_map(self, wcs: WCS, orig_image_frame_data: BinTableHDU, label_tables: list[BinTableHDU], base_size: tuple) -> np.ndarray:
        print(len(label_tables))
        label_base = np.zeros((*base_size, len(self.__label_encoder)))
        for label_table in label_tables:
            print(type(label_table))
            for run, type_, field, camcol,  ra, dec in zip(label_table.data['RUN'], label_table.data['OBJC_TYPE'], label_table.data['FIELD'], label_table.data['CAMCOL'], label_table.data['RA'], label_table.data['DEC']):
                if field != orig_image_frame_data.data['FIELD'] or camcol != orig_image_frame_data.data['CAMCOL'] or run != orig_image_frame_data.data['RUN']:
                    continue

                coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                pixels = np.round(coord.to_pixel(
                    wcs=wcs, origin=1), 0).astype(int)
                label_base[pixels[0], pixels[1],
                           self.__label_encoder[type_]] = 1

        return label_base
