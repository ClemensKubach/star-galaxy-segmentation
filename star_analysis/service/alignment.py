import math
from typing import Union
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU
import numpy as np
import logging
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


logger = logging.getLogger(__name__)


class AlignmentService:
    def __load_files(self, files: list[Union[str, HDUList]]) -> list[HDUList]:
        return [(fits.open(file) if isinstance(file, str) else file) for file in files]

    def align(self, files: list[Union[str, HDUList]]) -> np.ndarray:
        hdu_frames = self.__load_files(files)

        reference = hdu_frames[0][0]
        results = [reference]
        for other in hdu_frames[1:]:
            _, other_cutout = self.__align_image(reference, other[0])
            results.append(other_cutout)

        min_dims = np.min([result.shape for result in results], axis=0)
        return np.stack([cutout_frame.data[:min_dims[0], :min_dims[1]] for cutout_frame in results]).T

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
