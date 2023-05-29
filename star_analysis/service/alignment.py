from collections import defaultdict
from itertools import permutations
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU
from astropy.wcs import WCS
import numpy as np
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class AlignmentService:
    def __load_files(self, files: list[str]) -> dict[str, HDUList]:
        return {file: fits.open(file) for file in files}

    def align_optimal(self, files: list[str]) -> np.ndarray:
        loaded_files = self.__load_files(files=files)

        lost_pixels_for_permutation = defaultdict(int)
        logger.info("Finding best alignment")
        for file in tqdm(files):
            reference = loaded_files[file]
            for other_file in files:
                if file == other_file:
                    continue

                other = loaded_files[other_file]
                lost_pixels_for_permutation[file] += self.__get_lost_pixel_count(
                    reference=reference[0], other=other[0])

        best_permutation = min(lost_pixels_for_permutation,
                               key=lost_pixels_for_permutation.get)

        logger.info("Aligning images")
        reference = loaded_files[best_permutation]
        aligned_images = [reference[0].data]
        for other_file in tqdm(files):
            if other_file == best_permutation:
                continue

            other = loaded_files[other_file]
            aligned_images.append(self.__align_image(
                reference=reference[0], other=other[0]))

        return np.stack(aligned_images)

    def __get_lost_pixel_count(self, reference: PrimaryHDU, other: PrimaryHDU) -> int:
        _, aligned_pixels = self.__align_pixels(
            reference=reference, other=other)

        return self.__get_pixel_mask(reference, aligned_pixels).sum()

    def __get_pixel_mask(self, reference: PrimaryHDU, pixel_coords: np.ndarray) -> np.ndarray:
        return ~(pixel_coords < 0).any(
            axis=-1) & ((pixel_coords[:, 0] < reference.shape[0])) & ((pixel_coords[:, 1] < reference.shape[1]))

    def __align_pixels(self, reference: PrimaryHDU, other: PrimaryHDU) -> tuple[np.ndarray, np.ndarray]:
        reference_system = WCS(reference.header)
        other_system = WCS(other.header)

        xx, yy = np.meshgrid(
            range(reference.shape[0]), range(reference.shape[1]))
        coords = np.stack(
            (xx, yy), axis=-1).reshape((reference.shape[0] * reference.shape[1], 2))

        world_coords = reference_system.all_pix2world(coords, 1)
        converted_pixels = other_system.all_world2pix(world_coords, 1)

        return coords, converted_pixels

    def __align_image(self, reference: PrimaryHDU, other: PrimaryHDU) -> np.ndarray:
        coords, aligned_pixels = self.__align_pixels(
            reference=reference, other=other)

        aligned_pixels = np.round(aligned_pixels, 0).astype(int)
        mask = self.__get_pixel_mask(reference, aligned_pixels)
        aligned_image = np.zeros(reference.shape)

        masked_coords = coords[mask].astype(int)
        masked_pixels = aligned_pixels[mask]
        aligned_image[masked_coords[:, 0], masked_coords[:, 1]
                      ] = other.data[masked_pixels[:, 0], masked_pixels[:, 1]]

        return aligned_image
