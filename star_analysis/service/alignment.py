from collections import defaultdict
from itertools import permutations
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU
from astropy.wcs import WCS
import numpy as np


class AlignmentService:
    def __load_files(self, files: list[str]) -> dict[HDUList]:
        return {file: fits.open(file) for file in files}

    def align_optimal(self, files: list[str]) -> np.ndarray:
        loaded_files = self.__load_files(files=files)

        lost_pixels_for_permutation = defaultdict(int)
        for permutation in permutations(files, len(files)):
            reference = loaded_files[permutation[0]]
            for other_file in permutation[1:]:
                other = loaded_files[other_file]
                lost_pixels_for_permutation[permutation] += self.__get_lost_pixel_count(
                    reference=reference, other=other)

        best_permutation = min(lost_pixels_for_permutation,
                               key=lost_pixels_for_permutation.get)

        reference = loaded_files[best_permutation[0]]
        aligned_images = [reference]
        for other_file in best_permutation[1:]:
            other = loaded_files[other_file]
            aligned_images.append(self.__align_image(
                reference=reference, other=other))

        return np.stack(aligned_images)

    def __get_lost_pixel_count(self, reference: PrimaryHDU, other: PrimaryHDU) -> int:
        _, aligned_pixels = self.__align_pixels(
            reference=reference, other=other)

        return (aligned_pixels < 0).any(axis=-1).sum()

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

        mask = ~(aligned_pixels < 0).any(axis=-1)
        aligned_image = np.zeros(reference.shape)

        masked_coords = coords[mask].astype(int)
        masked_pixels = aligned_pixels[mask].astype(int)
        aligned_image[masked_coords[:, 0], masked_coords[:, 1]
                      ] = other.data[masked_pixels[:, 0], masked_pixels[:, 1]]

        return aligned_image
