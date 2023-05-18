

from collections import defaultdict
from itertools import chain, repeat
import numpy as np

from star_analysis.enum.phase import Phase


class DataSplitService():
    def __init__(self, deterministic: bool = True) -> None:
        self.__deterministic = deterministic

    def split(self, image: np.ndarray, chunk_shape: tuple[int, int], phase_distribution: dict[Phase, float]) -> dict[Phase, list[np.ndarray]]:
        if not sum(phase_distribution.values()) == 1:
            raise ValueError(
                f"The percentage of samples in each set must sum to one! {phase_distribution}")

        splitted_columns = np.array_split(
            image, indices_or_sections=chunk_shape[0], axis=0)
        chunks = [np.array_split(
            chunk, indices_or_sections=chunk_shape[0], axis=1) for chunk in splitted_columns]
        chunks = [chunk for lst in chunks for chunk in lst]

        chunks = self.__fix_shape(chunks=chunks, chunk_shape=chunk_shape)

        return self.__sort_into_phases(chunks=chunks, phase_distribution=phase_distribution)

    def __fix_shape(self, chunks: list[np.ndarray], chunk_shape: tuple[int, int]):
        corrected_chunnks = []
        for chunk in chunks:
            if chunk.shape == chunk_shape:
                corrected_chunnks.append(chunk)
                continue

            sized_image = np.zeros(chunk_shape)
            sized_image[:chunk.shape[0], :chunk_shape[1]] = chunk
            corrected_chunnks.append(sized_image)

        return corrected_chunnks

    def __sort_into_phases(self, chunks: list[np.ndarray], phase_distribution: dict[Phase, float]) -> dict[Phase, list[np.ndarray]]:
        if self.__deterministic:
            phase_data = self.__sort_into_phases_deterministic(
                chunks=chunks, phase_distribution=phase_distribution)
        else:
            phase_data = self.__sort_into_phases_random(
                chunks=chunks, phase_distribution=phase_distribution)

        return phase_data

    def __sort_into_phases_deterministic(self, chunks: list[np.ndarray], phase_distribution: dict[Phase, float]) -> dict[Phase, list[np.ndarray]]:
        phase_data = defaultdict(list)
        phases_to_assign = list(phase_distribution.keys())

        assigned_chunks_count = 0
        while phases_to_assign:
            for chunk, phase in zip(chunks[assigned_chunks_count:], chain.from_iterable(repeat(phases_to_assign))):
                phase_data[phase].append(chunk)

                if phase_data[phase] / len(chunks) >= phase_distribution[phase]:
                    phases_to_assign.remove(phase)
                    break

        return phase_data

    def __sort_into_phases_random(self, chunks: list[np.ndarray], phase_distribution: dict[Phase, float]) -> dict[Phase, list[np.ndarray]]:
        assigned_phases = np.random.choice(list(phase_distribution.keys()), len(
            chunks), replace=True, p=list(phase_distribution.values()))
        phase_data = defaultdict(list)

        for chunk, phase in zip(chunks, assigned_phases):
            phase_data[phase].append(chunk)

        return phase_data
