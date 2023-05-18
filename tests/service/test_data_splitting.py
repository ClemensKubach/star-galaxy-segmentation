from unittest.mock import MagicMock, patch
import numpy as np
from star_analysis.enum.phase import Phase
from star_analysis.service.data_split import DataSplitService


def get_dummy_image():
    return np.array([[1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1, 0]])


def dummy_fumction(*args, **kwargs):
    return kwargs['chunks'] if 'chunks' in kwargs else args[0]


def test_creates_chunks_of_correct_size():
    data_splitter = DataSplitService()
    shape = (2, 2)
    splitted_chunks = data_splitter.split(image=get_dummy_image(), chunk_shape=shape, phase_distribution={
        Phase.train: 1/3, Phase.test: 1/3, Phase.validation: 1/3})
    chunks = [i for lst in splitted_chunks.values() for i in lst]

    for chunk in chunks:
        assert chunk.shape == shape


@patch("star_analysis.service.data_split.DataSplitService._DataSplitService__sort_into_phases")
def test_creates_correct_chunks(phase_sorter_mock: MagicMock):
    phase_sorter_mock.side_effect = dummy_fumction

    data_splitter = DataSplitService()
    shape = (2, 2)
    splitted_chunks = data_splitter.split(image=get_dummy_image(), chunk_shape=shape, phase_distribution={
        Phase.train: 1/3, Phase.test: 1/3, Phase.validation: 1/3})

    expected_chunks = [np.array([[1, 1],
                                 [0, 1]]),
                       np.array([[0, 0],
                                 [1, 0]]),
                       np.array([[0, 0],
                                 [0, 0]]),
                       np.array([[0, 0],
                                 [0, 0]]),
                       np.array([[0, 0],
                                 [0, 0]]),
                       np.array([[1, 1],
                                 [0, 1]]),
                       np.array([[0, 0],
                                 [1, 0]]),
                       np.array([[0, 0],
                                 [0, 0]]),
                       np.array([[0, 0],
                                 [0, 0]]),
                       np.array([[0, 0],
                                 [0, 0]]),
                       np.array([[1, 1],
                                 [0, 0]]),
                       np.array([[0, 0],
                                 [0, 0]])]

    for real, expected in zip(splitted_chunks, expected_chunks):
        assert (real == expected).all()


def test_correct_split_ratio():
    data_splitter = DataSplitService()
    shape = (2, 2)
    distribution = {
        Phase.train: 7/10, Phase.test: 1/10, Phase.validation: 2/10}
    splitted_chunks = data_splitter.split(
        image=get_dummy_image(), chunk_shape=shape, phase_distribution=distribution)

    combined_len = len([i for lst in splitted_chunks.values() for i in lst])

    for phase, amount in distribution.items():
        assert len(splitted_chunks[phase]) / combined_len - amount < 0.05
