from unittest.mock import patch, MagicMock
from star_analysis.service.alignment import AlignmentService
import numpy as np

x = 999999999
c = 5


def dummy_count(*args, **kwargs):
    global x
    x -= 1
    return x


def dummy_image(*args, **kwargs):
    global c
    c -= 1
    return np.ones((2, 2)) * c


@patch("star_analysis.service.alignment.AlignmentService._AlignmentService__align_image", autospec=True)
@patch("star_analysis.service.alignment.AlignmentService._AlignmentService__get_lost_pixel_count", autospec=True)
@patch("star_analysis.service.alignment.AlignmentService._AlignmentService__load_files", autospec=True)
def test_alignment_is_optimal(
    file_load_function_mock: MagicMock,
    get_lost_pixel_count_mock: MagicMock,
    align_image_mock: MagicMock
):
    file_load_function_mock.return_value = [1, 2, 3, 4, np.ones((2, 2)) * 5]
    align_image_mock.side_effect = dummy_image
    get_lost_pixel_count_mock.side_effect = dummy_count
    alignment_service = AlignmentService()
    aligned_result = alignment_service.align_optimal([0, 1, 2, 3, 4])

    assert aligned_result[0, 0, 0] == 5
    assert aligned_result[1, 0, 0] == 4
    assert aligned_result[2, 0, 0] == 3
    assert aligned_result[3, 0, 0] == 2
    assert aligned_result[4, 0, 0] == 1
