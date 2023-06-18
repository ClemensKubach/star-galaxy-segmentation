import unittest

from torch import Tensor
from tqdm import tqdm

from star_analysis.data.configs import SdssDatasetConfig
from star_analysis.data.datasets import Sdss


class DatasetTestCase(unittest.TestCase):

    def setUp(self):
        self.dataset = Sdss(
            SdssDatasetConfig(
                prepare=False,
                patch_shape=(224, 224)
            )
        )
        self.dataset.prepare()

    def test_loading(self):
        input_data, target = self.dataset[0]
        assert input_data is not None
        assert target is not None

    def test_size(self):
        for i in range(len(self.dataset)):
            input_data, target = self.dataset[i]
            assert input_data.shape == (224, 224, 5)
            assert target.shape == (224, 224, 2)

    def test_completeness(self):
        """Takes a really long time. It only uses a single core here. Should be run only once."""
        size = self.dataset.cropped_image_shape[0] * self.dataset.cropped_image_shape[1]
        patch_size = self.dataset.patch_shape[0] * self.dataset.patch_shape[1]
        camcols = len(self.dataset.provider.camcols)
        fields = len(self.dataset.provider.fields)

        assert len(self.dataset) == size // patch_size * camcols * fields
        for i in range(len(self.dataset)):
            out = self.dataset[i]
            assert out is not None
            input_data, target = out
            assert input_data is not None
            assert target is not None
            _, __ = Tensor(input_data), Tensor(target)


if __name__ == '__main__':
    unittest.main()
