import unittest

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
        input_data, target = self.dataset[0]
        assert input_data.shape == (224, 224, 5)
        assert target.shape == (224, 224, 2)

    def test_completeness(self):
        """Takes a really long time. It only uses a single core here. Should be run only once."""
        assert len(self.dataset) == 33180  # ((1568*1120)//(224*224)) * 6 * 158
        for i in range(len(self.dataset))[:5]:
            input_data, target = self.dataset[i]
            assert input_data is not None
            assert target is not None


if __name__ == '__main__':
    unittest.main()
