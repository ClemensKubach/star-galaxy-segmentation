import unittest

from star_analysis.data.configs import SdssDatasetConfig
from star_analysis.data.datasets import Sdss


class DatasetTestCase(unittest.TestCase):
    def test_loading(self):
        dataset = Sdss(
            SdssDatasetConfig(
                prepare=True,
                patch_shape=(224, 224)
            )
        )
        input_data, target = dataset[0]
        assert input_data.shape == (224, 224, 5)
        assert target.shape == (224, 224, 2)


if __name__ == '__main__':
    unittest.main()
