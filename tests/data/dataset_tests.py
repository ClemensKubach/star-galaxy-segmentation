import unittest

from star_analysis.data.configs import SdssDatasetConfig
from star_analysis.data.datasets import Sdss


class DatasetCase(unittest.TestCase):
    def test_loading(self):
        dataset = Sdss(
            SdssDatasetConfig(
                prepare=True,
                patch_shape=(32, 32)
            )
        )
        print(dataset[0])


if __name__ == '__main__':
    unittest.main()
