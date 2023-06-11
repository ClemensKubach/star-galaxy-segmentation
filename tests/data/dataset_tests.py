import unittest

from star_analysis.data.datasets import Sdss


class DatasetCase(unittest.TestCase):
    def test_loading(self):
        dataset = Sdss(
            patch_shape=(32, 32),
            download=False
        )
        print(dataset[0])


if __name__ == '__main__':
    unittest.main()
