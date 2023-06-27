from star_analysis.data.configs import SdssDatasetConfig
from star_analysis.data.datasets import Sdss


def download_sdss_dataset(config: SdssDatasetConfig = None):
    print("Downloading SDSS dataset...")
    if config:
        if config.prepare:
            Sdss(config=config)
        else:
            print("Download is set to False, skipping download.")
    else:
        Sdss(
            SdssDatasetConfig(
                prepare=True
            )
        )


if __name__ == '__main__':
    download_sdss_dataset()
