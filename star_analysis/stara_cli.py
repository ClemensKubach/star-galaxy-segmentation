import fire as fire

from star_analysis.runner.runner import Runner


class StaraCLI:
    """The command line interface for the Star Analysis package."""

    class ml(Runner):
        pass

    class dataset:
        def download(self):
            """Downloads the SDSS dataset."""
            from star_analysis.utils.download_dataset import download_sdss_dataset
            download_sdss_dataset()

        def repair(self):
            """Repairs your local files."""
            from star_analysis.utils.repair_dataset import repair_data
            repair_data()

        def align(self):
            """Aligns the downloaded fits files."""
            from star_analysis.utils.alignsave_dataset import align_data
            align_data()


if __name__ == '__main__':
    fire.Fire(StaraCLI)
