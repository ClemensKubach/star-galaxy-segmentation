import fire as fire

from star_analysis.runner.runner import Runner


class StaraCLI:
    """The command line interface for the Star Analysis package.
    For complex tasks, it is recommended to create a python script and use the runner directly.
    A simple example run can be done via this tool.

    Note: For Apple Silicon set the environment variable: PYTORCH_ENABLE_MPS_FALLBACK=1.
    """

    def run(self):
        """Executes the train, test, save routine of run_script.py."""
        from star_analysis.run_script import execute
        execute()

    class dataset:
        def download(self):
            """Downloads the SDSS dataset.
            Make sure your device not goes to sleep.
            Otherwise, you can use repair afterward.
            """
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
