from star_analysis.data.datasets import Sdss


def download():
    Sdss(
        patch_shape=None,
        download=True
    )


if __name__ == '__main__':
    download()
