from star_analysis.dataprovider.sdss_dataprovider import SDSSDataProvider


def repair_data():
    print("Repairing SDSS dataset...")
    provider = SDSSDataProvider(
        include_train_set=True,
        include_test_set=True,
        force_realign=False,
        save_new_alignments=True
    )
    provider.repair()


if __name__ == '__main__':
    repair_data()
