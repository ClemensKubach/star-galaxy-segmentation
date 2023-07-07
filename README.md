# star_analysis

## Getting Started
### Prerequisites
- Python 3.10
- `pip install -r requirements.txt`
- navigate to the project root directory
- configure PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)`

Prerequisites for training:
- `python star_analysis/stara_cli.py download`. This can take hours depending on the server load.
- `python star_analysis/stara_cli.py repair`. This is only necessary, if the internet connection is lost while downloading. The downloaded files are checked for integrity and repaired if necessary.
- `python star_analysis/stara_cli.py align`. This can take a while but reduces computation time later on.


### Usage
Our deep learning approach can be used via two entrypoints:
- `python star_analysis/run_script.py` to train and test a model on the SDSS dataset.
- `python star_analysis/stara_cli.py --help` as a wip command line tool as a collection of useful additional utils like downloading, repairing and aligning data.
- for just using the trained model, take a look at the visualization notebook, linked below.

We recommend, if available, to use already downloaded and aligned data for faster processing. 
If the data is at the correct location and the script run from the correct location, the data should be used automatically. 
Also included are models that have already been trained. 
The model `final-models/model-run-UNET-2023-06-30 00:17:11.130331.pt` is the selected final model (purple in the report plots).
It is loaded and used in `star_analysis/experiments/visualizations.ipynb` for predictions and visualizations.

## General SDSS Information

### Glossary and Specifications
- SDSS: Sloan Digital Sky Survey. This is our data source.
- DR: Data Release. We selected DR17.
- Rerun: A rerun is a reprocessing of the data. We selected 301.
- Fields: A field is a region of the sky.
- Run-camcol-field identifier: A unique identifier for a field. I.e. 3704-3-91.
- Frame: Corresponding to a single filter (u, g, r, i, z) image of a field.
- Filter: One of the five filters (u, g, r, i, z) used by SDSS.
- Object: A star, galaxy, or other astronomical object.
- FITS: Flexible Image Transport System. A file format used to store astronomical data.

### Data
We are using the following data:
- FITS files for the five filters (u, g, r, i, z) for each field.
- DR17 rerun 301.
- Fields.csv: A CSV file containing the run-camcol-field identifiers (rcf-id) for each field.
- We can download the FITS files via the rcf-ids in Fields.csv.
- Fields are overlapping by 10%. We should skip one Field between subsets.

#### Alignement between:
- IRG: i, r, g as jpg. (frame-irg-001000-1-0027.jpg)
- u: u as FITS. (frame-u-001000-1-0027.fits)
- z: z from FITS. (frame-z-001000-1-0027.fits)
- Labels: ?

#### Download
Use syntax as described in https://www.sdss4.org/dr17/data_access/bulk/ .


## SDSS Information
https://www.sdss4.org/dr17/help/glossary/
https://www.sdss4.org/dr14/imaging/imaging_basics/
https://dr12.sdss.org/fields/raDec?ra=143&dec=15
