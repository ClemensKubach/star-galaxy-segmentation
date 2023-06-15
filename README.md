# star_analysis

## Glossary and Specifications
- SDSS: Sloan Digital Sky Survey. This is our data source.
- DR: Data Release. We selected DR17.
- Rerun: A rerun is a reprocessing of the data. We selected 301.
- Fields: A field is a region of the sky.
- Run-camcol-field identifier: A unique identifier for a field. I.e. 3704-3-91.
- Frame: Corresponding to a single filter (u, g, r, i, z) image of a field.
- Filter: One of the five filters (u, g, r, i, z) used by SDSS.
- Object: A star, galaxy, or other astronomical object.
- FITS: Flexible Image Transport System. A file format used to store astronomical data.

## Data
We are using the following data:
- FITS files for the five filters (u, g, r, i, z) for each field.
- DR17 rerun 301.
- Fields.csv: A CSV file containing the run-camcol-field identifiers (rcf-id) for each field.
- We can download the FITS files via the rcf-ids in Fields.csv.
- Fields are overlapping by 10%. We should skip one Field between subsets.

### Alignement between:
- IRG: i, r, g as jpg. (frame-irg-001000-1-0027.jpg)
- u: u as FITS. (frame-u-001000-1-0027.fits)
- z: z from FITS. (frame-z-001000-1-0027.fits)
- Labels: ?

### Download
Use syntax as described in https://www.sdss4.org/dr17/data_access/bulk/ .


# SDSS Information
https://www.sdss4.org/dr17/help/glossary/
https://www.sdss4.org/dr14/imaging/imaging_basics/
https://dr12.sdss.org/fields/raDec?ra=143&dec=15
