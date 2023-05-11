mkdir data

cd data
mkdir images

cd images
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/3918/3/frame-irg-003918-3-0213.jpg -nc
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/frame-r-008162-6-0080.fits.bz2 -nc
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/frame-g-008162-6-0080.fits.bz2 -nc
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/frame-i-008162-6-0080.fits.bz2 -nc
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/frame-u-008162-6-0080.fits.bz2 -nc
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/frame-z-008162-6-0080.fits.bz2 -nc

cd ..

mkdir calibration
cd calibration
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-003918-3-gal.fits.gz -nc
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-003918-3-star.fits.gz -nc
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-003918-3-sky.fits.gz -nc

