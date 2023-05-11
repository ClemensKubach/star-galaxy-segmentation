mkdir data

cd data
mkdir images

cd images
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/3918/3/frame-irg-003918-3-0213.jpg
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/3918/3/frame-r-003918-3-0213.fits.bz2
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-g-003918-3-0213.fits.bz2
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-i-003918-3-0213.fits.bz2
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-u-003918-3-0213.fits.bz2
wget https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-z-003918-3-0213.fits.bz2

cd ..

mkdir labels
cd labels
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-003918-3-gal.fits.gz
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-003918-3-star.fits.gz
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-003918-3-sky.fits.gz

