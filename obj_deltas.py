#loading all the libraries we will be using.
import numpy as np #We will make extensive use of Numpy arrays 
import multiprocessing as mp #Enable multiprocessing
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
import math #needed to check for NaNs

import matplotlib  #ploting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #for better display of FITS images
import matplotlib.image as mpimg
import time as t

import neossatlib as neo

#for PCA analysis 
import scipy.linalg.lapack as la
import scipy.optimize as op
	
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import aperture_photometry

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

workdir='281/'                #Directory that contains cleaned FITS files (don't forget the trailing '/')
fileslist=workdir+'files_clean.list'     #Simple text file that contains the names of all the FITS files to process.   
bpix=-1.0e10                   #value to mark bad pixels. Any pixel *below* bpix is considered invalid.
sigscalel=1.0                  #low bounds for display clipping.  Keep small to keep background 'dark'
sigscaleh=1.0                  #high bounds for display clipping.
nprocessor=8                  #Number of processors to use for data processing. 
#--- Parameters for photometry ---#
zmag=21.46                      #zero-point for instrumental magnitude (selected by eye, don't trust my value)
photap=4                 #photometric aperture

#Should get this from Header, but here's the values for NEOSSat 
gain=1.1 #e/ADU
readnoise=8 #e/pixel
imagefiles=neo.read_file_list(fileslist)

lightlist_in=[]
jddate=[]
exptime=[]
ra =[]
dec=[]
rol=[]
nfiles=len(imagefiles)

# Extract FITS data
photometry_data = {}
no_stars = []
print('Light, fixed-point files:')
for i in range(nfiles):
	filename=workdir+imagefiles[i]

	# Grab metadata
	hdulist = fits.open(filename)
	shutter=hdulist[0].header['SHUTTER']
	mode=hdulist[0].header['MODE']
	# Shutter open, fixed-point mode only
	if (int(shutter[0]) == 0) & (int(mode[0:2]) == 16):
		print(filename)

		lightlist_in.append(imagefiles[i])
		jddate.append(float(hdulist[0].header['JD-OBS']))
		temp_ra = hdulist[0].header['OBJCTRA']
		ra.append(Angle(temp_ra+' hours').degree)
		temp_dec = hdulist[0].header['OBJCTDEC']
		dec.append(Angle(temp_dec+' degrees').degree)
		rol.append(float(hdulist[0].header['OBJCTROL']))
		exptime.append(float(hdulist[0].header['EXPOSURE']))
	hdulist.close()

jddate=np.array(jddate)
exptime=np.array(exptime)
print("number of images: ",len(lightlist_in))

idx=np.argsort(jddate)
jddate=jddate[idx]
exptime=exptime[idx]
lightlist=[]
for i in idx:
	lightlist.append(lightlist_in[i])

# Object recognition for light, fixed-point images
for i, im in enumerate(lightlist):
    filename = workdir + im
    
    # Crop image
    trim,btrim,xsc,ysc,xov,yov=neo.getimage_dim(filename)
    scidata_raw=neo.read_fitsdata(filename)
    sh=scidata_raw.shape
    strim=np.array([sh[0]-xsc,sh[0],sh[1]-ysc,sh[1]])
    scidata=np.copy(scidata_raw[strim[0]:strim[1],strim[2]:strim[3]])
    imstat=neo.imagestat(scidata,bpix)
    
    # Object recognition
    mean, median, std = sigma_clipped_stats(scidata, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=2.0, threshold=35*std)
    sources = daofind(scidata - median)

    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=photap)
    phot_table = aperture_photometry(scidata-median, apertures)

    nstar=len(phot_table['aperture_sum'])

    for j in range(nstar):
        photometry_data[i,j] = [phot_table['xcenter'][j], phot_table['ycenter'][j]]
        temp_sum = j
    no_stars.append(temp_sum)

# Chose two images to compare
img_idx = [0, 1]

# compare first two imgs
img1 = []
img2 = []
for key in photometry_data.keys():
    if key[0] == img_idx[0]:
        img1.append([key[1], photometry_data[key][0], photometry_data[key][1]])
    if key[0] == img_idx[1]:
        img2.append([key[1], photometry_data[key][0], photometry_data[key][1]])
#So: img1 = [star num, x pos, y pos]
print()
print("Images to be compared:")
print(lightlist[img_idx[0]])
print(lightlist[img_idx[1]])

if len(img1) > len(img2):    # img1 is the shorter array
    img1, img2 = img2, img1  # ohh yeah big brain time in place swap

# Get distances between all nodes
pairs = {}

# Iterate over all stars in shorter list
for star1 in img1:
	# Compare to all stars in longer list
	for star2 in img2:
		dx = star1[1] - star2[1]
		dy = star1[2] - star2[2]
		dist = np.sqrt(dx**2 + dy**2)
		pairs[(star1[0], star2[0])] = dist / u.pix

# ID of closest star
closest_star = [0 for star in img1]
# Distance between pair - assume initial arbitrary distance of 2048
closest_star_dist = [2048 for star in img1]
for pair in pairs:
    # For each pair, compare distances and get closest pair
    if pairs[pair] < closest_star_dist[pair[0]]:
        # Assign closer neighbour
        closest_star[pair[0]] = pair[1]
        # Update distance
        closest_star_dist[pair[0]] = pairs[pair]

# Find z-score of distances
closest_star_dist_z = closest_star_dist / np.std(closest_star_dist)

# Get outliers
threshold_sigma = 3
outlier_dist = closest_star_dist_z > threshold_sigma

# Get pair coordinate
X = [[], []]
Y = [[], []]
for star1, star2 in enumerate(closest_star):
    # Grab star coordinates
    x1 = img1[star1][1] / u.pix
    x2 = img2[star2][1] / u.pix
    y1 = img1[star1][2] / u.pix
    y2 = img2[star2][2] / u.pix
    
    # Save to memory
    X[0].append(x1)
    X[1].append(x2)
    Y[0].append(y1)
    Y[1].append(y2)
# Gotta love numpy, eh?
X = np.array(X)
Y = np.array(Y)

# Grab only images we want to compare
compared_imgs = [fname for i, fname in enumerate(lightlist) if i in img_idx]
figs = []

for i, im in enumerate(compared_imgs):
    # Read image data
    filename = workdir + im
    
    # Crop image
    trim,btrim,xsc,ysc,xov,yov=neo.getimage_dim(filename)
    scidata_raw=neo.read_fitsdata(filename)
    sh=scidata_raw.shape
    strim=np.array([sh[0]-xsc,sh[0],sh[1]-ysc,sh[1]])
    scidata=np.copy(scidata_raw[strim[0]:strim[1],strim[2]:strim[3]])
    imstat=neo.imagestat(scidata,bpix)
    
    # Plot image
    figs.append(plt.figure(i, figsize = (20, 20)))
    plt.imshow(scidata)
    
    # Plot movements
    plt.scatter(X[0], Y[0], facecolors='none', edgecolors='r', s=80)
    plt.scatter(X[1], Y[1], facecolors='none', edgecolors='g', s=80)
    plt.plot([X[0], X[1]], [Y[0], Y[1]])
    
    # Label movements
    dist = np.sqrt((X[0] - X[1])**2 + (Y[0] - Y[1])**2)
    for j, label in enumerate(dist):
        label = "delta = " + str(np.around(label, 1) * u.pix) + \
                "\nz = " + str(np.around(closest_star_dist_z[j], 1))
        if outlier_dist[j]:
            plt.annotate(label, (X[0][j] + 5, Y[0][j] + 5), color = 'r')
        else:
            plt.annotate(label, (X[0][j] + 5, Y[0][j] + 5), color = 'w')

    # Apply image metadata
    subtitle = 'RA = ' + str(Angle(ra[i], u.deg).to_string(unit=u.hourangle)) + \
    		   ', DEC = ' + str(Angle(dec[i], u.deg).to_string(unit=u.deg)) + \
    		   ', ROL = ' + str(rol[i] * u.deg)
    plt.title(filename)
    plt.suptitle(subtitle, fontsize = 12)
    plt.xlabel('x [pix]')
    plt.ylabel('y [pix]')

    figs[i].show()
input()