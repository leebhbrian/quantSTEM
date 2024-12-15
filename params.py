import numpy as np
from util import *
from scipy.ndimage import gaussian_filter, rotate, convolve

fappend="01_03"
picname=f"/home/blee2/WMI/WMI_python/Au144/Raw_dm3/{fappend}_CL8cm_Spot0.2nm_CAp3_B3675_C2055_4000kX"
input_file = tif_to_numpy(picname+".tiff")
posname=f"/home/blee2/WMI/WMI_python/Au144/NP_Centers/{fappend}.txt"
ncpos=np.loadtxt(posname,skiprows=1, usecols=(1, 2)).astype(int)
ncpos=ncpos[:3,:]
#Debuggers
showfig=False

#=====(Data Input & Preparation)=====
runstitch = 0 # Set to 1 if stitching is required
fillhole = 0  # Fill holes in the image

numheadrow = 0 # Number of header rows in data files
numheadcol = 0 # Number of header columns in data files

runfilt = 0   # Run Gaussian filtering

gfsize = 3    # Gaussian filter size
gfsigma = 1   # Gaussian filter sigma
#=====(Input Parameters for Analysis)=====
np_size = 27  # Example nanoparticle size

pxscale = 0.05708

runphys = 1      # Calculate physical radii
runint = 1       # Calculate integrated intensity

icutrad = 3   # Initial cut-off radius
bkgwidth = 2  # Background width

ficfilt = 3    #Order of median filter used on the intensity change data for finding cut-off radius
threshlim = 0.1 # Threshold limit for intensity change
numincthresh = 3 # Number of consecutive iterations below threshold

N = 2         # For physical radii calculation

# Calculate parameters
fit_range = round(2 * (np_size / 2))  # Size of region used for Gaussian fitting
numit = round(2.5 * (np_size / 2))    # Number of integration radius expansion iterations
cell_size = round(3.5 * np_size)      # Size of cell around NP extracted for stitching

if (icutrad + numit + bkgwidth) > (cell_size / 2):
    numit = int(np.floor(cell_size / 2 - bkgwidth - icutrad))

# Fill holes in the image
if fillhole == 1:
    input_file[input_file == 0] = np.mean(np.percentile(input_file, 5))
# Gaussian filtering
if runfilt == 1:
    #wfimg5 = custom_wiener(input_file, (5, 5))
    wfimg5=np.copy(input_file)
    gfimg = gaussian_filter(wfimg5, sigma=gfsigma)
elif runfilt==2:
    gaussian_kernel_1d = cv2.getGaussianKernel(gfsize, gfsigma)  # 1D Gaussian kernel
    gbif = gaussian_kernel_1d @ gaussian_kernel_1d.T  # Create 2D Gaussian kernel by outer product
    # Apply the Gaussian filter to wfimg5 using convolution
    gfimg = convolve(input_file, gbif)
else:
    gfimg = input_file.copy()
# Shift image baseline
input_file -= np.min(input_file)
gfimg -= np.min(gfimg)
np.save(f"input_file_{fappend}.npy",input_file)
np.save(f"gfimg_{fappend}.npy",gfimg)
#print("gfimg",input_file.shape,gfimg.shape)
# =================(Initial display of input information)===================
nump = ncpos.shape[0]
height, width = input_file.shape