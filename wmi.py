import math
from PIL import Image
from util import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt,wiener
from scipy.optimize import curve_fit
from skimage.io import imread
from skimage.morphology import disk
import os
import warnings
import cv2
from fmgaussfit_WMI import *
from calcphys import *
from npintensity import *
warnings.filterwarnings("ignore", category=UserWarning)

from params import picname,input_file,ncpos,showfig,runstitch,fillhole,numheadrow,numheadcol,runfilt,gfsize,gfsigma,np_size
from params import pxscale,runphys,runint,icutrad,bkgwidth,ficfilt,threshlim,numincthresh,N,fit_range,numit
from params import cell_size,numit,gfimg,nump,height,width,N,fappend

if showfig:
    plt.figure('Raw ADF Image')
    plt.imshow(input_file, cmap='gray')
    plabels = [str(n+1) for n in range(nump)]
    plt.plot(ncpos[:, 0], ncpos[:, 1], 'r.')
    for i, (x, y) in enumerate(ncpos):
        plt.text(x, y - 7, plabels[i], color='y', fontweight='bold', ha='center', backgroundcolor='none')
    plt.show()

# ===========(2D Gaussian Fitting of each NP to find NP centers)============
physrad = np.zeros(nump)  # Physical radius to use
cucmin = np.zeros(nump, dtype=int)
cucmax = np.zeros(nump, dtype=int)
curmin = np.zeros(nump, dtype=int)
curmax = np.zeros(nump, dtype=int)

# Define the bounds of NP cell for Gaussian fitting
for inc in range(nump):
    cucmin[inc] = int(round(ncpos[inc, 0] - fit_range))
    cucmax[inc] = int(round(ncpos[inc, 0] + fit_range))
    curmin[inc] = int(round(ncpos[inc, 1] - fit_range))
    curmax[inc] = int(round(ncpos[inc, 1] + fit_range))

    # Ensure bounds are within image dimensions
    cucmin[inc] = max(cucmin[inc], 0)
    cucmax[inc] = min(cucmax[inc], width - 1)
    curmin[inc] = max(curmin[inc], 0)
    curmax[inc] = min(curmax[inc], height - 1)
print("cucmin",cucmin)
print("cucmax",cucmax)

# Initialize storage arrays for fitting inputs and outputs
gfx = []
gfy = []
gfz = []
gfitres = []
zfit = []
zfitheight = np.zeros(nump)
zfitwidth = np.zeros(nump)
fitmax = np.zeros((nump, 2))
fitcent = np.zeros((nump, 2))
print(type(nump))  # This will show the actual type of nump
print(nump)
# Perform 2D Gaussian fitting for each NP
for inc in range(nump):
    #print("test1")
    gfx_inc = np.arange(0,cucmax[inc]-cucmin[inc]+1)
    gfy_inc = np.arange(0,curmax[inc]-curmin[inc]+1)
    npcell = gfimg[curmin[inc]:curmax[inc]+1, cucmin[inc]:cucmax[inc]+1]
    #print("cuc",cucmax[inc],cucmin[inc],gfimg.shape,npcell.shape)
    npcell[npcell == 0] = np.min(npcell[npcell > 0])
    gfz_inc = npcell - np.min(npcell)
    try:
        fitresult, data_fitted, _,_,_,_=fmgaussfit_edit(gfx_inc, gfy_inc, gfz_inc, ncpos[inc,0]-cucmin[inc], ncpos[inc,1]-curmin[inc])
        gfitres.append(fitresult)
        zfit.append(data_fitted)
        zfitheight[inc], zfitwidth[inc] = data_fitted.shape
        zfitheight[inc], zfitwidth[inc] = data_fitted.shape

        # Find the center of the 2D Gaussian fit
        rfitmax, cfitmax = np.unravel_index(np.argmax(data_fitted), data_fitted.shape)
        fitmax[inc, :] = [cfitmax, rfitmax]
        fitcent[inc, :] = [cucmin[inc] + cfitmax, curmin[inc] + rfitmax]

        # Adjust if the fitted center is too far from the initial center
        fitcentdist = np.sqrt((ncpos[inc, 0] - fitcent[inc, 0])**2 + (ncpos[inc, 1] - fitcent[inc, 1])**2)
        if fitcentdist > (np_size / 2):
            fitcent[inc, :] = ncpos[inc, :]
    except RuntimeError:
        print(f"Error - curve_fit failed for NP {inc + 1}")
        fitcent[inc, :] = ncpos[inc, :]

# =========================(Calculate Physical Radii)==========================
firstfig=1
plabels=[]
if runphys == 1:
    physrad=calculate_physical_radii(zfit, fitmax, zfitwidth, zfitheight, fitcent, plabels, firstfig, gfx, gfy, curmin, curmax, cucmin, cucmax)
    print("test")
if runint==1:
    nctable=integrate_np_intensity(fitcent,physrad)
    print("test2")
    nctable.to_csv(f'pyres_{fappend}.csv')
