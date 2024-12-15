import numpy as np
from skimage.morphology import disk
from scipy.signal import medfilt
import pandas as pd

from params import picname,input_file,ncpos,showfig,runstitch,fillhole,numheadrow,numheadcol,runfilt,gfsize,gfsigma,np_size
from params import pxscale,runphys,runint,icutrad,bkgwidth,ficfilt,threshlim,numincthresh,N,fit_range,numit
from params import cell_size,numit,gfimg,nump,height,width,N

def integrate_np_intensity(fitcent,physrad):
    """
    Converts the provided MATLAB code into Python.

    Parameters:
    - input_file: ndarray
        The image data to be analyzed.
    - fitcent: ndarray
        Center coordinates of each nanoparticle (nump x 2 array).
    - icutrad: int
        Initial cutoff radius.
    - numit: int
        Number of iterations.
    - bkgwidth: int
        Background width.
    - nump: int
        Number of nanoparticles.
    - ficfilt: int
        Window size for median filter (should be an odd integer).

    Returns:
    - cluspxsums, cluspxnums, cluspxmeans, cluspxmaxs, bkgpxsums, bkgpxnums,
      bkgpxmeans, intchanges, mfintchanges, clusavgint, ringavgint
    """
    height, width = input_file.shape

    cluspxsums = np.zeros((numit + bkgwidth, nump))
    cluspxnums = np.zeros((numit + bkgwidth, nump))
    cluspxmeans = np.zeros((numit + bkgwidth, nump))
    cluspxmaxs = np.zeros((numit + bkgwidth, nump))
    bkgpxsums = np.zeros((numit, nump))
    bkgpxnums = np.zeros((numit, nump))
    bkgpxmeans = np.zeros((numit, nump))
    intchanges = np.zeros((numit + bkgwidth, nump))
    mfintchanges = np.zeros((numit + bkgwidth, nump))
    clusavgint = np.zeros((numit + bkgwidth, nump))
    ringavgint = np.zeros((numit + bkgwidth, nump))
    IntMaskNh = [None] * (numit + bkgwidth)
    NhCenter = [None] * (numit + bkgwidth)

    # Round fitcent to nearest pixel
    fitcentRound = np.round(fitcent).astype(int)
    fitcentRound[fitcentRound == 0] = 1  # Adjust zeros to ones (since index 0 is valid in Python)

    # Create integration masks
    cutrad = icutrad
    for i in range(numit + bkgwidth):
        IntMask = disk(cutrad)
        IntMaskNh[i] = IntMask
        NhCenter[i] = ((np.array(IntMask.shape) + 1) // 2)
        cutrad += 1

    # Loop over each nanoparticle
    for j in range(nump):
        # print(f'Analyzing Intensities of NP #{j+1}/{nump}')
        cutrad = icutrad

        for i in range(numit + bkgwidth):
            ImgForInt = np.zeros((height, width))

            NhRowT = 0
            NhRowB = 2 * cutrad + 1
            NhColL = 0
            NhColR = 2 * cutrad + 1

            RowTop = fitcentRound[j, 1] - cutrad
            RowBot = fitcentRound[j, 1] + cutrad
            ColLeft = fitcentRound[j, 0] - cutrad
            ColRight = fitcentRound[j, 0] + cutrad

            # Adjust for boundaries
            if RowTop < 0:
                NhRowT = -RowTop
                RowTop = 0
            if RowBot >= height:
                NhRowB = NhRowB - (RowBot - (height - 1))
                RowBot = height - 1
            if ColLeft < 0:
                NhColL = -ColLeft
                ColLeft = 0
            if ColRight >= width:
                NhColR = NhColR - (ColRight - (width - 1))
                ColRight = width - 1

            # Extract the mask and the image segment
            TruncIntMaskNh = IntMaskNh[i][NhRowT:NhRowB + 1, NhColL:NhColR + 1]
            ImgSegment = input_file[RowTop:RowBot + 1, ColLeft:ColRight + 1]

            # Multiply mask and image segment
            ImgForInt[RowTop:RowBot + 1, ColLeft:ColRight + 1] = ImgSegment * TruncIntMaskNh

            cluspxnums[i, j] = np.sum(TruncIntMaskNh)
            cluspxsums[i, j] = np.sum(ImgForInt)
            cluspxmaxs[i, j] = np.max(ImgForInt)

            if i >= bkgwidth:
                idx = i - bkgwidth
                bkgpxsums[idx, j] = cluspxsums[i, j] - cluspxsums[idx, j]
                bkgpxnums[idx, j] = cluspxnums[i, j] - cluspxnums[idx, j]
                if bkgpxnums[idx, j] != 0:
                    bkgpxmeans[idx, j] = bkgpxsums[idx, j] / bkgpxnums[idx, j]
                else:
                    bkgpxmeans[idx, j] = 0

                if cluspxnums[idx, j] != 0:
                    clusavgint[idx, j] = cluspxsums[idx, j] / cluspxnums[idx, j]
                else:
                    clusavgint[idx, j] = 0

                if idx == 0:
                    ringavgint[idx, j] = clusavgint[idx, j]
                else:
                    num_diff = cluspxnums[idx, j] - cluspxnums[idx - 1, j]
                    sum_diff = cluspxsums[idx, j] - cluspxsums[idx - 1, j]
                    if num_diff != 0:
                        ringavgint[idx, j] = sum_diff / num_diff
                    else:
                        ringavgint[idx, j] = 0

                cluspxsums[idx, j] = cluspxsums[idx, j] - bkgpxmeans[idx, j] * cluspxnums[idx, j]
                if cluspxnums[idx, j] != 0:
                    cluspxmeans[idx, j] = cluspxsums[idx, j] / cluspxnums[idx, j]
                else:
                    cluspxmeans[idx, j] = 0

                cluspxmaxs[idx, j] = cluspxmaxs[idx, j] - bkgpxmeans[idx, j]

                if idx == 0:
                    intchanges[idx, j] = 1
                else:
                    denom = cluspxsums[idx - 1, j]
                    if denom != 0:
                        intchanges[idx, j] = (cluspxsums[idx, j] - cluspxsums[idx - 1, j]) / denom
                    else:
                        intchanges[idx, j] = 0

            cutrad += 1

        # Apply median filter to intchanges
        mfintchanges[:, j] = medfilt(intchanges[:, j], kernel_size=ficfilt)
    # Initialize arrays
    cutoffnum = np.zeros(nump, dtype=int)       # The cut-off iteration to use
    cutoffradii = np.zeros(nump)                # The value of the cut-off radii to use
    ncint = np.zeros(nump)                      # Background-corrected integrated intensity for each NP

    # The cutoff radius will be prevented from being smaller than this (number of iterations)
    cutstart = int(np.floor(np_size / 3)) - icutrad

    # Ensure cutstart is not negative
    cutstart = max(cutstart, 0)

    # Loop through each nanoparticle
    for j in range(nump):
        found = False  # Flag to check if the threshold condition is met
        i = None       # Initialize i

        if numincthresh == 1:
            # Looks for the first iteration to drop below the threshold
            for i in range(cutstart, numit):
                if abs(mfintchanges[i, j]) <= threshlim:
                    found = True
                    break
            if not found:
                i = numit - 1  # Use the last iteration if threshold not met
        elif numincthresh == 2:
            # Looks for the first two sequential iterations to drop below the threshold
            for i in range(cutstart, numit - 1):
                if (abs(mfintchanges[i, j]) <= threshlim) and (abs(mfintchanges[i + 1, j]) <= threshlim):
                    found = True
                    break
            if not found:
                i = numit - 2  # Use the second last iteration if threshold not met
        else:
            # Looks for the first three sequential iterations to drop below the threshold
            for i in range(cutstart + 1, numit - 1):
                if (abs(mfintchanges[i - 1, j]) <= threshlim) and (abs(mfintchanges[i, j]) <= threshlim) and (abs(mfintchanges[i + 1, j]) <= threshlim):
                    found = True
                    break
            if not found:
                i = numit - 2  # Use the second last iteration if threshold not met

        # Assign values to arrays
        cutoffnum[j] = i
        cutoffradii[j] = icutrad + i - 1
        ncint[j] = cluspxsums[i, j]  # Background-corrected integrated intensity

    # Generate the results table
    ncnum = np.arange(1, nump + 1)  # NP numbers from 1 to nump

    if (runint == 1) and (runphys == 1):
        cutoffradiinm = cutoffradii * pxscale
        physradnm = physrad * pxscale
        data = {
            'NP': ncnum,
            'Integrated_Intensity': ncint,
            'Integration_Radius_nm': cutoffradiinm,
            'Avg_Physical_Radius_nm': physradnm
        }
        nctable = pd.DataFrame(data)
    elif (runint == 1) and (runphys != 1):
        cutoffradiinm = cutoffradii * pxscale
        data = {
            'NP': ncnum,
            'Integrated_Intensity': ncint,
            'Integration_Radius_nm': cutoffradiinm
        }
        nctable = pd.DataFrame(data)
    elif (runint != 1) and (runphys == 1):
        physradnm = physrad * pxscale
        data = {
            'NP': ncnum,
            'Avg_Physical_Radius_nm': physradnm
        }
        nctable = pd.DataFrame(data)
    else:
        data = {'NP': ncnum}
        nctable = pd.DataFrame(data)

    return nctable

    #return (cluspxsums, cluspxnums, cluspxmeans, cluspxmaxs, bkgpxsums, bkgpxnums,bkgpxmeans, intchanges, mfintchanges, clusavgint, ringavgint)
