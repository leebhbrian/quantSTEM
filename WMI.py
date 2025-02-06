from ncempy.io import dm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from skimage import filters, measure
from skimage.morphology import disk,remove_small_objects
import os
from matplotlib.patches import Circle
import argparse
from PIL import Image


def gaussian2d(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """
    2D Gaussian function for curve_fit.
    coords: a tuple (x, y) of numpy arrays
    """
    x, y = coords
    inner = ((x - xo)**2)/(2*sigma_x**2) + ((y - yo)**2)/(2*sigma_y**2)
    return offset + amplitude * np.exp(-inner)

def fmgaussfit_WMI(gfz, init_x, init_y):
    """
    Fit a 2D Gaussian to subimage gfz.
    init_x, init_y are the initial guesses (in the subimage coordinate system).
    Returns:
       popt: fitted parameters [amplitude, x0, y0, sigma_x, sigma_y, offset]
       zzfit: the fitted 2D Gaussian surface (same shape as gfz)
    """
    m, n = gfz.shape
    x = np.arange(n)
    y = np.arange(m)
    x_mesh, y_mesh = np.meshgrid(x, y)
    # initial guess: amplitude, x0, y0, sigma_x, sigma_y, offset
    initial_guess = (gfz.max(), init_x, init_y, n/4, m/4, gfz.min())
    try:
        popt, _ = curve_fit(gaussian2d, (x_mesh.ravel(), y_mesh.ravel()),
                            gfz.ravel(), p0=initial_guess)
    except RuntimeError:
        print("Warning: Gaussian fit did not converge; using initial guess.")
        popt = initial_guess
    zzfit = gaussian2d((x_mesh, y_mesh), *popt).reshape(m, n)
    return popt, zzfit

def find_profile_intersection(profile, target, N):
    """
    Given a 1D profile and a target value (max(profile)/N), 
    use linear interpolation around the index where the absolute difference is minimized.
    Returns a (possibly fractional) position along the profile.
    """
    idx = np.argmin(np.abs(profile - target))
    # Check for interpolation:
    if profile[idx] < target:
        if idx < len(profile) - 1:
            frac = (target - profile[idx]) / (profile[idx+1] - profile[idx] + 1e-12)
            pos = idx + frac
        else:
            pos = idx
    else:
        if idx > 0:
            frac = (profile[idx] - target) / (profile[idx] - profile[idx-1] + 1e-12)
            pos = idx - frac
        else:
            pos = idx
    return pos

def calc_npintensity(parser):
    args=parser.parse_args()
    inpath=args.inpath
    outpath=args.outpath
    fname=args.fname
    ftype=args.ftype
    calc_center=args.calc_center
    output_radius=args.output_radius
    fillhole=args.fillhole
    runfilt=args.runfilt
    get_histogram=args.get_histogram
    runint=args.runint
    runphys=args.runphys
    
    os.makedirs(outpath,exist_ok=True)
    pic_filename = f'{inpath}/{fname}.{ftype}'
    ncpos_filename = f'{inpath}/{fname}.csv'
    
    if runfilt:
        gfsigma  = 1.0

    # Analysis input parameters:
    np_size  = 27         # approximate NP size (diameter, in pixels)
    pxscale  = 0.05708    # pixel scale (nm/px)

    icutrad  = 3         # initial integration cutoff radius (in px)
    bkgwidth = 2         # annulus width for background determination

    ficfilt  = 3         # order (kernel size) of median filter for intensity change data
    threshlim= 0.1       # threshold (fraction, 0-1) for intensity change to set cut-off radius
    numincthresh = 3     # number of sequential increments that must fall below the threshold

    N = 2                # initial N for FW(1/N)M (e.g. N=2 for FWHM)

    # Derived parameters:
    fit_range = int(round(2 * (np_size/2)))  # size of region used for Gaussian fitting (roughly np_size)
    numit     = int(round(2.5 * (np_size/2)))  # number of integration radius iterations
    cell_size = int(round(3.5 * np_size))       # size of cell extracted around NP

    # Ensure integration expansion does not exceed half cell size:
    if (icutrad + numit + bkgwidth) > (cell_size / 2):
        numit = int(np.floor(cell_size/2 - bkgwidth - icutrad))

    # ======================= Define helper functions =============================



    # ======================= Main Processing =====================================

    # --- Read in the image ---
    if ftype=='dm3':
        input_file = dm.dmReader(pic_filename)['data']
    else:
        input_img=Image.open(pic_filename)
        input_file=np.array(input_img)

    # If the image has more than 2 dimensions (e.g. RGB), take one channel or convert to grayscale:
    if input_file.ndim > 2:
        input_file = input_file[...,0]


    height, width = input_file.shape

    # --- Fill holes if requested ---
    if fillhole:
        # Replace 0-intensity pixels with the 5th percentile of nonzero pixels
        nonzero_vals = input_file[input_file > 0]
        if nonzero_vals.size > 0:
            fill_val = np.percentile(nonzero_vals, 5)
            input_file[input_file==0] = fill_val

    # --- Create a Gaussian-filtered version for the fitting ---
    if runfilt:
        gfimg = gaussian_filter(input_file, sigma=gfsigma)
    else:
        gfimg = input_file.copy()

    # --- Shift baseline so that minimum = 0 ---
    input_file = input_file - input_file.min()
    gfimg = gfimg - gfimg.min()

    if calc_center:
        ncpos_output_filename=f'{outpath}/center_calc_{fname}.csv'
        thresh = filters.threshold_otsu(gfimg)
        binary = gfimg > thresh
        binary_clean = remove_small_objects(binary, min_size=20)
        label_image = measure.label(binary_clean)
        regions = measure.regionprops(label_image)
        centers = np.array([r.centroid for r in regions])
        centers_xy = centers[:, [1, 0]]
        np.savetxt(ncpos_output_filename,centers_xy,delimiter=',')
        ncpos = np.round(centers_xy).astype(int)
    else:
        ncpos = np.loadtxt(ncpos_filename, delimiter=',', skiprows=0)
        ncpos = np.round(ncpos).astype(int)
    nump = ncpos.shape[0]
    
    ### 2D Gaussian Fitting for Each NP
    fitcent = np.zeros((nump,2), dtype=float)   # refined NP centers (x,y) in image coordinates
    gfres_list = []  # list to store fitted parameters for each NP
    zzfit_list = []  # list to store fitted Gaussian surfaces

    # Pre-calculate the bounds for each NP’s cell used for fitting:
    cucmin_arr = np.zeros(nump, dtype=int)
    cucmax_arr = np.zeros(nump, dtype=int)
    curmin_arr = np.zeros(nump, dtype=int)
    curmax_arr = np.zeros(nump, dtype=int)

    for i in range(nump):
        # For each NP, set boundaries (note: careful with Python’s 0-indexing)
        cucmin = ncpos[i,0] - fit_range
        cucmax = ncpos[i,0] + fit_range
        curmin = ncpos[i,1] - fit_range
        curmax = ncpos[i,1] + fit_range
        cucmin = max(cucmin, 0)
        cucmax = min(cucmax, width-1)
        curmin = max(curmin, 0)
        curmax = min(curmax, height-1)
        cucmin_arr[i] = cucmin
        cucmax_arr[i] = cucmax
        curmin_arr[i] = curmin
        curmax_arr[i] = curmax

    # Loop through each NP and perform Gaussian fitting:
    for i in range(nump):
        # Prepare coordinate ranges for the sub-image:
        x_range = np.arange(cucmax_arr[i] - cucmin_arr[i] + 1)
        y_range = np.arange(curmax_arr[i] - curmin_arr[i] + 1)
        # Extract the NP cell from the Gaussian-filtered image.
        npcell = gfimg[curmin_arr[i]:curmax_arr[i]+1, cucmin_arr[i]:cucmax_arr[i]+1].copy()
        # Replace any 0 values with the minimum positive value in npcell:
        nonzero = npcell[npcell > 0]
        if nonzero.size > 0:
            npcell[npcell==0] = nonzero.min()
        # Subtract the minimum to use as background offset
        gfz = npcell - npcell.min()
        # Initial guess for the center in the sub-image coordinates:
        init_x = ncpos[i,0] - cucmin_arr[i]
        init_y = ncpos[i,1] - curmin_arr[i]
        # Perform the Gaussian fit:
        popt, zzfit = fmgaussfit_WMI(gfz, init_x, init_y)
        gfres_list.append(popt)
        zzfit_list.append(zzfit)
        # Find the peak of the fitted Gaussian:
        rfitmax, cfitmax = np.unravel_index(np.argmax(zzfit), zzfit.shape)
        # Compute refined center (in full image coordinates)
        refined_x = cucmin_arr[i] + cfitmax
        refined_y = curmin_arr[i] + rfitmax
        # If the fitted center is far from the original, revert to original NP center:
        if np.sqrt((ncpos[i,0] - refined_x)**2 + (ncpos[i,1] - refined_y)**2) > (np_size/2):
            fitcent[i, :] = ncpos[i, :]
        else:
            fitcent[i, :] = [refined_x, refined_y]

    ### Calculate Physical Radii via FW(1/N)M
    physrad = np.zeros(nump, dtype=float)
    for i in range(nump):
        zzfit = zzfit_list[i]
        # Find peak indices (from the fitted image)
        rfitmax, cfitmax = np.unravel_index(np.argmax(zzfit), zzfit.shape)
        # Extract horizontal (x) and vertical (y) profiles through the peak.
        profile_x = zzfit[rfitmax, :]
        profile_y = zzfit[:, cfitmax]
        # Determine the target intensity = max/N (e.g. half maximum when N=2)
        target_x = profile_x.max() / N
        target_y = profile_y.max() / N
        # Find (possibly fractional) positions along each profile where the intensity is closest to target.
        pos_x = find_profile_intersection(profile_x, target_x, N)
        pos_y = find_profile_intersection(profile_y, target_y, N)
        # Physical radius: average distance (in pixels) from the fitted center to these positions.
        # Note: cfitmax is the x-location in the subimage and rfitmax is the y-location.
        # (We assume symmetry so we average the two estimates.)
        rad_est = (abs(cfitmax - pos_x) + abs(rfitmax - pos_y)) / 2.0
        physrad[i] = rad_est

    ### Intensity Integration and Background Correction
    if runint:
        # Preallocate arrays; dimensions: iterations x NP
        n_iterations = numit + bkgwidth
        cluspxsums   = np.zeros((n_iterations, nump))
        cluspxnums   = np.zeros((n_iterations, nump))
        cluspxmaxs   = np.zeros((n_iterations, nump))
        # For background annulus (computed only when i > bkgwidth)
        bkgpxsums    = np.zeros((numit, nump))
        bkgpxnums    = np.zeros((numit, nump))
        bkgpxmeans   = np.zeros((numit, nump))
        clusavgint   = np.zeros((numit, nump))
        ringavgint   = np.zeros((numit, nump))
        intchanges   = np.zeros((numit, nump))
        
        # Pre-calculate integration masks for each iteration (using skimage.morphology.disk).
        # We store each disk mask (as a boolean array) and its “center” (which is simply [radius, radius])
        mask_list = []
        for i in range(n_iterations):
            cutrad = icutrad + i
            mask = disk(cutrad)  # creates a flat (binary) disk of radius cutrad; shape = (2*cutrad+1, 2*cutrad+1)
            mask_list.append(mask)
        
        # For intensity integration, we use the refined NP centers rounded to nearest integer.
        fitcentRound = np.round(fitcent).astype(int)
        fitcentRound[fitcentRound < 0] = 0  # ensure indices are nonnegative
        
        for j in range(nump):
            for i in range(n_iterations):
                cutrad = icutrad + i
                mask = mask_list[i]
                mask_shape = mask.shape  # (mask_height, mask_width)
                mask_center = (cutrad, cutrad)
                # Determine bounds in the image for placing the mask centered at NP j.
                row_center = fitcentRound[j, 1]
                col_center = fitcentRound[j, 0]
                RowTop = row_center - cutrad
                RowBot = row_center + cutrad
                ColLeft = col_center - cutrad
                ColRight = col_center + cutrad
                # Determine the corresponding indices in the mask:
                mask_row_start = 0
                mask_col_start = 0
                mask_row_end = mask_shape[0]
                mask_col_end = mask_shape[1]
                # Adjust if the integration region goes off the image boundaries:
                if RowTop < 0:
                    mask_row_start = -RowTop
                    RowTop = 0
                if ColLeft < 0:
                    mask_col_start = -ColLeft
                    ColLeft = 0
                if RowBot >= height:
                    diff = RowBot - (height - 1)
                    mask_row_end = mask_shape[0] - diff
                    RowBot = height - 1
                if ColRight >= width:
                    diff = ColRight - (width - 1)
                    mask_col_end = mask_shape[1] - diff
                    ColRight = width - 1
                
                # Extract the corresponding submask and image region:
                submask = mask[mask_row_start:mask_row_end, mask_col_start:mask_col_end]
                subimage = input_file[RowTop:RowBot+1, ColLeft:ColRight+1]
                # Multiply subimage by submask:
                masked_image = subimage * submask
                # Store sums, pixel counts, and max:
                cluspxnums[i, j] = submask.sum()
                cluspxsums[i, j] = masked_image.sum()
                cluspxmaxs[i, j] = masked_image.max()
                
                # For iterations greater than the background width, compute background annulus:
                if i >= bkgwidth:
                    idx = i - bkgwidth
                    bkgpxsums[idx, j] = cluspxsums[i, j] - cluspxsums[i - bkgwidth, j]
                    bkgpxnums[idx, j] = cluspxnums[i, j] - cluspxnums[i - bkgwidth, j]
                    if bkgpxnums[idx, j] != 0:
                        bkgpxmeans[idx, j] = bkgpxsums[idx, j] / bkgpxnums[idx, j]
                    else:
                        bkgpxmeans[idx, j] = 0
                    # Calculate the average intensity in the NP (before background subtraction):
                    clusavgint[idx, j] = cluspxsums[i - bkgwidth, j] / cluspxnums[i - bkgwidth, j]
                    if idx == 0:
                        ringavgint[idx, j] = clusavgint[idx, j]
                    else:
                        ringavgint[idx, j] = (cluspxsums[i - bkgwidth, j] - cluspxsums[i - bkgwidth - 1, j]) / \
                                            (cluspxnums[i - bkgwidth, j] - cluspxnums[i - bkgwidth - 1, j] + 1e-12)
                    # Background correction: subtract background contribution
                    bcorr = bkgpxmeans[idx, j] * cluspxnums[i - bkgwidth, j]
                    cluspxsums[i - bkgwidth, j] = cluspxsums[i - bkgwidth, j] - bcorr
                    # For intensity change: define first change as 1, subsequent as relative change:
                    if idx == 0:
                        intchanges[idx, j] = 1.0
                    else:
                        prev = cluspxsums[i - bkgwidth - 1, j]
                        if prev != 0:
                            intchanges[idx, j] = (cluspxsums[i - bkgwidth, j] - prev) / prev
                        else:
                            intchanges[idx, j] = 0

        # --- Apply a median filter along the iteration axis for each NP:
        mfintchanges = np.zeros_like(intchanges)
        for j in range(nump):
            # Use a 1D median filter (kernel size = ficfilt) along iterations:
            mfintchanges[:, j] = median_filter(intchanges[:, j], size=ficfilt)

        # --- Determine the optimal cut-off radius for each NP ---
        cutoffnum = np.zeros(nump, dtype=int)
        cutoffradii = np.zeros(nump, dtype=int)
        ncint = np.zeros(nump)
        cutstart = max(int(np.floor(np_size/3)) - icutrad, 0)
        for j in range(nump):
            # Depending on the number of sequential iterations required:
            found = False
            if numincthresh == 1:
                for i in range(cutstart, numit):
                    if abs(mfintchanges[i, j]) <= threshlim:
                        cutoff = i
                        found = True
                        break
            elif numincthresh == 2:
                for i in range(cutstart, numit-1):
                    if (abs(mfintchanges[i, j]) <= threshlim and 
                        abs(mfintchanges[i+1, j]) <= threshlim):
                        cutoff = i
                        found = True
                        break
            else:
                for i in range(cutstart+1, numit-1):
                    if (abs(mfintchanges[i-1, j]) <= threshlim and
                        abs(mfintchanges[i, j]) <= threshlim and 
                        abs(mfintchanges[i+1, j]) <= threshlim):
                        cutoff = i
                        found = True
                        break
            if not found:
                cutoff = numit - 1
            cutoffnum[j] = cutoff
            cutoffradii[j] = icutrad + cutoff - 1  # integration radius (in pixels)
            ncint[j] = cluspxsums[cutoff, j]

    ### Generate Final Results Table
    # Convert integration radius and physical radius to nm:
    if runint and runphys:
        cutoffradiinm = cutoffradii * pxscale
        physrad_nm = physrad * pxscale
        nctable = pd.DataFrame({
            'NP': np.arange(1, nump+1),
            'Integrated_Intensity': ncint,
            'Integration_Radius_nm': cutoffradiinm,
            'Avg_Physical_Radius_nm': physrad_nm
        })
    elif runint:
        cutoffradiinm = cutoffradii * pxscale
        nctable = pd.DataFrame({
            'NP': np.arange(1, nump+1),
            'Integrated_Intensity': ncint,
            'Integration_Radius_nm': cutoffradiinm
        })
    elif runphys:
        physrad_nm = physrad * pxscale
        nctable = pd.DataFrame({
            'NP': np.arange(1, nump+1),
            'Avg_Physical_Radius_nm': physrad_nm
        })
    else:
        nctable = pd.DataFrame({'NP': np.arange(1, nump+1)})

    output_filename = f'{outpath}/{fname}_results.csv'
    nctable.to_csv(output_filename, sep=',', index=False)
    
    ###Plot original image with physical and integrated radius of each particle
    if output_radius:
        plt.figure(figsize=(10, 10))
        plt.imshow(gfimg, cmap='gray')
        ax = plt.gca()

        # Loop through each particle and draw circles
        n_particles = fitcent.shape[0]
        for i in range(n_particles):
            center = (fitcent[i, 0], fitcent[i, 1])
            
            # Draw the physical radius (blue circle)
            circ_phys = Circle(center, physrad[i], edgecolor='blue', facecolor='none',
                            lw=2, label='Physical Radius' if i == 0 else None)
            ax.add_patch(circ_phys)
            
            # Draw the cutoff (integration) radius (yellow circle)
            circ_cut = Circle(center, cutoffradii[i], edgecolor='yellow', facecolor='none',
                            lw=2, label='Cutoff Radius' if i == 0 else None)
            ax.add_patch(circ_cut)
            
            # Mark the center:
            ax.plot(center[0], center[1], 'r*', markersize=10, label='Centers')

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels)

        plt.title('Original Image with Physical (blue) and Cutoff (yellow) Radii')
        plt.axis('off')
        plt.savefig(f'{outpath}/{fname}_radii.png')
        plt.close()

    ###Calculate histogram
    if get_histogram:
        data_int=nctable['Integrated_Intensity']
        binnum=30
        hist, bin_edges = np.histogram(data_int, bins=binnum)
        freq_ind = np.argmax(hist)
        bin_edges_norm=bin_edges/bin_edges[freq_ind]
        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges_norm[:-1], hist, width=np.diff(bin_edges_norm), edgecolor="black", alpha=0.7)
        plt.xlabel("Value")
        plt.ylabel("Normalized Frequency")
        plt.savefig(f"{outpath}/{fname}_histogram_normalized.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", alpha=0.7)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(f"{outpath}/{fname}_histogram.png")
        plt.close()

        histsave=np.hstack((bin_edges[:-1].reshape(-1,1),bin_edges_norm[:-1].reshape(-1,1)))
        histsave=np.hstack((histsave,hist.reshape(-1,1)))
        np.savetxt(f"{outpath}/{fname}_histogram.csv",histsave,delimiter=",",header="Bin, Normalized Bin, Frequency")

def main():
    parser = argparse.ArgumentParser(description="Get user input.")
    
    parser.add_argument("fname", type=str, help="Name of file to analyze")
    parser.add_argument("--ftype", type=str,default='dm3', help="File type of input data")
    parser.add_argument("--inpath", type=str,default='./', help="File path of input data")
    parser.add_argument("--outpath", type=str,default='./', help="File path for outputs")
    parser.add_argument("--calc_center", action="store_true", help="Use Otsu filter to get center of particle")
    parser.add_argument("--no-output_radius", action="store_false", dest="output_radius", help="Disable the radius output (default: enabled)")
    parser.add_argument("--no-fillhole", action="store_false", dest="fillhole", help="Do not fill holes in the image (default: enabled)")
    parser.add_argument("--no-runfilt", action="store_false", dest="runfilt", help="Do not use Gaussian filter on image (default: enabled)")
    parser.add_argument("--no-get_histogram", action="store_false", dest="get_histogram", help="Do not get histogram of particle intensities (default: enabled)")
    parser.add_argument("--no-runint", action="store_false", dest="runint", help="Do not get integrated intensity (default: enabled)")
    parser.add_argument("--no-runphys", action="store_false", dest="runphys", help="Do not get physical radius (default: enabled)")

    calc_npintensity(parser)
    
# Standard Python script execution check
if __name__ == "__main__":
    main()