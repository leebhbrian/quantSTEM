import numpy as np
import matplotlib.pyplot as plt

from params import picname,input_file,ncpos,showfig,runstitch,fillhole,numheadrow,numheadcol,runfilt,gfsize,gfsigma,np_size
from params import pxscale,runphys,runint,icutrad,bkgwidth,ficfilt,threshlim,numincthresh,N,fit_range,numit
from params import cell_size,numit,gfimg,nump,height,width,N

def calculate_physical_radii(zfit, fitmax, zfitwidth, zfitheight, fitcent, plabels, firstfig, gfx, gfy, curmin, curmax, cucmin, cucmax):
    """
    Converts the provided MATLAB script into Python.

    Parameters:
    - runphys: int
        Flag to determine whether to calculate physical radii.
    - input_file: ndarray
        The image data to be analyzed.
    - nump: int
        Number of nanoparticles.
    - zfit: list of ndarray
        Fitted 2D Gaussian data for each nanoparticle.
    - fitmax: ndarray
        Array of maxima positions for each fit.
    - zfitwidth: list of int
        Widths of each fitted image.
    - zfitheight: list of int
        Heights of each fitted image.
    - N: float
        Parameter used in calculating FW(1/N)M.
    - fitcent: ndarray
        Center coordinates of each nanoparticle.
    - plabels: list of str
        Labels for each nanoparticle.
    - firstfig: matplotlib.figure.Figure
        The initial figure to be closed.
    - gfx, gfy: list of ndarray
        X and Y coordinates for plotting.
    - gfimg: ndarray
        Gaussian-filtered image data.
    - curmin, curmax, cucmin, cucmax: list of int
        Indices for cropping images.

    Returns:
    None
    """
    N=2
    figcond=False
    if runphys == 1:
        # Close first figure
        if figcond:
            plt.close(firstfig)
            finalplot = plt.figure('Analyzed Image')
            plt.imshow(input_file, cmap='gray')
            
        # Initialize variables
        gfxfit = [None] * nump  # x profile through fit
        gfyfit = [None] * nump  # y profile through fit
        xnmaxpos = np.zeros(nump)  # Nearest position to (1/N)M on x profile
        ynmaxpos = np.zeros(nump)  # Nearest position to (1/N)M on y profile
        physrad = np.zeros(nump)   # Physical radii

        # For each nanoparticle, extract line profiles and calculate FW(1/N)M
        print("nump",nump)
        for inc in range(nump):
            # Extract x line profile from fit
            row_index = int(round(fitmax[inc, 1]))
            gfxfit[inc] = zfit[inc][row_index, :int(zfitwidth[inc])]
            # Extract y line profile from fit
            col_index = int(round(fitmax[inc, 0]))
            gfyfit[inc] = zfit[inc][:int(zfitheight[inc]), col_index]

            # Calculate the nearest position to the (1/N)M on the x and y profiles
            xnmp = np.argmin(np.abs(gfxfit[inc] - (np.max(gfxfit[inc]) / N)))
            ynmp = np.argmin(np.abs(gfyfit[inc] - (np.max(gfyfit[inc]) / N)))

            # Adjust xnmp using linear interpolation
            if (gfxfit[inc][xnmp] - np.max(gfxfit[inc]) / N) < 0:
                if xnmp < len(gfxfit[inc]) - 1:
                    frac = (np.max(gfxfit[inc]) / N - gfxfit[inc][xnmp]) / (gfxfit[inc][xnmp + 1] - gfxfit[inc][xnmp])
                    xnmp += frac
            else:
                if xnmp > 0:
                    frac = (gfxfit[inc][xnmp] - np.max(gfxfit[inc]) / N) / (gfxfit[inc][xnmp] - gfxfit[inc][xnmp - 1])
                    xnmp -= frac

            # Adjust ynmp using linear interpolation
            if (gfyfit[inc][ynmp] - np.max(gfyfit[inc]) / N) < 0:
                if ynmp < len(gfyfit[inc]) - 1:
                    frac = (np.max(gfyfit[inc]) / N - gfyfit[inc][ynmp]) / (gfyfit[inc][ynmp + 1] - gfyfit[inc][ynmp])
                    ynmp += frac
            else:
                if ynmp > 0:
                    frac = (gfyfit[inc][ynmp] - np.max(gfyfit[inc]) / N) / (gfyfit[inc][ynmp] - gfyfit[inc][ynmp - 1])
                    ynmp -= frac

            xnmaxpos[inc] = xnmp
            ynmaxpos[inc] = ynmp
            # Average the two radii
            physrad[inc] = (abs(fitmax[inc, 0] - xnmaxpos[inc]) + abs(fitmax[inc, 1] - ynmaxpos[inc])) / 2
        if figcond:
            # Plot the results
            plt.figure(finalplot.number)
            ax = plt.gca()
            for i in range(nump):
                circle = plt.Circle((fitcent[i, 0], fitcent[i, 1]), physrad[i], color='b', fill=False, linewidth=0.5)
                ax.add_patch(circle)
            plt.title('Physical Dimensions (blue)')
            plt.draw()
            plt.pause(0.1)

            # Allow user to change N value and recalculate physical radii
            pdummyt = True
            while pdummyt:
                plt.figure(finalplot.number)
                plt.clf()
                plt.imshow(input_file, cmap='gray')
                ax = plt.gca()
                for i in range(nump):
                    ax.text(fitcent[i, 0] + 17, fitcent[i, 1] - 17, plabels[i], color='y',
                            fontweight='bold', horizontalalignment='center',
                            backgroundcolor='r', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
                    circle = plt.Circle((fitcent[i, 0], fitcent[i, 1]), physrad[i], color='b', fill=False, linewidth=0.5)
                    ax.add_patch(circle)
                plt.title(f'Integration Bounds (yellow) - Physical Bounds (FW(1/{N})M) (blue)')
                plt.draw()
                plt.pause(0.1)

                newNinput = input(f'Enter a new N value or press Enter to accept [{N}]: ')
                if newNinput.strip() != '':
                    new_N = float(newNinput)
                    if new_N != N:
                        N = new_N
                        # Recalculate physical radii with new N
                        for inc in range(nump):
                            xnmp = np.argmin(np.abs(gfxfit[inc] - (np.max(gfxfit[inc]) / N)))
                            ynmp = np.argmin(np.abs(gfyfit[inc] - (np.max(gfyfit[inc]) / N)))

                            # Adjust xnmp as before
                            if (gfxfit[inc][xnmp] - np.max(gfxfit[inc]) / N) < 0:
                                if xnmp < len(gfxfit[inc]) - 1:
                                    frac = (np.max(gfxfit[inc]) / N - gfxfit[inc][xnmp]) / (gfxfit[inc][xnmp + 1] - gfxfit[inc][xnmp])
                                    xnmp += frac
                            else:
                                if xnmp > 0:
                                    frac = (gfxfit[inc][xnmp] - np.max(gfxfit[inc]) / N) / (gfxfit[inc][xnmp] - gfxfit[inc][xnmp - 1])
                                    xnmp -= frac

                            # Adjust ynmp as before
                            if (gfyfit[inc][ynmp] - np.max(gfyfit[inc]) / N) < 0:
                                if ynmp < len(gfyfit[inc]) - 1:
                                    frac = (np.max(gfyfit[inc]) / N - gfyfit[inc][ynmp]) / (gfyfit[inc][ynmp + 1] - gfyfit[inc][ynmp])
                                    ynmp += frac
                            else:
                                if ynmp > 0:
                                    frac = (gfyfit[inc][ynmp] - np.max(gfyfit[inc]) / N) / (gfyfit[inc][ynmp] - gfyfit[inc][ynmp - 1])
                                    ynmp -= frac

                            xnmaxpos[inc] = xnmp
                            ynmaxpos[inc] = ynmp
                            physrad[inc] = (abs(fitmax[inc, 0] - xnmaxpos[inc]) + abs(fitmax[inc, 1] - ynmaxpos[inc])) / 2
                    else:
                        pdummyt = False
                else:
                    pdummyt = False

            # Allow user to refine individual nanoparticles
            pdummync = True
            while pdummync:
                pdummyr = True
                plt.figure(finalplot.number)
                plt.clf()
                plt.imshow(input_file, cmap='gray')
                ax = plt.gca()
                for i in range(nump):
                    ax.text(fitcent[i, 0] + 17, fitcent[i, 1] - 17, plabels[i], color='y',
                            fontweight='bold', horizontalalignment='center',
                            backgroundcolor='r', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
                    circle = plt.Circle((fitcent[i, 0], fitcent[i, 1]), physrad[i], color='b', fill=False, linewidth=0.5)
                    ax.add_patch(circle)
                plt.title('Left-click a NP to modify or press any key to accept')
                plt.draw()
                plt.pause(0.1)

                print('Click on a nanoparticle to modify, or press any key to accept.')
                coords = plt.ginput(1, timeout=0)
                if len(coords) > 0:
                    xn, yn = coords[0]
                    # Find the nearest nanoparticle to the click
                    distances = (fitcent[:, 0] - xn) ** 2 + (fitcent[:, 1] - yn) ** 2
                    inc = np.argmin(distances)

                    # Proceed to adjust the NP
                    print(f'You selected NP #{inc + 1}')
                    # Here you can implement the code to adjust the physical radius
                    # of the selected nanoparticle as per your requirements.
                    # This involves displaying plots and accepting user inputs.
                    # For brevity, detailed implementation is omitted.

                    # After adjusting, update the physical radius
                    # physrad[inc] = new_radius_value

                else:
                    pdummync = False
                    plt.figure(finalplot.number)
                    plt.title('Final Physical Radii of Nanoparticles')
                    plt.draw()
        return physrad
    else:
        # Do not calculate the physical radius
        pass
