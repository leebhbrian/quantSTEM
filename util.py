import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

def custom_wiener(img, window_size):
    # Calculate local mean and variance
    local_mean = uniform_filter(img, window_size)
    local_mean_sq = uniform_filter(img**2, window_size)
    local_var = local_mean_sq - local_mean**2

    # Estimate noise (can be adjusted)
    noise_est = np.mean(local_var)

    # Apply Wiener filter formula
    result = local_mean + (np.maximum(local_var - noise_est, 0) / 
                           np.maximum(local_var, 1e-5)) * (img - local_mean)
    return result

def tif_to_numpy(file_path):
    # Open the .tif file using Pillow
    with Image.open(file_path) as img:
        # Convert the image to a numpy array
        image_array = np.array(img)    
    return image_array

# Define necessary functions
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function for curve fitting.
    """
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    x0 = xo * np.cos(theta) - yo * np.sin(theta)
    y0 = xo * np.sin(theta) + yo * np.cos(theta)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - ( a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()