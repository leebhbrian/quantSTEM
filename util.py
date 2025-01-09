import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

def get_closeind(x,y,pos):
    distances = np.linalg.norm(pos - np.array([x, y]), axis=1)
    closest_index = np.argmin(distances)
    min_distance = distances[closest_index]
    return closest_index,min_distance

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
    
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