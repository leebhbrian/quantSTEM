import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
import sys

def get_closeind(x,y,pos):
    distances = np.linalg.norm(pos - np.array([x, y]), axis=1)
    closest_index = np.argmin(distances)
    min_distance = distances[closest_index]
    return closest_index,min_distance

fappend=sys.argv[1]
mask=np.load(f'./Au144/mask_{fappend}/masksave.npy')
image=np.load(f'./Au144/gfimg_{fappend}.npy')
cenpos=np.load(f'./Au144/{fappend}_centers.npy')
wf=open(f"./Au144/{fappend}_res_new.csv",'w')
print(mask.shape,image.shape)
nmask=mask.shape[0]
bkgwidth=2
erod_struct = np.ones((2 * bkgwidth + 1, 2 * bkgwidth + 1), dtype=bool)
dilate_struct = np.ones((2 * bkgwidth + 1, 2 * bkgwidth + 1), dtype=bool)
for i1 in range(1,nmask):
    maskorig=mask[i1,:,:].astype(bool)
    true_indices = np.argwhere(maskorig)
    centroid = true_indices.mean(axis=0)
    centroid_row, centroid_col = centroid.astype(int)
    closest_index,min_distance=get_closeind(centroid_col,centroid_row,cenpos)
    if min_distance<15:
        cond=True
        masknow=np.copy(maskorig)
        while cond:
            dilated_mask = binary_dilation(masknow, structure=dilate_struct)
            annulus_mask=dilated_mask & ~masknow
            sum_now=np.sum(image[masknow])
            n_now=masknow.sum()
            avg_now=sum_now/n_now
            sum_bkg=np.sum(image[annulus_mask])
            n_bkg=annulus_mask.sum()
            avg_bkg=sum_bkg/n_bkg
            cond=False
        sum_fin=sum_now-avg_bkg*n_now
        tw=f"{i1},{centroid_col},{centroid_row},{closest_index+1},{min_distance},{n_now},{sum_fin},{sum_now},{avg_now},{avg_bkg}\n"
        wf.write(tw)
    """
    eroded_mask = binary_erosion(masknow, structure=erod_struct)
    dilated_mask = binary_dilation(masknow, structure=dilate_struct)
    sum_orig = np.sum(image[masknow])
    sum_erod = np.sum(image[eroded_mask])
    sum_dilate=np.sum(image[dilated_mask])
    n_orig=masknow.sum()
    n_erod=eroded_mask.sum()
    n_dilate=dilated_mask.sum()
    print(i1,n_orig,n_erod,n_dilate,sum_orig,sum_erod,sum_dilate)
    print(i1,centroid_row,centroid_col,closest_index,min_distance,cenpos[closest_index,0],cenpos[closest_index,1],ntrue)
    if min_distance<15:
        tw=f"{i1},{centroid_col},{centroid_row},{closest_index+1},{min_distance},{ntrue},{sum_intensity}\n"
        wf.write(tw)
    """