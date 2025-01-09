import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
import glob
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from util import *
from scipy.ndimage import binary_erosion,binary_dilation

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process a file and additional parameters.")

    # Add arguments
    parser.add_argument('-input_path', type=str, required=True, help="Path to input files.")
    parser.add_argument('-file_type', type=str, required=True, help="File extension.")
    
    # Parse arguments
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    np.random.seed(3)
    bkgwidth=2
    changewidth=1
    thresh=0.1
    dilate_struct = np.ones((2 * bkgwidth + 1, 2 * bkgwidth + 1), dtype=bool)
    dilate_struct2 = np.ones((2 * changewidth + 1, 2 * changewidth + 1), dtype=bool)

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # Access the arguments
    input_path = args.input_path
    file_type=args.file_type
    files = glob.glob(f"{input_path}/*.{file_type}")
    for file in files:
        fname=file.rsplit('.', 1)[0]
        image = Image.open(file)
        image = np.array(image).astype(np.float32)
        image_orig=np.copy(image)
        image= (np.repeat(image[:, :, np.newaxis], 3, axis=2)/np.max(image)*1.0).astype(np.float32)
        masks = mask_generator.generate(image)
        nmask=len(masks)
        centers=np.zeros((nmask,2))
        masktot=np.zeros((nmask,masks[0]['segmentation'].shape[0],masks[0]['segmentation'].shape[1]))
        wf=open(f"{fname}_analysis.csv",'w')
        for i1 in range(nmask):
            mask=masks[i1]['segmentation']
            masktot[i1,:,:]=mask
            true_indices = np.argwhere(mask)
            centroid = true_indices.mean(axis=0)
            centroid_row, centroid_col = centroid.astype(int)
            centers[i1,1]=centroid_row
            centers[i1,0]=centroid_col

            cond=True
            masknow=np.copy(mask)
            sumcor_now=np.sum(image[masknow])
            count=-1
            while cond:
                sumcor_prev=sumcor_now
                dilated_mask = binary_dilation(masknow, structure=dilate_struct)
                annulus_mask=dilated_mask & ~masknow
                sum_now=np.sum(image[masknow])
                n_now=masknow.sum()
                avg_now=sum_now/n_now
                sum_bkg=np.sum(image[annulus_mask])
                n_bkg=annulus_mask.sum()
                avg_bkg=sum_bkg/n_bkg
                sumcor_now=sum_now-avg_bkg*n_now
                intchange=(sumcor_prev-sumcor_now)/sumcor_prev
                count+=1
                masknow=binary_dilation(masknow, structure=dilate_struct2)
                if abs(intchange)<thresh:
                    cond=False
            tw=f"{i1},{centroid_col},{centroid_row},{n_now},{sumcor_now},{sum_now},{avg_now},{avg_bkg},{count}\n"
            wf.write(tw)
        np.save(f"{fname}_masks.npy",masktot) 
        np.save(f"{fname}_centroids.npy",centers) 
if __name__ == "__main__":
    main()