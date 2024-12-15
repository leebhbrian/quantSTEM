import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
fappend=sys.argv[1]
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

np.random.seed(3)
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

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

image1 = Image.open(f'./images/cars.jpg')
image1 = np.array(image1).astype(np.float32)
image2 = Image.open(f'./Au144/{fappend}.tiff')
image2 = np.array(image2)
image2= (np.repeat(image2[:, :, np.newaxis], 3, axis=2)/np.max(image2)*1.0).astype(np.float32)
image3=np.load(f'./Au144/gfimg_{fappend}.npy')
image3orig=np.copy(image3)
image3= (np.repeat(image3[:, :, np.newaxis], 3, axis=2)/np.max(image3)*1.0).astype(np.float32)
cenpos=np.load(f'./Au144/{fappend}_centers.npy')
""" 
image22 = Image.fromarray(image2.astype(np.uint8))
image22.save('./Au144/test2.tiff',format="TIFF")
image33 = Image.fromarray(image3.astype(np.uint8))
image33.save('./Au144/test3.tiff',format="TIFF")
 """
print("shape",image2.shape,image1.shape,image3.shape)
print("type",type(image2[0,0,0]),type(image1[0,0,0]),type(image3[0,0,0]))
print("maxes",np.max(image2),np.max(image1),np.max(image3))
masks2 = mask_generator.generate(image3)
nmask=len(masks2)
print("masklen",nmask)
centers=np.zeros((nmask,2))
masktot=np.zeros((nmask,masks2[0]['segmentation'].shape[0],masks2[0]['segmentation'].shape[1]))
#nmask=4
wf=open(f"./Au144/mask_{fappend}/{fappend}_res.csv",'w')
for i1 in range(nmask):
    data=masks2[i1]['segmentation']
    np.save(f"./Au144/mask_{fappend}/{fappend}_{i1}.npy",data)
    masktot[i1,:,:]=data
    ntrue=data.sum()
    sum_intensity = np.sum(image3orig[data])
    true_indices = np.argwhere(data)
    centroid = true_indices.mean(axis=0)
    centroid_row, centroid_col = centroid.astype(int)
    centers[i1,1]=centroid_row
    centers[i1,0]=centroid_col
    closest_index,min_distance=get_closeind(centroid_col,centroid_row,cenpos)
    print(i1,centroid_row,centroid_col,closest_index,min_distance,cenpos[closest_index,0],cenpos[closest_index,1],ntrue)
    if min_distance<15:
        tw=f"{i1},{centroid_col},{centroid_row},{closest_index+1},{min_distance},{ntrue},{sum_intensity}\n"
        wf.write(tw)
np.save(f"./Au144/mask_{fappend}/masksave.npy",masktot)    
np.save(f"./Au144/mask_{fappend}/centers.npy",centers)
plt.figure(figsize=(20, 20))
plt.imshow(image2)
show_anns(masks2[1:nmask])
plt.axis('off')
plt.savefig(f"./Au144/mask_{fappend}/fig.png")
