from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()

import cv2
import einops
import numpy as np
import torch
import random
import glob
import os
import argparse

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3

from ciconv2d0 import CIConv2d
from PIL import Image


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)

def gray(t): return (
        np.clip((t[0][0] if len(t.shape) == 4 else t[0]).detach().cpu().numpy(), 0, 1) * 255).astype(
    np.uint8)

if __name__ == '__main__':

    input_folder = '1/gate'
    output_folder = 'priors_2021/gate'
    
    extraction_model = CIConv2d('W', k=3, scale=0.8)
    extraction_model = extraction_model.cuda()


    # extract prior
    img_list = glob.glob(f"{input_folder}/*.*")
    print(f"Find {len(img_list)} files in {input_folder}")


    H_folder = os.path.join(output_folder, 'W_0.8')
    #S_folder = os.path.join(output_folder, 'S')
    #RGB_folder = os.path.join(output_folder, 'RGB_order')
    #Ww_folder = os.path.join(output_folder, 'Ww')

    os.makedirs(H_folder, exist_ok=True)
    #os.makedirs(S_folder, exist_ok=True)
    #os.makedirs(RGB_folder, exist_ok=True)
    #os.makedirs(Ww_folder, exist_ok=True)


    for img_path in img_list:
        
        input_image = cv2.imread(img_path)
        
        input_tensor = (torch.from_numpy(input_image.copy()).cuda().to(dtype=torch.float32) / 255.0).unsqueeze(0)
        #input_mean = input_tensor.mean()
        #input_tensor = torch.clamp(input_tensor * (0.488/input_mean), 0, 1)
        input_tensor = einops.rearrange(input_tensor, 'b h w c -> b c h w').clone()

        #print(input_tensor.shape)

        with torch.no_grad():
            features = extraction_model(input_tensor)
        # H = features[:, :1, :, :]
        # S = features[:, 1:2, :, :]
        # RGB_order = features[:, 2:5, :, :]
        # Ww = features[:, 5:, :, :]

        print(features.shape)

        #f_np = (f_np - f_np.min()) / (f_np.max() - f_np.min())
        #colored_f = cv2.cvtColor(f_np*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        #colored_f = cv2.applyColorMap(colored_f, cv2.COLORMAP_PLASMA)
        
        H_path = os.path.join(H_folder, os.path.basename(img_path))

        cv2.imwrite(H_path, gray(features))
        

    