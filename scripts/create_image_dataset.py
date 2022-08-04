"""
Creates color+normal datasets based on annotation file.
The datasets can be used to train MLP, CycleGAN, Pix2Pix models.
"""
import csv
import logging
import os

import cv2
import hydra
import imageio
import numpy as np
import torch

from src.third_party import data_utils
from src.dataio.generate_sphere_gt_normals import generate_sphere_gt_normals
from src.dataio.data_loader import data_loader
log = logging.getLogger(__name__)
BASE_PATH="/home/shuk/digit-depth"


@hydra.main(config_path="/home/shuk/digit-depth/config", config_name="rgb_to_normal.yaml")
def main(cfg):
    train_dataloader, train_dataset = data_loader(dir_dataset=os.path.join(BASE_PATH, "images"), params=cfg.dataloader)
    dataset_type = cfg.dataset.dataset_type
    print(os.path.join(BASE_PATH, "images"))
    dirs = ["/home/shuk/digit-depth/datasets/A",
            "/home/shuk/digit-depth/datasets/B",
            "/home/shuk/digit-depth/datasets/AB"]
    for dir in dirs: os.makedirs(f"{dir}/{dataset_type}", exist_ok=True)
    # iterate over datasets
    print(dirs)
    ds_idx = 0
    mm_to_pixel = 21.09
    radius_bearing = np.int32(0.5 * 6.0 * mm_to_pixel)
    while ds_idx < len(train_dataset):
        print(f"Dataset idx: {ds_idx:04d}")
        # read img + annotations
        data = train_dataset[ds_idx]

        if cfg.dataloader.annot_flag:
            img, annot = data
            if annot.shape[0] == 0:
                ds_idx = ds_idx + 1
                continue
        else:
            img = data

        # get annotation circle params
        if cfg.dataloader.annot_flag:
            annot_np = annot.cpu().detach().numpy()
            center_y, center_x, radius_annot = annot_np[0][1], annot_np[0][0], annot_np[0][2]
        else:
            center_y, center_x, radius_annot = 0, 0, 0

        img_color_np = img.permute(2, 1, 0).cpu().detach().numpy()  # (3,320,240) -> (240,320,3)

        # apply foreground mask
        fg_mask = np.zeros(img_color_np.shape[:2], dtype='uint8')
        fg_mask = cv2.circle(fg_mask, (center_x, center_y), radius_annot, 255, -1)

        # 1. rgb -> normal (generate gt surface normals)
        img_mask = cv2.bitwise_and(img_color_np, img_color_np, mask=fg_mask)
        img_normal_np = generate_sphere_gt_normals(img_mask, center_x, center_y, radius=radius_bearing)

        """"
        # script for generating normals as csv file
        print(f"img_normal_np shape: {img_normal_np.shape}")
        img_normal_np = data_utils.interpolate_img(img=torch.tensor(img_normal_np).permute(2, 0, 1), rows=160, cols=120)
        img_normal_np = img_normal_np.permute(1, 2, 0).cpu().detach().numpy()

        # CSV generation for normals
        nx=img_normal_np[:,:,0]
        ny=img_normal_np[:,:,1]
        nz=img_normal_np[:,:,2]

        for y in range(img_normal_np.shape[0]):
            for x in range(img_normal_np.shape[1]):
                # print(f"x{x}, y{y}, nx{nx[y,x]}, ny{ny[y,x]}, nz{nz[y,x]}")
                value=np.hstack([x,y,nx[y,x],ny[y,x],nz[y,x]])
                with open(f"{BASE_PATH}images/B/image_{ds_idx}.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(value)
        print("CSV written for image {}".format(ds_idx))
        """
        if cfg.dataset.save_dataset:
            # downsample: (320,240,3) -> (160,120,3)
            img_color_ds = data_utils.interpolate_img(img=torch.tensor(img_color_np).permute(2, 0, 1), rows=160,
                                                      cols=120)
            img_normal_ds = data_utils.interpolate_img(img=torch.tensor(img_normal_np).permute(2, 0, 1), rows=160,
                                                       cols=120)
            img_color_np = img_color_ds.permute(1, 2, 0).cpu().detach().numpy()
            img_normal_np = img_normal_ds.permute(1, 2, 0).cpu().detach().numpy()
            imageio.imwrite(f"{dirs[0]}/{dataset_type}/{ds_idx:04d}.png", img_color_np)
            imageio.imwrite(f"{dirs[1]}/{dataset_type}/{ds_idx:04d}.png", img_normal_np)
            print(f"Saved image {ds_idx:04d}")

        ds_idx = ds_idx + 1

    if cfg.dataset.save_dataset:
        os.system(
            f"python {BASE_PATH}/src/dataio/combine_A_and_B.py --fold_A {dirs[0]} --fold_B {dirs[1]} --fold_AB {dirs[2]}")
        log.info(f"Created color+normal datasets of {ds_idx} images at {dirs[2]}/{dataset_type}.")

if __name__ == '__main__':
    main()
