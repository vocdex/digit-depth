"""
Creates color+normal dataset based on annotation file.
The dataset can be used to train MLP, CycleGAN, Pix2Pix models.
"""

import csv
import logging
import os

import cv2
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from attrdict import AttrDict

from src.third_party import data_utils, geom_utils, vis_utils
from src.dataio.generate_sphere_gt_normals import generate_sphere_gt_normals
from src.dataio.data_loader import data_loader
log = logging.getLogger(__name__)
plt.ion()

BASE_PATH="home/shuk/digit-depth/"
# visualizer
view_params = AttrDict({'fov': 60, 'front': [-0.56, 0.81, 0.14], 'lookat': [
                        -0.006, -0.0117, 0.033], 'up': [0.0816, -0.112, 0.990], 'zoom': 5.0})

vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)


@hydra.main(config_path="/config", config_name="rgb_to_normal.yaml")
def main(cfg):
    train_dataloader, train_dataset = data_loader(params=cfg.dataloader)

    dataset_name = cfg.dataset.dataset_names  # selecting the folder containing A, B and AB image folders
    dataset_type = cfg.dataset.dataset_type

    dirs = [f"{BASE_PATH}/local/datasets/Mike/{dataset_name}/A",
            f"{BASE_PATH}/local/datasets/Mike/{dataset_name}/B",
            f"{BASE_PATH}/local/datasets/Mike/{dataset_name}/AB"]
    #
    for dir in dirs: os.makedirs(f"{dir}/{dataset_type}", exist_ok=True)

    # iterate over dataset
    ds_idx = 0

    mm_to_pixel = 21.09
    radius_bearing = np.int32(0.5 * 6.0 * mm_to_pixel)
    while ds_idx < len(train_dataset):
        print(f"Dataset idx: {ds_idx:04d}")

        fig1, axs1 = plt.subplots(nrows=3, ncols=3, num=1, clear=True, figsize=(24, 12))

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

        # normal integration params
        img_normal = torch.FloatTensor(img_normal_np)
        img_normal = img_normal.permute(2, 0, 1)  # (320,240,3) -> (3,320,240)
        boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1]))
        bg_mask = torch.tensor(1 - fg_mask / 255., dtype=torch.bool)

        # 2. normal -> grad depth
        img_normal = geom_utils._preproc_normal(img_normal=img_normal, bg_mask=bg_mask)
        gradx, grady = geom_utils._normal_to_grad_depth(img_normal=img_normal, gel_width=cfg.sensor.gel_width,
                                                        gel_height=cfg.sensor.gel_height, bg_mask=bg_mask, )

        # 3. grad depth -> depth
        img_depth = geom_utils._integrate_grad_depth(gradx, grady, boundary=boundary, bg_mask=bg_mask, max_depth=0.0237)
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
                with open(f"/home/shuk/digits2/tactile-in-hand/inhandpy/local/datasets/Mike/ball/B/image_{ds_idx}.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(value)
        print("CSV written for image {}".format(ds_idx))

        # visualize normals/depth
        img_depth_np = img_depth.cpu().detach().numpy()
        if cfg.visualize.normals:
            vis_utils.visualize_imgs(fig=fig1, axs=[axs1[0, 0], axs1[0, 1], axs1[0, 2]],
                                     img_list=[img_normal_np[:, :, 0], img_normal_np[:, :, 1], img_normal_np[:, :, 2]],
                                     titles=['nx', 'ny', 'nz'], cmap='coolwarm')
            vis_utils.visualize_imgs(fig=fig1, axs=[axs1[1, 0], axs1[1, 1]],
                                     img_list=[gradx.cpu().detach().numpy(), grady.cpu().detach().numpy()],
                                     titles=['gradx', 'grady'], cmap='coolwarm')
            vis_utils.visualize_imgs(fig=fig1, axs=[axs1[2, 0], axs1[2, 1], axs1[2, 2]],
                                     img_list=[img_color_np, img_normal_np, img_depth_np],
                                     titles=['img_color', 'img_normal', 'img_depth_recon'], cmap='coolwarm')
            plt.pause(1e-3)

        if cfg.save_dataset:
            # downsample: (320,240,3) -> (160,120,3)
            img_color_ds = data_utils.interpolate_img(img=torch.tensor(img_color_np).permute(2, 0, 1), rows=160,
                                                      cols=120)
            img_normal_ds = data_utils.interpolate_img(img=torch.tensor(img_normal_np).permute(2, 0, 1), rows=160,
                                                       cols=120)
            img_color_np = img_color_ds.permute(1, 2, 0).cpu().detach().numpy()
            img_normal_np = img_normal_ds.permute(1, 2, 0).cpu().detach().numpy()
            imageio.imwrite(f"{dirs[0]}/{dataset_type}/{ds_idx:04d}.png", img_color_np)
            imageio.imwrite(f"{dirs[1]}/{dataset_type}/{ds_idx:04d}.png", img_normal_np)

        ds_idx = ds_idx + 1

    if cfg.save_dataset:
        os.system(
            f"python {cfg.pix2pix.scripts_dir}/img_translation/combine_A_and_B.py --fold_A {dirs[0]} --fold_B {dirs[1]} --fold_AB {dirs[2]}")
        log.info(f"Created color+normal dataset of {ds_idx} images at {dirs[2]}/{dataset_type}.")
    ds_idx = ds_idx + 1

if __name__ == '__main__':
    main()
