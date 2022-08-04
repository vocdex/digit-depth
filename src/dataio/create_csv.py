import numpy as np
import csv
import glob
from PIL import Image
import pandas as pd


def create_normal_csv(save_dir:str, img_dir: str):
    normal_imgs=sorted(glob.glob(f"{img_dir}/*.png"))
    print(f"Found {len(normal_imgs)} normal images")
    for idx,img_path in enumerate(normal_imgs):
        img_normal=Image.open(img_path)
        img_normal_np=np.array(img_normal)
        nx = img_normal_np[:, :, 0]
        ny = img_normal_np[:, :, 1]
        nz = img_normal_np[:, :, 2]

        for y in range(img_normal_np.shape[0]):
            for x in range(img_normal_np.shape[1]):
                value = np.hstack([x, y, nx[y, x]/255, ny[y, x]/255, nz[y, x]/255])
                with open(f"{save_dir}/image_{idx}.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(value)
        print("Normal image CSV written for image {}".format(idx))


def create_color_csv(img_dir: str, save_dir:str):
    color_imgs=sorted(glob.glob(f"{img_dir}/*.png"))
    print(f"Found {len(color_imgs)} color images")
    for idx,img_path in enumerate(color_imgs):
        img=Image.open(img_path)
        img_np=np.array(img)
        xy_coords = np.flip(np.column_stack(np.where(np.all(img_np >= 0, axis=2))), axis=1)
        rgb = np.reshape(img_np, (np.prod(img_np.shape[:2]), 3))
        # Add pixel numbers in front
        pixel_numbers = np.expand_dims(np.arange(1, xy_coords.shape[0] + 1), axis=1)
        value = np.hstack([pixel_numbers, xy_coords, rgb])
        # Properly save as CSV
        df = pd.DataFrame(value, columns=['pixel_number', 'X', 'Y', 'R', 'G', 'B'])
        del df['pixel_number']
        df.to_csv(f"{save_dir}/image_{idx}.csv", sep=",",
                  index=False)
        print(f'Color image CSV written for image {idx}')

save_dir="/home/shuk/digit-depth/datasets/B/csv"
img_dir="/home/shuk/digit-depth/datasets/B/imgs"
create_normal_csv(save_dir, img_dir)