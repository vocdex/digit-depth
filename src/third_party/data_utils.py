
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import math

import torch
import pytorch3d.transforms as p3d_t
# from scipy.spatial.transform import Rotation as R

"""
dataloader, logger helpers
"""

def pandas_col_to_numpy(df_col):

    df_col = df_col.apply(lambda x: np.fromstring(x.replace('\n', '').replace(
        '[', '').replace(']', '').replace('  ', ' '), sep=', '))
    df_col = np.stack(df_col)

    return df_col

def pandas_string_to_numpy(arr_str):
    arr_npy = np.fromstring(arr_str.replace('\n', '').replace(
        '[', '').replace(']', '').replace('  ', ' '), sep=', ')
    return arr_npy

def interpolate_img(img, rows, cols):
    '''
    img: C x H x W
    '''
    
    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)
    
    return img

"""
quant metrics 
"""

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap

def traj_error_trans(xyz_gt, xyz_est):

    diff = xyz_gt - xyz_est
    diff_sq = diff**2

    rmse_trans = np.sqrt(np.mean(diff_sq.flatten()))
    error = rmse_trans

    return error

def traj_error_rot(rot_mat_gt, rot_mat_est, convention="XYZ"):

    if (torch.is_tensor(rot_mat_gt) & torch.is_tensor(rot_mat_est)):
        rot_rpy_gt = p3d_t.matrix_to_euler_angles(rot_mat_gt, convention)
        rot_rpy_est = p3d_t.matrix_to_euler_angles(rot_mat_est, convention)
    else:
        rot_rpy_gt = (p3d_t.matrix_to_euler_angles(torch.tensor(rot_mat_gt), convention)).cpu().detach().numpy()
        rot_rpy_est = (p3d_t.matrix_to_euler_angles(torch.tensor(rot_mat_est), convention)).cpu().detach().numpy()

    diff = wrap_to_pi(rot_rpy_gt - rot_rpy_est)
    diff_sq = diff**2

    rmse_rot = np.sqrt(np.mean(diff_sq.flatten()))
    error = rmse_rot

    return error

"""
miscellaneous
"""

def coords_to_circle_params(coords):
    """
    Ref: http://www.ambrsoft.com/trigocalc/circle3d.htm
    """

    y1, x1 = coords[1], coords[2]
    y2, x2 = coords[3], coords[4]
    y3, x3 = coords[5], coords[6]

    A = x1 * (y2-y3) - y1 * (x2-x3) + x2*y3 - x3*y2
    B = (x1**2 + y1**2)*(y3-y2) + (x2**2 + y2**2)*(y1-y3) + (x3**2 + y3**2)*(y2-y1)
    C = (x1**2 + y1**2)*(x2-x3) + (x2**2 + y2**2)*(x3-x1) + (x3**2 + y3**2)*(x1-x2)
    D = (x1**2 + y1**2)*(x3*y2-x2*y3) + (x2**2 + y2**2)*(x1*y3-x3*y1) + (x3**2 + y3**2)*(x2*y1-x1*y2)

    center_x = - (B / 2*A)
    center_y = - (C / 2*A)
    radius = np.sqrt((B**2 + C**2 - 4*A*D) / (4 * A**2))

    center_x, center_y = np.int(center_x), np.int(center_y)
    radius = np.int(radius)

    return center_x, center_y, radius
    
def regularization_loss(params_net, reg_type):
    reg_loss = 0
        
    if(reg_type == "l1"):
        for param in params_net:
            reg_loss += torch.sum(torch.abs(param))
    
    elif(reg_type == "l2"):
        for param in params_net:
            reg_loss += torch.sum(torch.norm(param))
    
    return reg_loss

def network_update(output, optimizer):
    # clear, backprop and apply new gradients
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

def normalize_imgfeats(imgfeat, norm_mean_list, norm_std_list, device=None):
    norm_mean = torch.cuda.FloatTensor(norm_mean_list) if device == torch.device(
        'cuda:0') else torch.FloatTensor(norm_mean_list)
    norm_std = torch.cuda.FloatTensor(norm_std_list) if device == torch.device(
        'cuda:0') else torch.FloatTensor(norm_std_list)

    imgfeat_norm = torch.div(torch.sub(imgfeat, norm_mean), norm_std)
    
    return imgfeat_norm

def denormalize_img(img_norm, norm_mean, norm_std):
    img = torch.zeros(img_norm.shape)

    img[:, 0, :, :] = torch.add(
        torch.mul(img_norm[:, 0, :, :], norm_std[0]), norm_mean[0])
    img[:, 1, :, :] = torch.add(
        torch.mul(img_norm[:, 1, :, :], norm_std[1]), norm_mean[1])
    img[:, 2, :, :] = torch.add(
        torch.mul(img_norm[:, 2, :, :], norm_std[2]), norm_mean[2])

    return img

def weighted_mse_loss(input, target, mse_wts):
    return torch.mean(mse_wts * (input - target) ** 2)

