""" Publishes a ROS topic with name /depth/compressed and shows the image on OpenCV window.
    Issues: rqt_image_view is not showing the image due to some data conversion issues but OpenCV is showing the image."""
import os
import hydra
import open3d as o3d
from src.third_party import geom_utils
from src.digit import DigitSensor
from src.train import MLP
from src.train.prepost_mlp import *
from attrdict import AttrDict
from src.third_party import vis_utils
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


@hydra.main(config_path="/home/shuk/digit-depth/config", config_name="rgb_to_normal.yaml", version_base=None)
def show_point_cloud(cfg):
    view_params = AttrDict({'fov': 60, 'front': [-0.1, 0.1, 0.1], 'lookat': [
        -0.001, -0.01, 0.01], 'up': [0.04, -0.05, 0.190], 'zoom': 2.5})
    vis3d = vis_utils.Visualizer3d(base_path=base_path, view_params=view_params)

    # projection params
    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset)
    proj_mat = torch.tensor(cfg.sensor.P)
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    # base image depth map
    base_img = cv2.imread(cfg.base_img_path)
    base_img = preproc_mlp(base_img)
    base_img_proc = model(base_img).cpu().detach().numpy()
    base_img_proc, normal_base = post_proc_mlp(base_img_proc)
    # get gradx and grady
    gradx_base, grady_base = geom_utils._normal_to_grad_depth(img_normal=base_img_proc, gel_width=cfg.sensor.gel_width,
                                                              gel_height=cfg.sensor.gel_height, bg_mask=None)

    # reconstruct depth
    img_depth_base = geom_utils._integrate_grad_depth(gradx_base, grady_base, boundary=None, bg_mask=None,
                                                      max_depth=0.0237)
    img_depth_base = img_depth_base.detach().cpu().numpy() # final depth image for base image
    # setup digit sensor
    digit = DigitSensor(30, "QVGA", cfg.sensor.serial_num)
    digit_call = digit()
    while True:
        frame = digit_call.get_frame()
        img_np = preproc_mlp(frame)
        img_np = model(img_np).detach().cpu().numpy()
        img_np, normal_img = post_proc_mlp(img_np)
        # get gradx and grady
        gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=cfg.sensor.gel_width,
                                                                gel_height=cfg.sensor.gel_height,bg_mask=None)
        # reconstruct depth
        img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,max_depth=0.0237)
        view_mat = torch.eye(4)  # torch.inverse(T_cam_offset)
        # Project depth to 3D
        points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat, V=view_mat, params=cfg.sensor)
        points3d = geom_utils.remove_background_pts(points3d, bg_mask=None)
        cloud = o3d.geometry.PointCloud()
        clouds = geom_utils.init_points_to_clouds(clouds=[copy.deepcopy(cloud)], points3d=[points3d])
        vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds)


if __name__ == "__main__":
    show_point_cloud()