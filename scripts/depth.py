
import os

import cv2
import hydra
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from src.third_party import geom_utils
from src.digit import DigitSensor
from src.train import MLP
from src.train.prepost_mlp import *
from PIL import Image
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class ImageFeature:

    def __init__(self):
        # topic where we publish

        self.image_pub = rospy.Publisher("/depth/compressed",
                                         CompressedImage, queue_size=10)
        self.br = CvBridge()


@hydra.main(config_path="/home/shuk/digit-depth/config", config_name="rgb_to_normal.yaml", version_base=None)
def show_depth(cfg):
    model=MLP()
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    ic = ImageFeature()
    br = CvBridge()
    rospy.init_node('depth_node', anonymous=True)
    # base image depth map
    base_img = preproc_mlp(cfg.base_img_path)
    base_img_np = model(base_img).detach().cpu().numpy()
    base_img_np, normal_base = post_proc_mlp(base_img_np)
    # get gradx and grady
    gradx_base, grady_base = geom_utils._normal_to_grad_depth(img_normal=base_img_np, gel_width=cfg.sensor.gel_width,
                                                              gel_height=cfg.sensor.gel_height, bg_mask=None)

    # reconstruct depth
    img_depth_base = geom_utils._integrate_grad_depth(gradx_base, grady_base, boundary=None, bg_mask=None,
                                                      max_depth=0.0237)
    img_depth_base = img_depth_base.detach().cpu().numpy() # final depth image for base image
    # setup digit sensor
    digit = DigitSensor(30, "QVGA", cfg.sensor.serial_num)
    digit_call = digit()
    while not rospy.is_shutdown():
        frame = digit_call.get_frame()
        frame = cv2.imwrite("framex.png", frame)
        filename = "framex.png"
        cv2.imshow("frame", frame)
        img_np = preproc_mlp(filename)
        img_np = model(img_np).detach().cpu().numpy()
        img_np, normal_img = post_proc_mlp(img_np)
        # get gradx and grady
        gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=cfg.sensor.gel_width,gel_height=cfg.sensor.gel_height,bg_mask=None)
        # reconstruct depth
        img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,max_depth=0.0237)
        img_depth = img_depth.detach().cpu().numpy() # final depth image for current image
        # get depth difference
        cv2.imshow("depth", img_depth)
        depth_diff = img_depth - img_depth_base
        msg = br.cv2_to_compressed_imgmsg(img_depth, "png")
        ic.image_pub.publish(msg)
        now = rospy.get_rostime()
        rospy.loginfo("published depth image at {}".format(now))
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("starting...")
    show_depth()

