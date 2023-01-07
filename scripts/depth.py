""" Publishes a ROS topic with name /digit//depth/image_raw.
    Be sure to tune the parameters in rqt_image_view for better visualization."""
import os
import cv2
import hydra
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from digit_depth.third_party import geom_utils
from digit_depth.digit import DigitSensor
from digit_depth.train.prepost_mlp import *
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

class ImageFeature:
    def __init__(self):
        self.image_pub = rospy.Publisher("/digit/depth/image_raw/",
                                         Image, queue_size=10)
        self.br = CvBridge()


@hydra.main(config_path=f"{BASE_PATH}/config", config_name="rgb_to_normal.yaml", version_base=None)
def show_depth(cfg):
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    ic = ImageFeature()
    br = CvBridge()

    rospy.init_node('depth_node', anonymous=True)
    # base image depth map
    base_img = cv2.imread(cfg.base_img_path)
    base_img = preproc_mlp(base_img)
    base_img_proc = model(base_img).cpu().detach().numpy()
    base_img_proc, _ = post_proc_mlp(base_img_proc)
    # get gradx and grady
    gradx_base, grady_base = geom_utils._normal_to_grad_depth(img_normal=base_img_proc, gel_width=cfg.sensor.gel_width,
                                                              gel_height=cfg.sensor.gel_height, bg_mask=None)

    # reconstruct depth
    img_depth_base = geom_utils._integrate_grad_depth(gradx_base, grady_base, boundary=None, bg_mask=None,
                                                      max_depth=0.0247)
    img_depth_base = img_depth_base.detach().cpu().numpy() # final depth image for base image
    # setup digit sensor
    digit = DigitSensor(cfg.sensor.fps, "QVGA", cfg.sensor.serial_num)
    digit_call = digit()
    while not rospy.is_shutdown():
        frame = digit_call.get_frame()
        img_np = preproc_mlp(frame)
        img_np = model(img_np).detach().cpu().numpy()
        img_np, _ = post_proc_mlp(img_np)
        # get gradx and grady
        gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=cfg.sensor.gel_width,
                                                                gel_height=cfg.sensor.gel_height,bg_mask=None)
        # reconstruct depth
        img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,max_depth=0.0217)
        img_depth = img_depth.detach().cpu().numpy() # final depth image for current image
        # get difference
        diff = img_depth_base - img_depth
        diff= diff*8000
        # print(diff)
        diff = diff - np.max(diff)
        diff = diff*-1
        # print(diff)
        ret, thresh4 = cv2.threshold(diff, 0, 255, cv2.THRESH_TOZERO)
        # cv2.imshow("depth", thresh4)
        msg = br.cv2_to_imgmsg(thresh4, encoding="passthrough")
        msg.header.stamp = rospy.Time.now()
        ic.image_pub.publish(msg)
        now = rospy.get_rostime()
        rospy.loginfo("published depth image at {}".format(now))
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.loginfo("starting...")
    # show_depth()

