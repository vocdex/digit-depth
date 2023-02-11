""" Publishes a ROS topic with name /digit//depth/image_raw.
    Be sure to tune the parameters in rqt_image_view for better visualization."""
import cv2
import hydra
import rospy
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from digit_depth.third_party import geom_utils
from digit_depth.digit import DigitSensor
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model, find_background_img
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = Path(__file__).parent.parent.resolve()


class ImageFeature:
    def __init__(self):
        self.image_pub = rospy.Publisher("/digit/depth/image_raw/",
                                         Image, queue_size=10)
        self.br = CvBridge()


@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def show_depth(cfg):
    model_path = find_recent_model(base_path)
    model = torch.load(model_path).to(device)
    model.eval()
    ic = ImageFeature()
    br = CvBridge()

    rospy.init_node('depth_node', anonymous=True)
    # base image depth map
    background_img_path = find_background_img(base_path)
    background_img = cv2.imread(background_img_path)
    background_img = preproc_mlp(background_img)
    background_img_proc = model(background_img).cpu().detach().numpy()
    background_img_proc, _ = post_proc_mlp(background_img_proc)
    # get gradx and grady
    gradx_base, grady_base = geom_utils._normal_to_grad_depth(img_normal=background_img_proc, gel_width=cfg.sensor.gel_width,
                                                              gel_height=cfg.sensor.gel_height, bg_mask=None)

    # reconstruct depth
    img_depth_back = geom_utils._integrate_grad_depth(gradx_base, grady_base, boundary=None, bg_mask=None,
                                                      max_depth=cfg.max_depth)
    img_depth_back = img_depth_back.detach().cpu().numpy() # final depth image for base image
    # setup digit sensor
    digit = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
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
        img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None,max_depth=cfg.max_depth)
        img_depth = img_depth.detach().cpu().numpy() # final depth image for current image
        # get difference
        diff = img_depth_back - img_depth
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

