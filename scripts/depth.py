""" Publishes a ROS topic with name /digit//depth/image_raw.
    Be sure to tune the parameters in rqt_image_view for better visualization."""
import cv2
import hydra
import rospy
from pathlib import Path
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from digit_depth.third_party import geom_utils
from digit_depth.digit import DigitSensor
from digit_depth.train.prepost_mlp import *
from digit_depth.handlers import find_recent_model
from digit_depth.third_party.vis_utils import ContactArea
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = Path(__file__).parent.parent.resolve()


@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def show_depth(cfg):
    depth_pub = rospy.Publisher("/digit/depth/image_raw/",
                                    Image, queue_size=10)
    br = CvBridge()
    model_path = find_recent_model(f"{base_path}/models")
    model = torch.load(model_path).to(device)
    model.eval()

    rospy.init_node('depth_node', anonymous=True)
    digit = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
    digit_call = digit()
    dm_zero_counter = 0
    dm_zero = 0
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
        # Get the first 50 frames and average them to get the zero depth
        if dm_zero_counter < 50:
            dm_zero += img_depth
            dm_zero_counter += 1
            continue
        elif dm_zero_counter == 50:
            dm_zero = dm_zero/50
            dm_zero_counter += 1
        
        # remove the zero depth
        diff = img_depth - dm_zero
        # convert pixels into 0-255 range
        diff = diff*255
        diff = diff*-1

        ret, thresh4 = cv2.threshold(diff, 0, 255, cv2.THRESH_TOZERO)
        if cfg.visualize.ellipse:
            img = thresh4
            pt = ContactArea()
            theta = pt.__call__(target=thresh4)
            msg = br.cv2_to_imgmsg(img, encoding="passthrough")
            msg.header.stamp = rospy.Time.now()
            depth_pub.publish(msg)
        else:
            msg = br.cv2_to_imgmsg(thresh4, encoding="passthrough")
            msg.header.stamp = rospy.Time.now()
            depth_pub.publish(msg)
        now = rospy.get_rostime()
        rospy.loginfo("published depth image at {}".format(now))
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.loginfo("starting...")
    show_depth()
