""" ROS RGB image publisher for DIGIT sensor """

import hydra
from cv_bridge import CvBridge
from pathlib import Path
import rospy
from sensor_msgs.msg import CompressedImage
from digit_depth.digit.digit_sensor import DigitSensor

base_path = Path(__file__).parent.parent.parent.resolve()


class ImageFeature:
    def __init__(self):
        self.image_pub = rospy.Publisher("/digit/rgb/image_raw/compressed",
                                         CompressedImage, queue_size=10)
        self.br = CvBridge()

@hydra.main(config_path=base_path / "config", config_name="digit.yaml")
def rgb_pub(cfg):
    digit_sensor = DigitSensor(cfg.sensor.fps, "QVGA", cfg.sensor.serial_num)
    ic = ImageFeature()
    rospy.init_node('image_feature', anonymous=True)
    digit_call = digit_sensor()
    br = CvBridge()
    while True:
        frame = digit_call.get_frame()
        msg = br.cv2_to_compressed_imgmsg(frame, "png")
        msg.header.stamp = rospy.Time.now()
        ic.image_pub.publish(msg)
        rospy.loginfo("Published image")


if __name__ == "__main__":
    rgb_pub()