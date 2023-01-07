""" ROS RGB image publisher for DIGIT sensor """

import argparse
# OpenCV
from cv_bridge import CvBridge
# Ros libraries
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
from digit_depth.digit.digit_sensor import DigitSensor


class ImageFeature:
    def __init__(self):
        # topic where we publish

        self.image_pub = rospy.Publisher("/digit/rgb/image_raw/compressed",
                                         CompressedImage, queue_size=10)
        self.br = CvBridge()


def rgb_pub(digit_sensor: DigitSensor):
    # Initializes and cleanup ros node
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fps", type=int, default=30)
    argparser.add_argument("--resolution", type=str, default="QVGA")
    argparser.add_argument("--serial_num", type=str, default="D00003")
    args, unknown = argparser.parse_known_args()
    digit = DigitSensor(args.fps, args.resolution, args.serial_num)
    rgb_pub(digit)
