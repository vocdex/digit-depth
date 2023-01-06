"""
Allows you to quickly check the meter-per-pixel
of a DIGIT sensor
Directions:
Press a pair of calipers on the finger of the gelsight and take a screenshot.
Run this script with the length and image path as arguments:
rosrun gelsight_ros get_mpp.py _dist_mm:=(dist in mm) _img_path:=(path to img)
Click the two points of calipers and check the command-line for the value.
"""
import cv2
import math
import argparse
import os
from digit_depth.digit import DigitSensor
from record import record_frame
# Global variables
dist = None
click_a = None

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def click_cb(event, x, y, flags, param):
    global dist, click_a
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_a is None:
            click_a = (x, y)
        else:
            px_dist = math.sqrt((x-click_a[0])**2 + (y-click_a[1])**2)
            print(f"MPP: {dist/px_dist}")
            exit()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fps", type=int, default=30, required=False, help="Frames per second. Max:60 on QVGA")
    argparser.add_argument("--resolution", type=str, default="QVGA", required=False, help="QVGA, VGA")
    argparser.add_argument("--serial_num", type=str, default="D00003", required=False, help="Serial number of DIGIT")
    args = argparser.parse_args()
    digit = DigitSensor(args.fps, args.resolution, args.serial_num)
    record_frame(digit, os.path.join(base_path, "mm_to_pix"))
    img = cv2.imread(os.path.join(base_path, "mm_to_pix", "frame_0.png"))
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", click_cb)
    while True:
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC hit
            print("Escape hit, closing...")
            break
    cv2.destroyAllWindows()
