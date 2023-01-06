"""
Allows you to quickly check the mm-per-pixel of a DIGIT sensor
Usage:
    Press a pair of calipers on the finger of the gelsight and take a screenshot (using scripts/record.py)
    Run this script with the path to the screenshot and the distance between the calipers in mm.
    Measurements will be repeated 4 times and the average will be printed.
"""

import cv2
import math
import argparse
import os
from record import record_frame
from digit_depth.digit import DigitSensor
# Global variables
dist = None
click_a = None
total_measurements = []

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def click_cb(event, x, y, flags, param):
    global dist, click_a
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_a is None:
            click_a = (x, y)
            cv2.circle(img, click_a, 5, (0,255,0), -1)
            cv2.imshow("Click the two points of the calipers", img)
        else:
            click_b = (x, y)
            cv2.circle(img, click_b, 5, (0,0,255), -1)
            cv2.line(img, click_a, click_b, (255,0,0), 2)
            px_dist = math.sqrt((x-click_a[0])**2 + (y-click_a[1])**2)
            mm_to_px = px_dist / dist
            print("Meters per pixel: {}".format(mm_to_px))
            cv2.imshow("Click the two points of the calipers", img)
            total_measurements.append(mm_to_px)
            click_a = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_mm", type=float, required=True, help="Distance between 2 measure points in mm")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second. Max:60 on QVGA")
    parser.add_argument("--resolution", type=str, default="QVGA", help="QVGA, VGA")
    parser.add_argument("--serial_num", type=str, default="D00003", help="Serial number of DIGIT")
    args = parser.parse_args()
    digit = DigitSensor(args.fps, args.resolution, args.serial_num)
    dist = args.dist_mm
    dir_path = os.path.join(base_path, "mm_to_pix")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print("Directory {} created for saving mm_to_pix images".format(dir_path))
    record_frame(digit, os.path.join(base_path, "mm_to_pix"))
    img_path = os.path.join(base_path, "mm_to_pix", "frame_1.png")
    for i in range(4):
        img = cv2.imread(img_path)
        cv2.imshow("Click the two points of the calipers", img)
        cv2.setMouseCallback("Click the two points of the calipers", click_cb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("Average Meters per pixel: {}".format(sum(total_measurements)/len(total_measurements)))

