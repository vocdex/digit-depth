"""
Allows you to quickly check the mm-per-pixel of a DIGIT sensor
Usage:
    Press a pair of calipers on the finger of the gelsight and take a screenshot (using scripts/record.py)
    Run this script with the path to the screenshot and the distance between the calipers in mm.
    Measurements will be repeated 4 times and the average will be printed.
"""

import cv2
import math
import os
import numpy as np
import hydra
from pathlib import Path
from digit_depth.digit.digit_sensor import DigitSensor  
# Global variables
dist = None
click_a = None
total_measurements = []

base_path = Path(__file__).parent.parent.resolve()

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


@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def main(cfg):
    digit_sensor = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
    digit_call = digit_sensor()
    if not os.path.exists(f"{base_path}/mm_to_pix"):
        os.makedirs(f"{base_path}/mm_to_pix")
    global dist, click_a, img, total_measurements
    click_a = None
    total_measurements = []
    while True:
        frame = digit_call.get_frame()
        cv2.imshow(f"{digit_call.serial}", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC hit
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACEBAR hit
            img_name = "{}/frame_{:0>1}.png".format("mm_to_pix", 0)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img = cv2.imread(img_name)
            break
    cv2.destroyAllWindows()
    img_path = f"{base_path}/mm_to_pix/frame_0.png"
    img = cv2.imread(img_path)
    dist = float(input("Enter the distance between the calipers in mm: "))
    for i in range(4):
        cv2.imshow("Click the two points of the calipers", img)
        cv2.setMouseCallback("Click the two points of the calipers", click_cb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("Average mm per pixel: {}".format(np.mean(total_measurements)))

if __name__ == "__main__":
    main()

