"""
Script for capturing individual frames while the camera output is displayed.
-- First image captured is the background image.
-- Press SPACEBAR to capture
-- Press ESC to terminate the program.
"""
import os
import os.path
import hydra
import cv2
from pathlib import Path
from digit_depth.digit.digit_sensor import DigitSensor

base_path = Path(__file__).parent.parent.resolve()


@hydra.main(config_path=f"{base_path}/config", config_name="digit.yaml", version_base=None)
def record_frame(cfg):
    digit_sensor = DigitSensor(cfg.sensor.fps, cfg.sensor.resolution, cfg.sensor.serial_num)
    dir_name = create_dir("images")
    img_counter = len(os.listdir(dir_name))
    digit_call = digit_sensor()
    capture_background = True
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
            if capture_background:
                img_name = "{}/background.png".format(dir_name)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                capture_background = False
                print("Background image captured.")
            else:
                img_name = "{}/frame_{:0>1}.png".format(dir_name, img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
    cv2.destroyAllWindows()
    

def create_dir(dir_name:str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Directory {} created for saving images".format(dir_name))
    else:
        print("Directory {} already exists".format(dir_name))
    return dir_name


if  __name__ == "__main__":
    record_frame()
