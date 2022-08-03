"""
Script for capturing individual frames while the camera output is displayed.
-- Press SPACEBAR to capture
-- Press ESC to terminate the program.
"""
import cv2
import os
import os.path
import argparse
from digit.digit_sensor import DigitSensor


def record_frame(digit_sensor,dir_name:str):
    img_counter=len(os.listdir(dir_name))
    digit_call = digit_sensor()
    while True:
        frame = digit_call.get_frame()
        cv2.imshow('Capture Frame',frame)
        k=cv2.waitKey(1)
        if k%256==27:
            #ESC hit
            print("Escape hit, closing...")
            break
        elif k%256==32:
            #SPACEBAR hit
            img_name="{}/{:0>4}.png".format(dir_name,img_counter)
            cv2.imwrite(img_name,frame)
            print("{} written!".format(img_name))
            img_counter+=1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    argparser=argparse.ArgumentParser()
    argparser.add_argument('--fps',type=int,default=30)
    argparser.add_argument('--resolution',type=str,default="QVGA")
    argparser.add_argument('--serial_num',type=str,default="D20001")
    argparser.add_argument('--dir_name',type=str,default="digit_depth/qt_images")
    args=argparser.parse_args()
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
        print("Directory {} created for saving images".format(args.dir_name))
    digit = DigitSensor(args.fps, args.resolution, args.serial_num)

    record_frame(digit, args.dir_name)