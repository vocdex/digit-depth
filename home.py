import glob
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2
import math
import csv


def window():
    app=QApplication(sys.argv)
    win=QMainWindow()
    win.setGeometry(500,200,500,500) # x,y,w,h => x,y are where one screen is located, w,h are the size of the screen
    win.setWindowTitle('PyQt5 Tutorial')
    label=QtWidgets.QLabel(win)
    label.setText('Hello World')
    label.move(100,100)
    win.show()
    sys.exit(app.exec_()) # this will make the program to run until the user closes the window

center_x,center_y = [],[]
circumference_x,circumference_y= [],[]
radii=[]
columns=['img_name', 'center_x', 'center_y', 'radius']
filename='csv/annotate.csv'

def click_and_store(event,x,y,flags,param):
    global center_x, center_y, circumference_x, circumference_y, radius
    if event==cv2.EVENT_LBUTTONDOWN:
        center_x.append(x)
        center_y.append(y)
        print("center: ",x,y)
        cv2.circle(image,(x,y),1,(0,0,255),-1)
        cv2.imshow('image',image)
    elif event==cv2.EVENT_RBUTTONDOWN:
        circumference_x.append(x)
        circumference_y.append(y)
        print("circumference: ",x,y)
        cv2.circle(image,(x,y),1,(0,0,255),-1)
        cv2.imshow('image',image)
    elif circumference_x and center_x and event!= cv2.EVENT_MOUSEMOVE:
        radius=math.sqrt((center_x[0]-circumference_x[0])**2+(center_y[0]-circumference_y[0])**2)
        print("radius: ",int(radius))
        radii.append(int(radius))
        cv2.circle(image,(center_x[0],center_y[0]),int(radius),(255,100,255),-1)
        cv2.imshow('image',image)
    with open(filename, 'a') as csv_file:
        if event != cv2.EVENT_MOUSEMOVE:
            if center_x and center_y and circumference_x and circumference_y and radii:
               writer = csv.writer(csv_file)
               writer.writerow([center_x[0], center_y[0], radii[0]])
               center_x.pop()
               center_y.pop()
               circumference_x.pop()
               circumference_y.pop()
               radii.pop()

if __name__=='__main__':
    img_files=sorted(glob.glob(f'qt_images/*.png'))
    for img in img_files:
        image=cv2.imread(img)
        cv2.imshow('image',image)
        cv2.setMouseCallback('image',click_and_store,image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()