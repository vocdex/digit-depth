from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2
import math
import numpy as np
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
if __name__=='__main__':
    # window()
    image=cv2.imread(r'/Users/shuk/PycharmProjects/digit-depth/sample_ball.png')
    cv2.imshow('image',image)
    cv2.setMouseCallback('image',click_and_store,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
drawing = False # true if mouse is pressed
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
             k = cv2.waitKey(1)
             if k == ord('r'):
                print("hi")
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
             elif k==ord('c'):
                cv2.circle(img,(int((ix+x)/2), int((iy+y)/2)),int(math.sqrt( ((ix-x)**2)+((iy-y)**2) )),(0,0,255),-1)
             elif k== ord('l'):
                cv2.line(img,(ix,iy),(x,y),(255,0,0),5)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
img = cv2.imread(r'/Users/shuk/PycharmProjects/digit-depth/sample_ball.png')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
"""