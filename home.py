from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


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
if __name__=='__main__':
    window()