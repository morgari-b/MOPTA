#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:05:16 2024

@author: frulcino
"""


from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

import sys

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.button = QPushButton("Press Me!")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.the_button_was_clicked)

        self.setFixedSize(QSize(400,300))
        # Set the central widget of the Window.
        self.setCentralWidget(self.button)
        
    def the_button_was_clicked(self, checked):
        if checked == True:
            self.button.setText("I love you <3")
        else:
            self.button.setText("Press Me again!")
        

app = QApplication(sys.argv) #if it takes commands from commandline

window = MainWindow()
window.show()

app.exec()