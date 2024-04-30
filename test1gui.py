import sys
import os# Change the current working directory
os.chdir('/home/frulcino/codes/MOPTA/')

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFormLayout)
from PyQt6.QtCore import Qt
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

#import our functions:
from opt import run_OPT

# %%
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hydrogen Network Optimization")
        self.setGeometry(100, 100, 400, 600)  # x, y, width, height

        # Central Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Form layout for inputs
        form_layout = QFormLayout()
        
        form_layout.addRow(QLabel("A"))
        self.hydrogen_cost_input = QLineEdit()
        self.helectrolyzer_cost_input = QLineEdit()
        self.location_input = QComboBox()
        self.location_input.addItems(['Location A', 'Location B', 'Location C'])
        self.capacity_input = QLineEdit()
        self.purity_input = QLineEdit()
        
        form_layout.addRow("Hydrogen Storage Unit Cost ($/kg):", self.hydrogen_cost_input)
        form_layout.addRow("Hydrogen Electrolyzer Cost ($/unit):", self.helectrolyzer_cost_input)
        form_layout.addRow("Geographical Location:", self.location_input)
        form_layout.addRow("Production Capacity (kg/day):", self.capacity_input)
        form_layout.addRow("Purity Level (%):", self.purity_input)
        
        self.layout.addLayout(form_layout)
        # Submit button
        self.submit_button = QPushButton("Optimize Network")
        self.submit_button.clicked.connect(self.run_optimization)
        self.layout.addWidget(self.submit_button)

        ## Output area ##
        
         # Initial figure and canvas setup
        self.canvas = FigureCanvas(Figure(figsize=(5, 5), dpi=100))
       
        # Initial plot
        
        
        # SetUp text
        self.output_label = QLabel("Output will be shown here.")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.canvas)
        
    def change_figure(self,fig):
        self.canvas.figure.clf()
        self.canvas.figure = fig
        self.canvas.draw()
        
        
    def run_optimization(self):
        hydrogen_cost = self.hydrogen_cost_input.text()
        location = self.location_input.currentText()
        capacity = self.capacity_input.text()
        purity = self.purity_input.text()

        # Call the optimization function
        result, fig = run_OPT()
        self.change_figure(fig)
        
        # Display the results
        #self.output_label.setText(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
