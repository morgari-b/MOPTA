import sys
import os# Change the current working directory
#os.chdir('/home/frulcino/codes/MOPTA/')

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFormLayout)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
import numpy as np

#import our functions:
from scenario_generation import read_parameters, SG_beta, SG_weib, quarters_df_to_year
from opt import run_OPT


# %%
"""
TODOS:
    - Generalize scenario generation:
        - More years
        - Grouped 
        - Option for running on only Challenge Days
"""

# Worker class that handles the long-running task
class ScenarioGenerator(QObject):
    finished = pyqtSignal()  # Signal to indicate completion
    progress = pyqtSignal(str)  # Signal to send progress text back to GUI
    results_ready = pyqtSignal(object)
    
    def __init__(self, n_scenarios, country, params):
        super().__init__()
        self.n_scenarios = n_scenarios
        self.country = country
        self.params = params
        self.result = None #read only after SG
        

    def run(self):
        self.progress.emit(f"Generating Wind scenarios for {self.country}...")
        # Generate Scenarios
        n_scenarios = self.n_scenarios
        params = self.params
        scenarios = {}
        scenarios["wind"] = SG_weib(n_scenarios, params["wind"][0], params["wind"][1])
        self.progress.emit(f"Wind done, generating PV scenarios for {self.country}...")
        scenarios["PV"] = SG_beta(n_scenarios, parameters_df = params["PV"][0], E_h = params["PV"][1], night_hours = params["PV"][2], save = False)
        # After processing
        self.progress.emit("Scenarios generated, fetching Electricity loads...")
        #Fetch loads: 
        path = "data.xlsx"
        EL=pd.read_excel(path,sheet_name='Electricity Load')
        GL=pd.read_excel(path,sheet_name='Gas Load')
        locations = EL["Location_Electricity"].unique()
        scenarios["Electricity_load"] = {}
        scenarios["Hydrogen_load"] = {}
        for location in locations:
            #loads are provided by one single day for season, for each location, we expand the days to each day in the season for all year
            scenarios["Electricity_load"][location] = quarters_df_to_year("electric-demand", location, EL, "Load", "Location_Electricity", save = False, n_scenarios = n_scenarios, n_hours = 24*365)
        #%%
        self.progress.emit("Scenarios generated, fetching Hydrogen loads...")
        scenarios["Hydrogen_load"]["g"] = quarters_df_to_year("hydrogen-demand", "g", GL, "Load", "Location_Gas", save = False, n_scenarios = n_scenarios, n_hours = 24*365)
        self.result = scenarios
        self.results_ready.emit(self.result)
        self.progress.emit("Done.")
        self.finished.emit()  # Emit finished signal when done

class MainWindow(QMainWindow):
    def __init__(self):
        
        #initialize parameters
        self.scenarios = None
        
        #App layout
        super().__init__()
        self.setWindowTitle("Hydrogen Network Optimization")
        self.setGeometry(100, 100, 400, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        form_layout = QFormLayout()
        self.location_input = QComboBox()
        self.location_input.addItems(['Austria',
                                     'Belgium',
                                     'Bulgaria',
                                     'Cyprus',
                                     'Czech Republic',
                                     'Germany',
                                     'Denmark',
                                     'Estonia',
                                     'Spain',
                                     'Finland',
                                     'France',
                                     'Greece',
                                     'Croatia',
                                     'Hungary',
                                     'Italy',
                                     'Lithuania',
                                     'Luxembourg',
                                     'Latvia',
                                     'Malta',
                                     'Netherlands',
                                     'Poland',
                                     'Portugal',
                                     'Romania',
                                     'Sweden',
                                     'Slovenia',
                                     'Slovakia'])  # Countries
        form_layout.addRow("Geographical Location:", self.location_input)

        self.n_scenarios_input = QLineEdit()
        form_layout.addRow("Number of Scenarios:", self.n_scenarios_input)

        self.layout.addLayout(form_layout)

        self.SG_button = QPushButton("Generate Scenarios")
        self.SG_button.clicked.connect(self.start_scenario_generation)
        self.layout.addWidget(self.SG_button)

        self.SG_output_label = QLabel("Output will be shown here.")
        self.layout.addWidget(self.SG_output_label)

        self.SG_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.layout.addWidget(self.SG_canvas)
        
        # Optimization Button
        self.submit_button = QPushButton("Optimize Network")
        self.submit_button.clicked.connect(self.run_optimization)
        self.layout.addWidget(self.submit_button)
        
        # Optimization Output Area
        self.canvas = FigureCanvas(Figure(figsize=(5, 5), dpi=100))
        self.output_label = QLabel("Output will be shown here.")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.canvas)
        
        
        
    def start_scenario_generation(self):
        n_scenarios = int(self.n_scenarios_input.text() or 1)
        country = self.location_input.currentText()
        params = read_parameters(country)
        self.SG_output_label.setText("Parameters have been imported, generating Wind scenarios, this can take a while...")
        
        self.thread = QThread()  # Create a QThread object
        self.worker = ScenarioGenerator(n_scenarios, country, params)  # Create a worker object
        self.worker.moveToThread(self.thread)  # Move worker to thread
        self.thread.started.connect(self.worker.run)  # Start worker when thread starts
        self.worker.results_ready.connect(self.handle_scenarios_result) #send results to handle_scenarios_results
        self.worker.finished.connect(self.thread.quit)  # Quit thread when worker is done
        self.worker.finished.connect(self.worker.deleteLater)  # Delete worker when done
        self.thread.finished.connect(self.thread.deleteLater)  # Delete thread when done
        self.worker.progress.connect(self.update_output_label)  # Connect progress signal to update the GUI
        self.SG_output_label.setText("Done.")
        self.thread.start()
    
    def run_optimization(self):
        hs_cost = 7
        #hs_cost = self.hydrogen_cost_input.text() #hydrogen storage cost
        #if hs_cost != "":
        #    hs_cost = float(hs_cost)
        #else:
        #    hs_cost = 10000
        
        scenarios = self.scenarios
        if scenarios == None:
            self.output_label.setText("Generate Scenarios before running optimization")
        else:
            # Call the optimization function
            result, fig = run_OPT(ch=hs_cost) 
            self.canvas.figure.clf()
            self.canvas.figure = fig
            self.canvas.draw()
            
            # Display the results
            self.output_label.setText(str(result))

    def update_output_label(self, text):
        self.SG_output_label.setText(text)
        
    def handle_scenarios_result(self, result):
        print("Handling scenarios result")
        self.scenarios = result #save scenarios for later optimization

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())



"""
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
        SGform_layout = QFormLayout()
        SGform_layout.addRow(QLabel("Scenario Generation"))
        
        self.location_input = QComboBox()
        self.location_input.addItems(['Austria',
                                     'Belgium',
                                     'Bulgaria',
                                     'Cyprus',
                                     'Czech Republic',
                                     'Germany',
                                     'Denmark',
                                     'Estonia',
                                     'Spain',
                                     'Finland',
                                     'France',
                                     'Greece',
                                     'Croatia',
                                     'Hungary',
                                     'Italy',
                                     'Lithuania',
                                     'Luxembourg',
                                     'Latvia',
                                     'Malta',
                                     'Netherlands',
                                     'Poland',
                                     'Portugal',
                                     'Romania',
                                     'Sweden',
                                     'Slovenia',
                                     'Slovakia'])  # Add location items
        SGform_layout.addRow("Geographical Location:", self.location_input)
        self.n_scenarios_input = QLineEdit()
        SGform_layout.addRow("Number of Scenarios:", self.location_input)
        self.hydrogen_cost_input = QLineEdit()
        SGform_layout.addRow("Hydrogen Storage Unit Cost ($/kg):", self.hydrogen_cost_input)
        self.layout.addLayout(SGform_layout)
        SG_output_layout = QVBoxLayout()
        # Scenario Generation Button
        self.SG_button = QPushButton("Generate Scenarios")
        self.SG_button.clicked.connect(self.SG)
        SG_output_layout.addWidget(self.SG_button)
        
        # Scenario Generation Output Area
        self.SG_canvas = FigureCanvas(Figure(figsize=(5, 5), dpi=100))
        self.SG_output_label = QLabel("Output will be shown here.")
        self.SG_output_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        SG_output_layout.addWidget(self.SG_output_label)
        SG_output_layout.addWidget(self.SG_canvas)
        self.layout.addLayout(SG_output_layout)
        # Add the form layout to the main layout before other widgets
        
        
       
        
        # Optimization Button
        self.submit_button = QPushButton("Optimize Network")
        self.submit_button.clicked.connect(self.run_optimization)
        self.layout.addWidget(self.submit_button)
        
        # Optimization Output Area
        self.canvas = FigureCanvas(Figure(figsize=(5, 5), dpi=100))
        self.output_label = QLabel("Output will be shown here.")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.canvas)

        
    def SG(self):
        
        n_scenarios = self.n_scenarios_input.text()
        country = self.location_input.currentText()
        self.SG_output_label.setText(f"Importing PV and Wind parameters for {country}...")
        if n_scenarios == "":
            n_scenarios = 1
        else:
            n_scenarios = float(self.n_scenarios)
            
        params = read_parameters(country)
        self.SG_output_label.setText(f"Parameters have been imported, generating Wind scenarios, this can take a while...")
        EW = SG_weib(n_scenarios, params["wind"][0], params["wind"][1])
        self.SG_output_label.setText("PGenerated Wind scenarios, generating PV scenarios...")
        ES = SG_beta(n_scenarios, parameters_df = params["PV"][0], E_h = params["PV"][1], night_hours = params["PV"][2], save = False)
        
        
        
    
    def run_optimization(self):
        hs_cost = self.hydrogen_cost_input.text() #hydrogen storage cost
        if hs_cost != "":
            hs_cost = float(hs_cost)
        else:
            hs_cost = 10000
            
        
        print(type(hs_cost), hs_cost, "yeee")
        # Call the optimization function
        result, fig = run_OPT(ch=hs_cost) 
        self.canvas.figure.clf()
        self.canvas.figure = fig
        self.canvas.draw()
        
        # Display the results
        self.output_label.setText(str(result))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
"""
