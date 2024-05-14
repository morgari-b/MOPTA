import sys
import os# Change the current working directory
#os.chdir('/home/frulcino/codes/MOPTA/')

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox, QFormLayout,  QTabWidget)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

#import our functions:
from scenario_generation import read_parameters, SG_beta, SG_weib, quarters_df_to_year
from prova_bianca import OPT


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
    
    def __init__(self, n_scenarios, country, params): #qua posso mettere altri parameteri di generazione, tipo se usare scenare già fatti o meno
        super().__init__()
        self.n_scenarios = n_scenarios
        self.country = country
        self.params = params
        self.result = None #read only after SG
        

    def run(self):
        self.progress.emit(f"Generating Wind scenarios for {self.country}...")
        # Generate Scenarios
        scenarios = {}
        if self.country == "Italy (pre generated)":
            n_scenarios = 100
            scenarios["PV"]=pd.read_csv('scenarios/PV_scenario100.csv',index_col=0)
            scenarios["PV"].columns = pd.to_datetime(scenarios["PV"].columns)
            scenarios["wind"]=pd.read_csv('scenarios/wind_scenarios.csv',index_col=0)
            scenarios["wind"].columns = pd.to_datetime(scenarios["PV"].columns)
        else:
            n_scenarios = self.n_scenarios
            params = self.params
            
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
        scenarios["Hydrogen_load"] = quarters_df_to_year("hydrogen-demand", "g", GL, "Load", "Location_Gas", save = False, n_scenarios = n_scenarios, n_hours = 24*365) #to change if there are more than one load
        self.result = scenarios
        self.results_ready.emit(self.result)
        self.progress.emit("Done.")
        self.finished.emit()  # Emit finished signal when done

#create tab widget
class MyTabWidget(QWidget): 
    def __init__(self, parent): 
        super(QWidget, self).__init__(parent) 
        self.layout = QVBoxLayout(self) 
  
        # Initialize tab screen 
        self.tabs = QTabWidget() 
        self.tab1 = QWidget() 
        self.tab2 = QWidget() 
        self.tab3 = QWidget() 
        self.tabs.resize(300, 200) 
  
        # Add tabs 
        self.tabs.addTab(self.tab1, "SG") 
        self.tabs.addTab(self.tab2, "Optimize") 
        self.tabs.addTab(self.tab3, "Results") 
  
        # Create first tab 
        self.tab1.layout = QVBoxLayout() 
        self.tab1.setLayout(self.tab1.layout) 
        self.tab2.layout = QVBoxLayout() 
        self.tab2.setLayout(self.tab2.layout) 
        self.tab3.layout = QVBoxLayout() 
        self.tab3.setLayout(self.tab3.layout) 
        # Add tabs to widget 
        self.layout.addWidget(self.tabs) 
        self.setLayout(self.layout) 
          
class MainWindow(QMainWindow):
    
    def __init__(self):
        
        #initialize parameters
        self.scenarios = None
        self.wind_Pmax = 4 #max wind prodution per generator
        self.PV_Pmax = 0.015 #max PV production per generator
        
        #App layout
        super().__init__()
        self.setWindowTitle("Hydrogen Network Optimization")
        self.left = 0
        self.top = 0
        self.width = 1000
        self.height = 500
        self.setGeometry(self.left, self.top, self.width, self.height) 
        
        #initialize tab widget
        self.central_widget = MyTabWidget(self)
        self.setCentralWidget(self.central_widget)

        #TAB1: Scenario Generation
        tab1 = self.central_widget.tab1.layout
        form_layout = QFormLayout()
        self.location_input = QComboBox()
        self.location_input.addItems(['Italy (pre generated)',
                                     'Austria',
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

        tab1.addLayout(form_layout)

        self.SG_button = QPushButton("Generate Scenarios")
        self.SG_button.clicked.connect(self.start_scenario_generation)
        tab1.addWidget(self.SG_button)

        self.SG_output_label = QLabel("Output will be shown here.")
        tab1.addWidget(self.SG_output_label)

        self.SG_canvas = FigureCanvas(Figure(figsize=(5, 12)))
        tab1.addWidget(self.SG_canvas)
        
        #TAB2:Optimization 
        tab2 = self.central_widget.tab2.layout
        self.submit_button = QPushButton("Optimize Network")
        self.submit_button.clicked.connect(self.run_optimization)
        tab2.addWidget(self.submit_button)
        
        # Optimization Output Area
        self.canvas = FigureCanvas(Figure(figsize=(5, 12), dpi=100))
        self.output_label = QLabel("Output will be shown here.")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab2.addWidget(self.output_label)
        tab2.addWidget(self.canvas)
        
        # Creating tab widgets 
  
        
    def start_scenario_generation(self):
        n_scenarios = int(self.n_scenarios_input.text() or 1)
        country = self.location_input.currentText()
        if country == 'Italy (pre generated)':
            self.params = read_parameters("Italy")
        else:
            self.params = read_parameters(country)
            self.SG_output_label.setText("Parameters have been imported, generating Wind scenarios, this can take a while...")
            
        self.thread = QThread()  # Create a QThread object
        self.worker = ScenarioGenerator(n_scenarios, country, self.params)  # Create a worker object
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
        """
        ES: solar el production - dxinst array
        EW: wind el production - dxinst array
        EL: electricity load - dxinst array
        HL: hydrogen load - dxinst array
        cs: cost of solar panels
        c2: cost of wind turbines
        mw: max socially acceptable wind turbines
        ch: cost of hydrogen storage???
        chte: cost of H to el
        fhte: efficiency of H to el
        Mhte: max H to el at an instance
        ceth: cost of el to H
        feth: efficiency of el to H
        Meth: max el to H at an instance
        """
        scenarios = self.scenarios
        if scenarios == None:
            self.output_label.setText("Generate Scenarios before running optimization")
        else:
            # Call the optimization function
            
            ES = self.PV_Pmax*np.matrix(scenarios["PV"])
            EW = self.wind_Pmax*np.matrix(scenarios["wind"])
            EL_dict = scenarios["Electricity_load"]
            first = True
            for location_df in EL_dict.values(): #this is only to be done when we need to aggregate locations, not sure about graph thingy
                if first: #first iteration:
                    EL = np.matrix(location_df)
                    first = False
                else:
                    EL += np.matrix(location_df)
                    
            HL = np.matrix(scenarios["Hydrogen_load"])
            result, fig = OPT(ES=ES,EW=EW,EL=EL,HL=HL,ch=hs_cost) 
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
        params = self.params
        # Create a new figure with 3 rows and 2 columns
        fig, axs = plt.subplots(3, 2, figsize=(5, 12))
        fig.suptitle('Scenarios Overview', fontsize=16)
        
        
        # Subplot for Covariance of PV values at different hours
        c_pv = axs[0, 0].imshow(params["PV"][1][0:24*3, 0:24*3], interpolation="nearest", aspect='auto')
        fig.colorbar(c_pv, ax=axs[0, 0])
        axs[0, 0].set_title("Covariance of PV values at different hours")
        axs[0, 0].set_xlabel("Hour")
        axs[0, 0].set_ylabel("Hour")
        
        # Subplot for Covariance of Wind values at different hours
        c_wind = axs[0, 1].imshow(params["wind"][1][0:24*3, 0:24*3], interpolation="nearest", aspect='auto')
        fig.colorbar(c_wind, ax=axs[0, 1])
        axs[0, 1].set_title("Covariance of Wind values at different hours")
        axs[0, 1].set_xlabel("Hour")
        axs[0, 1].set_ylabel("Hour")
        print("plotted covariance")
        
    
        # Subplot for PV and Wind Power Output through the Year
        wind_scenarios = self.scenarios["wind"]
        PV_scenarios = self.scenarios["PV"]
        n_scenarios = wind_scenarios.shape[0]
        
        
        max_iter = min(n_scenarios, 1) #plot at most 5 scenarios
        wind_scenarios = wind_scenarios.iloc[:max_iter,:]
        print(wind_scenarios.shape)
        PV_scenarios =PV_scenarios.iloc[:max_iter,:]
        colormap = cm.get_cmap('tab10', 2 * max_iter)
        for index, row in enumerate(wind_scenarios.iterrows()):
            axs[1, 0].plot(row[1].index, row[1].values, color=colormap(index), alpha=0.7)
        axs[1, 0].set_title("Wind Power Output through the Year")
        axs[1, 0].set_xlabel("Datetime")
        axs[1, 0].set_ylabel("Power Output (p.u.)")
        axs[1, 0].legend(["Wind Power"], loc='upper right')
        print("plotted wind")
        for index, row in enumerate(PV_scenarios.iterrows()):
            print(index)
            axs[1, 1].plot(row[1].index, row[1].values, color=colormap(max_iter + index), alpha=0.7)
        
        axs[1, 1].set_title("PV Power Output through the Year")
        axs[1, 1].set_xlabel("Datetime")
        axs[1, 1].set_ylabel("Power Output (p.u.)")
        axs[1, 1].legend(["PV Power"], loc='upper right')
        print("plotted pv")
        
        # Subplot for 3 Days of PV and Wind Power Output
        wind_scenario = wind_scenarios.iloc[0, 0:24*3]
        solar_scenario = PV_scenarios.iloc[0, 0:24*3]
        
        axs[2, 0].plot(wind_scenario.index, wind_scenario, color="blue", alpha=0.7)
        axs[2, 1].plot(solar_scenario.index, solar_scenario, color="yellow", alpha=0.7)
        
        axs[2, 0].set_title("Wind Output for Three Days")
        axs[2, 0].set_xlabel("Datetime")
        axs[2, 0].set_ylabel("Power Output (p.u.)")
        axs[2, 0].legend(["Wind Power"], loc='upper right')
        
        axs[2, 1].set_title("PV Output for Three Days")
        axs[2, 1].set_xlabel("Datetime")
        axs[2, 1].set_ylabel("Power Output (p.u.)")
        axs[2, 1].legend(["PV Power"], loc='upper right')
        
        print("plotted the rest")
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])
          
        self.SG_canvas.figure.clf()
        self.SG_canvas.figure = fig
        self.SG_canvas.draw()
        print("drew plot")

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
