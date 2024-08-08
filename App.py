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

#%%
# import dash
import os
os.chdir("C:/Users/ghjub/codes/MOPTA")
from dash import dcc, html, Input, Output, State, dash_table, DiskcacheManager
import dash
import plotly.graph_objects as go

#%%
# import dash
import os
os.chdir("C:/Users/ghjub/codes/MOPTA")
from dash import dcc, html, Input, Output, State, dash_table, DiskcacheManager
import dash
import plotly.graph_objects as go
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

class Network:
    """
    Network class, saving parameters of model and having optimization methods
    cs=4000, cw=3000000,ch=10,Mns=np.inf,Mnw=np.inf,Mnh=np.inf,chte=2,fhte=0.75,Mhte=np.inf,ceth=200,feth=0.7,Meth=np.inf
    """
    def __init__(self):
        #cs=4000,cw=3000000,Mns=np.inf,Mnw=100,Mnh=np.inf,ch=7,chte=0,fhte=0.75,Mhte=200000,ceth=0,feth=0.7,Meth=15000
        self.cs = 4000
        self.cw = 3000000
        self.ch = 10
        self.Mns = np.inf
        self.Mnw = np.inf
        self.Mnh = np.inf
        self.chte = 2
        self.Mhte = np.inf
        self.ceth = 200
        self.feth = 0.7
        self.fhte=0.75
        self.Meth = np.inf
        self.PV_Pmax = 0.015
        self.wind_Pmax = 4
        self.n_scenarios_opt = 5

    def run_optimization(self, ES, EW, EL, HL):
        ES = ES*self.PV_Pmax
        EW = EW*self.wind_Pmax
        result, fig, status = OPT(ES=ES, EW=EW, EL=EL, HL=HL, cs=self.cs, cw=self.cw, Mns=self.Mns, Mnw=self.Mnw, Mnh=self.Mnh, ch=self.ch, chte=self.chte, fhte=self.fhte, Mhte=self.Mhte, ceth=self.ceth, feth=self.feth, Meth=self.Meth)

        return result, fig, status 

class OptimizationWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    result_ready = pyqtSignal(object, object, object)  # Emit result and figure

    def __init__(self, scenarios, network):
        super().__init__()
        self.scenarios = scenarios
        self.network = network
        self.n_scenarios_opt = network.n_scenarios_opt

    def run(self):
        # Perform the optimization
        ES = np.matrix(self.scenarios["PV"].iloc[:self.n_scenarios_opt,:])
        print(f"ES shape is {ES.shape} while n_scenarios is {self.n_scenarios_opt}")
        EW =  np.matrix(self.scenarios["wind"].iloc[:self.n_scenarios_opt,:])
        EL_dict = self.scenarios["Electricity_load"]
        first = True
        for location_df in EL_dict.values():
            if first:
                EL = np.matrix(location_df.iloc[:self.n_scenarios_opt,:])
                first = False
            else:
                EL += np.matrix(location_df.iloc[:self.n_scenarios_opt,:])

        HL = np.matrix(self.scenarios["Hydrogen_load"].iloc[:self.n_scenarios_opt,:])
        self.progress.emit(f"Creating and solving model with {self.n_scenarios_opt} scenarios, this should take around {int(self.n_scenarios_opt / 5 * 30)} seconds...")
        result, fig , status= self.network.run_optimization(ES,EW,EL,HL)

        # Emit the result
        self.result_ready.emit(result, fig, status)
        self.finished.emit()


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
        
        # Generate Scenarios
        scenarios = {}
        if self.country == "Italy (pre generated)":
            self.progress.emit(f"Fetching scenarios for {self.country}...")
            n_scenarios = 100
            scenarios["PV"]=pd.read_csv('scenarios/PV_scenario100.csv',index_col=0)
            scenarios["PV"].columns = pd.to_datetime(scenarios["PV"].columns)
            scenarios["wind"]=pd.read_csv('scenarios/wind_scenarios.csv',index_col=0)
            scenarios["wind"].columns = pd.to_datetime(scenarios["PV"].columns)
        else:
            self.progress.emit(f"Generating scenarios, this can take a while...")
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
        self.network = Network()
        self.scenarios = None
        
        
        #App layout
        super().__init__()
        self.setWindowTitle("Hydrogen Network Optimization")
        self.left = 0
        self.top = 0
        self.width = 1000
        self.height = 800
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
        
        #TAB2:Optimization, ask parameters from user:
        """
        todo: would be nice if it run more than once with different parameters which radically change network behavior:
            - high storage costs incentivizes 
        cs: cost of solar panels
        Mnw: cost of wind turbines
        Mns: max socially acceptable wind turbines
        ch: cost of hydrogen storage???
        chte: cost of H to el
        fhte: efficiency of H to el
        Mhte: max H to el at an instance
        ceth: cost of el to H
        feth: efficiency of el to H
        Meth: max el to H at an instance
        """
        
        tab2 = self.central_widget.tab2.layout
        opt_form_layout = QFormLayout()
        self.cs_input = QLineEdit()
        self.cw_input = QLineEdit()
        self.Mnw_input = QLineEdit()
        self.Mns_input = QLineEdit()
        self.ch_input = QLineEdit()
        self.chte_input = QLineEdit()
        self.ceth_input = QLineEdit()
        self.Meth_input = QLineEdit()
        self.wind_Pmax_input = QLineEdit()
        self.PV_Pmax_input = QLineEdit()
        self.n_scenarios_opt_input = QLineEdit()
        opt_form_layout.addRow(f"Cost of Solar panels (euro/unit), default value: {self.network.cs}", self.cs_input)
        opt_form_layout.addRow(f"Cost of SWind turbines (euro/unit), default value: {self.network.cw}", self.cw_input)
        opt_form_layout.addRow(f"Maximum number of Wind turbines, default value: {self.network.Mnw}", self.Mnw_input)
        opt_form_layout.addRow(f"Maximum number of Solar panels, default value: {self.network.Mns}", self.Mns_input)
        opt_form_layout.addRow(f"Cost of Hydrogen Storage (euro/kg), default value: {self.network.ch}", self.ch_input)
        opt_form_layout.addRow(f"Cost of Power Cell operation (euro/Kg), default value:  {self.network.chte}", self.chte_input)
        opt_form_layout.addRow(f"Cost of Electrolyzer operation, default value: {self.network.ceth}", self.ceth_input)
        opt_form_layout.addRow(f"Maximum Electrolyzer Capacity (kg/h), default value: {self.network.Meth}", self.Meth_input)
        opt_form_layout.addRow(f"Maximum Power output of a Wind turbine (MWh), default value: {self.network.wind_Pmax} ", self.wind_Pmax_input)
        opt_form_layout.addRow(f"Maximum Power output of a Solar panel (MWh), default value: {self.network.PV_Pmax} ", self.PV_Pmax_input)
        opt_form_layout.addRow(f"Number of scenarios to run for Optimization, default value: {self.network.n_scenarios_opt}",self.n_scenarios_opt_input)
        
        tab2.addLayout(opt_form_layout)
        
        
        self.output_label = QLabel("Press button to solve Model.")
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        tab2.addWidget(self.output_label)
        self.submit_button = QPushButton("Optimize Network")
        self.submit_button.clicked.connect(self.run_optimization)
        tab2.addWidget(self.submit_button)
        
        
        #TAB3:Results
        tab3 = self.central_widget.tab3.layout
        self.results_label = QLabel("Results will be shown here.")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab3.addWidget(self.results_label)
        # Optimization Output Area, Plots and explanation
        self.canvas = FigureCanvas(Figure(figsize=(5, 12), dpi=100))
        tab3.addWidget(self.canvas)
        
        # Creating tab widgets 
  
        
    def start_scenario_generation(self):
        n_scenarios = int(self.n_scenarios_input.text() or 5)
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
        
        #Set network parameters
        #self.network = Network() #reinitialize network
        self.network.cs = float(self.cs_input.text() or self.network.cs)
        self.network.cw = float(self.cw_input.text() or self.network.cw)
        self.network.Mnw = float(self.Mnw_input.text() or self.network.Mnw)
        self.network.Mns = float(self.Mns_input.text() or self.network.Mns)
        self.network.ch = float(self.ch_input.text() or self.network.ch)
        self.network.chte = float(self.chte_input.text() or self.network.chte)
        self.network.ceth = float(self.ceth_input.text() or self.network.ceth)
        self.network.Meth = float(self.Meth_input.text() or self.network.Meth)
        self.network.wind_Pmax = float(self.wind_Pmax_input.text() or self.network.wind_Pmax)
        self.network.PV_Pmax = float(self.PV_Pmax_input.text() or self.network.PV_Pmax)
        
        self.network.n_scenarios_opt = int(self.n_scenarios_opt_input.text() or self.network.n_scenarios_opt)
        scenarios = self.scenarios
        if scenarios is None:
            self.output_label.setText("Generate Scenarios before running optimization")
        else:
            # Create the worker and thread
            self.optimization_thread = QThread()
            self.optimization_worker = OptimizationWorker(scenarios, self.network)
            self.optimization_worker.moveToThread(self.optimization_thread)
    
            # Connect signals
            self.optimization_thread.started.connect(self.optimization_worker.run)
            self.optimization_worker.finished.connect(self.optimization_thread.quit)
            self.optimization_worker.finished.connect(self.optimization_worker.deleteLater)
            self.optimization_thread.finished.connect(self.optimization_thread.deleteLater)
            self.optimization_worker.progress.connect(self.update_optimization_label)
            self.optimization_worker.result_ready.connect(self.update_optimization_result)
    
            # Start the thread
            self.optimization_thread.start()

    def update_optimization_label(self, text):
        #updates optimization label
        self.output_label.setText(text)
        
    def update_optimization_result(self, result, fig, status):
        #updates scenario generationlabel
        if status == 2:
            self.output_label.setText("Optimization Succesful.")
            self.canvas.figure.clf()
            self.canvas.figure = fig
            self.canvas.draw()
            self.canvas.figure.tight_layout()
            self.canvas.resize(self.canvas.size())
            self.canvas.updateGeometry()
            generators_cost = result[0]*self.network.cs + result[1]*self.network.cw + result[2]*self.network.ch
            operation_cost = result[5].sum()*self.network.ceth + result[6].sum()*self.network.chte
            print(operation_cost)
            self.output_label.setText(f"Number of solar panels: {int(result[0])} \nNumber of Wind Turbines: {int(result[1])} \nHydrogen Storage Capacity (Kg): {result[2]} \nPower Cells Capacity (Kg/h): {result[3]} \nElectrolyzers Capacity (MWh): {result[3]}"+
                                       f"\nCost of Generators: {generators_cost} euros, One year Operation costs: {int(operation_cost)} ")
        else:
            self.output_label.setText("Optimization failed, problem is unfeasible.")
            

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
        
        
        max_iter = min(n_scenarios, 4) #plot at most 5 scenarios
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
        axs[1, 1].set_xlabel("Date")
        axs[1, 1].set_ylabel("Power Output (capacity factor)")
        axs[1, 1].legend(["PV Power"], loc='upper right')
        print("plotted pv")
        
        # Subplot for 3 Days of PV and Wind Power Output
        wind_scenario = wind_scenarios.iloc[0:max_iter, 0:24*3]
        solar_scenario = PV_scenarios.iloc[0:max_iter, 0:24*3]
        for index, row in enumerate(wind_scenario.iterrows()):
            axs[2, 0].plot(row[1].index, row[1].values,  color=colormap(index), alpha=0.7)
            
        for index, row in enumerate(solar_scenario.iterrows()):
            axs[2, 1].plot(row[1].index, row[1].values,  color=colormap(max_iter + index), alpha=0.7)
        
        axs[2, 0].set_title("Wind Output for Three Days")
        axs[2, 0].set_xlabel("Date")
        axs[2, 0].set_ylabel("Power Output (capacity factor)")
        axs[2, 0].legend(["Wind Power"], loc='upper right')
        
        axs[2, 1].set_title("PV Output for Three Days")
        axs[2, 1].set_xlabel("Date")
        axs[2, 1].set_ylabel("Power Output (capacity factor)")
        axs[2, 1].legend(["PV Power"], loc='upper right')
        
        print("plotted the rest")
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])
          
        self.SG_canvas.figure.clf()
        self.SG_canvas.figure = fig
        self.SG_canvas.draw()
        self.SG_canvas.figure.tight_layout()
        self.SG_canvas.resize(self.canvas.size())
        self.SG_canvas.updateGeometry()
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
