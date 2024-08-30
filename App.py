
#%%
# import dash
import os
os.chdir("C:/Users/ghjub/codes/MOPTA")
from dash import dcc, html, Input, Output, State, dash_table, DiskcacheManager
import dash
import plotly.graph_objects as go
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
# Import the necessary functions and classes
#from scenario_generation.scenario_generation import read_parameters, SG_beta, SG_weib, quarters_df_to_year
#from model.prova_bianca import OPT
from model.YUPPY import Network
from model.OPT_methods import OPT_time_partition, OPT3, OPT_agg
from model.EU_net import EU
from model.scenario_generation.scenario_generation import import_generated_scenario, generate_independent_scenarios, import_scenarios, plot_scenarios_df, scenario_to_array, country_name_to_iso
from model.OPloTs import plotOPT3_secondstage, plotOPT_time_partition, node_results_df, plotOPT_pie, plotE_balance, plotH_balance
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
 # Diskcache for non-production apps when developing locally
import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# data path
scenarios_path = "model/scenario_generation/scenarios/"
# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)



# init graphs
scenarios_nodes_graph = go.Figure()
scenarios_locations_graph = go.Figure()

#dicts for names
vars_name_d = {"ns" : "N solar panels", "nw":"N wind turbines", "mhte":"N power Cells", "mete":"N fusion cells", "node":"Node"}
 
#TODO: store this so it doesn't get calculated over and over
# Define the layout

#%% Define the layout of the app
app.layout = dbc.Container([
    dcc.Store(id='network-store'),
    dcc.Location(id='url', refresh=False),  # This will keep track of the current URL
    dbc.NavbarSimple([
        dbc.NavItem(dcc.Link('Your Energy Grid', href='/', className='nav-link')),
        dbc.NavItem(dcc.Link('Scenarios', href='/page-1', className='nav-link')),
         dbc.NavItem(dcc.Link('Optimize and Results', href='/page-2', className='nav-link')),
    ], brand='TheClowder App', color='primary', dark=True, className='mb-4'),
    html.Div(id='page-content')  # This will hold the content of the page based on URL
])

#%% define network page

define_network_page = html.Div([
     html.H2("Hydrogen Network Optimization",  className='text-center text-primary mb-4'),
        dcc.Markdown('''
        In this section you can define the network you want to optimize or you can modify one of the predefined examples by selecting it in the dropdown menu.
        '''),
     dcc.Dropdown(
            id='example-dropdown',
            options=[
                {'label': 'Define your own Energy Grid', 'value': 'define-network'},
                {'label': 'Small EU Network', 'value': 'small-eu'},
            ],
            value='Choose Example'
        ),
        html.Div(id='selected-example-output'),
       
    #define Nodes
    dbc.Row([
       
        html.H4("Create Network, add node"),
        dash_table.DataTable(id = 'nodes-table',
                            data = [],
                            columns = [{"name": i, "id": i} for i in ["node","location","lat","long","Mhte","Meth","feth","fhte","Mns","Mnw","Mnh","MP_wind","MP_solar","meanP_load","meanH_load"]],
                            editable=True,
                            fill_width=True,),
        dbc.Button("Add node", id='add-node-button', color='success', className='mr-2 mt-2'),
        ]),
    #define edges
    dbc.Row([
        html.H4("Create Network, add edges", className='mt-4'),    
        dbc.Col([
            dash_table.DataTable(id = 'edgesP-table',
                                data = [],
                                columns = [{"name": i, "id": i} for i in ['start_node', 'end_node', 'NTC']],
                            editable=True,
                            fill_width=True,),
            dbc.Button("Add P-edge", id='add-Pedge-button', color='success', className='mr-2 mt-2 mb-2'),
            
        ]),
        dbc.Col([
            dash_table.DataTable(id = 'edgesH-table',
                                data = [],
                                columns = [{"name": i, "id": i} for i in ['start_node', 'end_node', 'MH']],
                                editable=True,
                                fill_width=True,),
            dbc.Button("Add H-edge", id='add-Hedge-button', color='success', className='mr-2 mt-2 mb-2'),
        ])
    ]),
    #define costs
    dbc.Row([
        dash_table.DataTable(id = 'costs-table',
                             data =[],
                             columns = [{"name": i, "id": i} for i in ["cs", "cw","ch","ch_t","chte","ceth","cNTC","cMH"]],
                             editable=True,
                             fill_width=True,),

        dbc.Button("Add Cost", id='add-Cost-button', color='success', className='mr-2 mt-2'),
    ]),
    dbc.Row([
        html.Button('Update Network', id='select-example', className='mr-2 mt-2'),
        html.Iframe(id='map', srcDoc='', width='100%', height='500'),
    ]),
   
    
])

#%% optimize page

optimize_page = html.Div([
    html.H2("Optimization and Results", className='text-center text-primary mb-4'),
    dcc.Markdown('''
    In this page we solve the model and display the results.
    '''),
    dcc.Dropdown(id = "OPT-method-dropdown", 
                 options = [{'label': 'Hourly time resolution (slow)', 'value': 'OPT3'},
                            {'label': 'Optimize with time aggregation (less accurate but faster)', 'value': 'OPT_time_partition'}],
                 value ="Choose optimization method"
            ),
    dbc.Button('Run optimization', id='optimize-button', color='success', className='mr-2 mt-2 mb-2'),
    html.Div(id='optimization-pie-text', children=''),
    dcc.Graph(id = "generators-pie-graph"),
    dcc.Graph(id = "E-balance-graph"),
    dcc.Graph(id = "H-balance-graph"),
    dcc.Graph(id = "hydrogen-storage-graph"),
    dcc.Graph(id = "P-edge-graph"),
    
])
#%% page handling callbacks


# Define callback to update the page content based on the current URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/page-1':
        return scenarios_page
    elif pathname == '/page-2':
        return optimize_page
    else:
        return define_network_page
#%% define scenarios page  Define the layout of the scenarios page
#TODO add explanation of plots and say which is which
scenarios_page = html.Div([
    html.H2("Generate scenarios", className='text-center text-primary mb-4'),
    html.Div(children="Select number of scenarios:", className='mb-2'),
    dcc.Slider(
        id='my-slider',
        min=1,
        max=20,
        step=1,
        value=5,
        marks={str(i): str(i) for i in range(1, 20, 4)},
    ),
    html.Div(id='scenario-slider-output-container'),
    dbc.Button('Generate Scenarios', id='generate-scenarios-button', color='success', className='mr-2 mt-2'),
    
    dcc.Markdown(''' 
    The following plot, shows the p.u. wind power production scenario for each location in the network. 
    '''),
    dcc.Graph(id='scenarios-nodes-graph-wind'),
    dcc.Markdown(''' 
    The following plots, for each location plot a sample of scenarios of wind power production. 
    '''),
    dcc.Graph(id='scenarios-locations-graph-wind'),
     dcc.Markdown(''' 
    The following plot, shows the p.u. PV production scenario for each location in the network. 
    '''),
    dcc.Graph(id='scenarios-nodes-graph-PV'),
     dcc.Markdown(''' 
    The following plots, for each location plot a sample of scenarios of PV power production. 
    '''),
    dcc.Graph(id='scenarios-locations-graph-PV'),
])
#%% define network callbacs
#update dataframes after example selection
# Function to add an empty row
def add_row(n_clicks, rows, columns):
    if n_clicks is None:
        return rows

    cols = [c['id'] for c in columns]
    df = pd.DataFrame(rows, columns=cols)
    new_row = pd.DataFrame([[None] * len(columns)], columns=cols)
    df = pd.concat([df, new_row])
    return df.to_dict('records')


@app.callback(
    [
        Output('nodes-table', 'data'),
        Output('edgesP-table', 'data'),
        Output('edgesH-table', 'data'),
        Output('costs-table', 'data'),
        Output('network-store', 'data', allow_duplicate=True),
    ],
    [
        Input('example-dropdown', 'value'),
        Input('add-node-button', 'n_clicks'),
        Input('add-Pedge-button', 'n_clicks'),
        Input('add-Hedge-button', 'n_clicks'),
        Input('add-Cost-button', 'n_clicks')
    ],
    [
        State('nodes-table', 'data'),
        State('nodes-table', 'columns'),
        State('edgesP-table', 'data'),
        State('edgesP-table', 'columns'),
        State('edgesH-table', 'data'),
        State('edgesH-table', 'columns'),
        State('costs-table', 'data'),
        State('costs-table', 'columns')
    ],
    prevent_initial_call=True
)
def update_tables(example, add_node_clicks, add_Pedge_clicks, add_Hedge_clicks, add_Cost_clicks,
                  nodes_data, nodes_columns, edgesP_data, edgesP_columns, edgesH_data, edgesH_columns, costs_data, costs_columns):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("updating tables...")
    if triggered_id == 'example-dropdown':
        if example == 'small-eu':
            print("setting network to eu")
            network = EU()
            network.n.reset_index(inplace=True)
            
            return network.n.to_dict('records'), network.edgesP.to_dict('records'), network.edgesH.to_dict('records'), network.costs.to_dict('records'), network.to_dict()
        elif example == 'define-network':
            print("resetting network...")
            network = Network()
            if 'node' not in network.columns:
                network.n.reset_index(inplace=True)
            return network.n.to_dict('records'), network.edgesP.to_dict('records'), network.edgesH.to_dict('records'), network.costs.to_dict('records'), network.to_dict()

    if triggered_id == 'add-node-button':
        updated_nodes_data = add_row(add_node_clicks, nodes_data, nodes_columns)
        return updated_nodes_data, edgesP_data, edgesH_data, costs_data, dash.no_update

    if triggered_id == 'add-Pedge-button':
        updated_edgesP_data = add_row(add_Pedge_clicks, edgesP_data, edgesP_columns)
        return nodes_data, updated_edgesP_data, edgesH_data, costs_data, dash.no_update

    if triggered_id == 'add-Hedge-button':
        updated_edgesH_data = add_row(add_Hedge_clicks, edgesH_data, edgesH_columns)
        return nodes_data, edgesP_data, updated_edgesH_data, costs_data, dash.no_update

    if triggered_id == 'add-Cost-button':
        updated_costs_data = add_row(add_Cost_clicks, costs_data, costs_columns)
        return nodes_data, edgesP_data, edgesH_data, updated_costs_data, dash.no_update

    return nodes_data, edgesP_data, edgesH_data, costs_data, dash.no_update

@app.callback(
    Output('map', 'srcDoc'),
    Output('network-store', 'data', allow_duplicate=True),
    Input('select-example', 'n_clicks'),
    State('nodes-table', 'data'),
    State('nodes-table', 'columns'),
    State('edgesP-table', 'data'),
    State('edgesP-table', 'columns'),
    State('edgesH-table', 'data'),
    State('edgesH-table', 'columns'),
    State('costs-table', 'data'),
    State('costs-table', 'columns'),
    State('network-store', 'data'),
    prevent_initial_call=True
)
def update_map(n_clicks, nodes_data, nodes_columns, edgesP_data, edgesP_columns, edgesH_data, edgesH_columns, costs_data, costs_columns, network_data):
    
    def dict_to_df(rows, columns):
        cols = [c['id'] for c in columns]
        df = pd.DataFrame(rows, columns=cols)
        return df
    def cols_to_float(df, cols):
        for col in cols:
            df[col] = df[col].astype(float)
        return df
    if n_clicks is None:
        return '', dash.no_update

    network = Network.from_dict(network_data)
    #print(f"nodes data: {nodes_data}, nodes columns: {nodes_columns}")
    network.n = dict_to_df(nodes_data, nodes_columns)
    network.n = cols_to_float(network.n, ['lat', 'long', 'Mhte', 'Meth', 'feth', 'fhte', 'Mns', 'Mnw', 'Mnh'])
    print(network.n)
    if 'node' not in network.n.columns:
        network.n.reset_index(inplace=True)
    print(network.n)
    network.edgesP = dict_to_df(edgesP_data, edgesP_columns)
    network.edgesP = cols_to_float(network.edgesP, ['NTC'])
    network.edgesH = dict_to_df(edgesH_data, edgesH_columns)
    network.edgesH = cols_to_float(network.edgesH, ['MH'])
    network.costs = dict_to_df(costs_data, costs_columns)
    network.costs = cols_to_float(network.costs, ["cs", "cw","ch","ch_t","chte","ceth","cNTC","cMH"])
    return network.plot(), network.to_dict()

#%% define scenarios callbacs
# Define scenarios callbacks
@app.callback(
    Output('scenarios-nodes-graph-wind', 'figure'),
    Output('scenarios-locations-graph-wind', 'figure'),
    Output('scenarios-nodes-graph-PV', 'figure'),
    Output('scenarios-locations-graph-PV', 'figure'),
    Output('network-store', 'data'),
    Input('generate-scenarios-button', 'n_clicks'),
    State('my-slider', 'value'),
    State('network-store', 'data'),
    background=True,
    manager=background_callback_manager,
)
def generate_scenarios(n_clicks, value, network_data):
    if n_clicks is None:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), dash.no_update
    network = Network.from_dict(network_data)
    network.n_scenarios = value
    locations = network.n.location.unique().tolist()
    print(network.n)
    if len(locations) == 0:
        raise ValueError("Network must have more than one location")
    pre_gen_locations = ["France", "Italy", "Spain", "Germany", "Austria"]
    samples = 10    
    new_locations = [location for location in locations if location not in pre_gen_locations]
    plots = {}
    if len(new_locations) > 1:
        
        #print("Generating scenarios...")
        scenarios = generate_independent_scenarios(new_locations, 200, save=True, saveas=dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        old_scenarios = import_scenarios("small-EU")
        scenarios["wind"] = pd.concat([scenarios["wind"], old_scenarios["wind"]], axis=0)
        scenarios["PV"] = pd.concat([scenarios["PV"], old_scenarios["PV"]], axis=0)
        scenarios["wind"].reset_index(inplace=True)
        scenarios["PV"].reset_index(inplace=True)
        
    else:
        #print("We already have those :)")
        scenarios = import_scenarios("small-EU")

    plots["wind"] = scenarios["wind"][scenarios["wind"]["scenario"] < samples]
    plots["PV"] = scenarios["PV"][scenarios["PV"]["scenario"] < samples]
    scenarios["wind"] = scenarios["wind"][scenarios["wind"]["scenario"] < value]
    scenarios["PV"] = scenarios["PV"][scenarios["PV"]["scenario"] < value]

    #print(type(scenarios["wind"]))
    network.genW_t = scenario_to_array(scenarios["wind"])
    network.genS_t = scenario_to_array(scenarios["PV"])
    wind_figs = plot_scenarios_df(plots["wind"], var_name = "p.u. Wind power output", title1 = "Wind power output in each node", title2 = "Wind power output for various scenarios" )
    pv_figs = plot_scenarios_df(plots["PV"], var_name = "p.u. PV power output", title1 = "PV power output in each node", title2 = "PV power output for various scenarios")

    #print("Done")

    #TODO model demand as done with production:

    # Import load scenarios
    #01_scenario_generation\scenarios\electricity_load_2023.csv
    scen_path = 'model/scenario_generation/scenarios/'
    elec_load_df = pd.read_csv(scen_path+'electricity_load_2023.csv')
    time_index = range(elec_load_df.shape[0])#pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')
    scenario = 0
    elec_load_scenario = xr.DataArray(
        np.expand_dims(elec_load_df[[country_name_to_iso(location) for location in locations]].values, axis = 2), #add one dimension to correspond with scenarios
        coords={'time': time_index, 'node':locations, 'scenario': [0]},
        dims=['time', 'node', 'scenario']
    )
    #print(network.n)
    hydrogen_demand_scenario = import_generated_scenario(scen_path+'hydrogen_demandg.csv',len(locations), scenario, node_names=locations)
    print("1",network.n)
    if 'node' not in network.n.columns:
        network.n.reset_index(inplace=True)
    print("2",network.n)
    return wind_figs[0], wind_figs[1], pv_figs[0], pv_figs[1], network.to_dict()


#%% optimize page callbacks


@app.callback(
    Output('network-store', 'data', allow_duplicate=True),
    Output('optimization-pie-text', 'children'),
    Output('generators-pie-graph', 'figure'),
    Output('P-edge-graph','figure'),
    Output('hydrogen-storage-graph', 'figure'),
    Output('E-balance-graph', 'figure'),
    Output('H-balance-graph', 'figure'),
    Input('optimize-button', 'n_clicks'),
    State('network-store','data'),
    State('OPT-method-dropdown', 'value'),
    background=True,
    manager=background_callback_manager,
    prevent_initial_call=True)
def optimize(n_clicks, data, method):
    #print("selected method:",method)
    if n_clicks is None:
        return dash.no_update
    #TODO save results
    print("comment out after testing...")
    network = Network.from_dict(data)
    print("3",network.n)
    if 'node' not in network.n.columns:
        if "index" in network.n.columns:
            network.n["node"] = network.n["index"]
        else:
            network.n.reset_index(inplace=True)
    print("4",network.n)
    #network = EU(1)
    if method == 'OPT3':
        #print("OPT3 selected, running...")
        #print("type edge", type(network.edgesP.index.to_list()[0]))
        #print("Optimizing over n scenario: ",  network.n_scenarios)
        results = OPT3(network)
        hydro_fig = plotOPT3_secondstage(results, network, "H",  yaxis_title = "Hydrogen Storage (Kg)", title = "Hydrogen Storage over time")
        P_edge_fig = plotOPT3_secondstage(results, network, "P_edge", yaxis_title="Power flow (MWh)", title = "Power flow through lines", xaxis_title = "Time")
        #node_results = node_results_df(results)
        first_text = "Hover over the graph to see the explicit optimization results"
        generators_pie_fig =  plotOPT_pie(results,network, vars = ["ns","nw","mhte"],  label_names = ["Solar","Wind", "Power Cells"], title_text = "Number of generators by type and percentage of maximum energy production")
        x = network.genS_t.time
        Ebalance_fig = plotE_balance( network, results, x = x)
        Hbalance_fig = plotH_balance(network, results, x= x)

    if method == 'OPT_time_partition':
        #results = OPT_time_partition(network, N_iter = 10, N_refining = 2)
        results = OPT_agg(network)
        last_results = results[-1]
        first_text = "Hover over the graph to see the explicit optimization results"
        generators_pie_fig =  plotOPT_pie(last_results,network, vars = ["ns","nw","mhte"],  label_names = ["Solar","Wind", "Power Cells"], title_text = "Number of generators by type and percentage of maximum energy production")
        P_edge_fig = go.Figure()
        hydro_fig = go.Figure()
        Ebalance_fig = plotE_balance( network, last_results, plot_H = False)
        Hbalance_fig = plotH_balance( network, last_results)
        

        #TODO: add generation chart
    #network.results = results #TODO: convert decently to json

    return network.to_dict(),first_text, generators_pie_fig,P_edge_fig, hydro_fig, Ebalance_fig, Hbalance_fig


if __name__ == '__main__':
    app.run_server(debug=True, port=7051)


# %%
