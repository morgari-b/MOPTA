
import dash
from dash import dash_table, dcc, html

app = dash.Dash(__name__)

# Define the table data
table_data = [
    ['Text', 'Fillable Text', 'Dropdown', 'Latitude', 'Longitude', 'Number'],
    ['Example', '', 'Option 1', 37.7749, -122.4194, 10],
    ['Example', '', 'Option 2', 33.7749, -84.4194, 20],
]

# Define the table columns
table_columns = [
    {'name': 'Column 1', 'id': 'column1'},
    {'name': 'Column 2', 'id': 'column2', 'presentation': 'dropdown'},
    {'name': 'Column 3', 'id': 'column3'},
    {'name': 'Column 4', 'id': 'column4'},
    {'name': 'Column 5', 'id': 'column5'},
    {'name': 'Column 6', 'id': 'column6'},
]

# Create the table
table = dash_table.DataTable(
    id='fillable-table',
    data=table_data,
    columns=table_columns,
    editable=True,
    fill_width=True,
)

# Create the app layout
app.layout = html.Div([
    table,
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)



# %%
