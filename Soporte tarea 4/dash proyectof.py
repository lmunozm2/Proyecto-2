import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv # pip install python-dotenv
import os
import psycopg2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
env_path="C://Users//angie//OneDrive//Desktop//Octavo semestre//Analitica computacional//Proyecto 2 personal//env//app.env"
#load env 
load_dotenv(dotenv_path=env_path)
#extract env variables
USER=os.getenv('USER')
PASSWORD=os.getenv('PASSWORD')
HOST=os.getenv('HOST')
PORT=os.getenv('PORT')
DBNAME=os.getenv('DBNAME')
engine = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT
)


# cargar archivo de disco
model = tf.keras.models.load_model("models/modelo.keras")

#Hasta aqui todo esta bien
#GRAFICOS
import pandas.io.sql as sqlio
cursor = engine.cursor()
query = """
SELECT * 
FROM proy1;"""
df = sqlio.read_sql_query(query, engine)
fig_edades = px.histogram(df, x="x5", title="Distribución de Edades")
fig_edades.update_xaxes(title_text="Edades")

gender_labels = {1: 'Masculino', 2: 'Femenino'}
education_labels = {1: 'Posgrado', 2: 'Universidad', 3: 'Bachillerato', 4: 'Otro'}
marital_status_labels = {1: 'Casado', 2: 'Soltero', 3: 'Otro'}
categorical_columns = ['x2', 'x3', 'x4']

# Obtener las frecuencias de las variables categóricas
gender_freq = df['x2'].map(gender_labels).value_counts()
education_freq = df['x3'].map(education_labels).value_counts()
marital_status_freq = df['x4'].map(marital_status_labels).value_counts()

# Crear gráficos de barras
gender_fig = go.Figure(data=[go.Bar(x=gender_freq.index, y=gender_freq.values, marker=dict(color='lightgreen'))])
gender_fig.update_layout(title=' Género')

education_fig = go.Figure(data=[go.Bar(x=education_freq.index, y=education_freq.values, marker=dict(color='salmon'))])
education_fig.update_layout(title='Nivel de Educación')

marital_status_fig = go.Figure(data=[go.Bar(x=marital_status_freq.index, y=marital_status_freq.values, marker=dict(color='skyblue'))])
marital_status_fig.update_layout(title='Estado Civil')

# Layout de la aplicación Dash
app.layout = html.Div([
    html.H1("Predicción de modelo", style={'font-family': 'Calibri Light'}),
    html.Div([
        
        # GÉNERO (X2)
        html.Div([
            html.Label('Género'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Masculino', 'value': 1},
                    {'label': 'Femenino', 'value': 2}
                ],
                value=1
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),
        # EDUCACIÓN (X3)
        html.Div([
            html.Label('Nivel de Educación'),
            dcc.Dropdown(
                id='education-dropdown',
                options=[
                    {'label': 'Graduado', 'value': 1},
                    {'label': 'Universidad', 'value': 2},
                    {'label': 'Bachillerato', 'value': 3},
                    {'label': 'Otros', 'value': 4}
                ],
                value=1
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        # ESTADO CIVIL (X4)
        html.Div([    
            html.Label('Estado Civil'),
            dcc.Dropdown(
                id='marital-status-dropdown',
                options=[
                    {'label': 'Casado', 'value': 1},
                    {'label': 'Soltero', 'value': 2},
                    {'label': 'Otros', 'value': 3}
                ],
                value=1
            ),
        ],style={'width': '25%', 'display': 'inline-block'}),
        #EDAD (X5)
        html.Div([
            html.Label('Edad'),
            dcc.Input(id='age-input', type='number', min=0, max=80, step=1, value=0),
        ],style = {'width': '25%', 'display': 'inline-block'}),

        #SEPTIEMBRE (X6)
        html.Label('Historial de Pagos en Septiembre (X6)'),
        dcc.Dropdown(
            id='payment-sept-dropdown',
            options=[
                {'label': 'Pago anticipado', 'value': -2},
                {'label': 'Pago puntual', 'value': -1},
                {'label': 'Retraso de un mes', 'value': 1},
                {'label': 'Retraso de dos meses', 'value': 2},
                {'label': 'Retraso de tres meses', 'value': 3},
                {'label': 'Retraso de cuatro meses', 'value': 4},
                {'label': 'Retraso de cinco meses', 'value': 5},
                {'label': 'Retraso de seis meses', 'value': 6}
            ],
            value=-2
        ),
        #AGOSTO (X7)
        html.Label('Historial de Pagos en Agosto (X7)'),
        dcc.Dropdown(
            id='payment-aug-dropdown',
            options=[
                {'label': 'Pago anticipado', 'value': -2},
                {'label': 'Pago puntual', 'value': -1},
                {'label': 'Retraso de un mes', 'value': 1},
                {'label': 'Retraso de dos meses', 'value': 2},
                {'label': 'Retraso de tres meses', 'value': 3},
                {'label': 'Retraso de cuatro meses', 'value': 4},
                {'label': 'Retraso de cinco meses', 'value': 5},
                {'label': 'Retraso de seis meses', 'value': 6}
            ],
            value=-2
        ),
        #JULIO (X8)
        html.Label('Historial de Pagos en Julio (X8)'),
        dcc.Dropdown(
            id='payment-jul-dropdown',
            options=[
                {'label': 'Pago anticipado', 'value': -2},
                {'label': 'Pago puntual', 'value': -1},
                {'label': 'Retraso de un mes', 'value': 1},
                {'label': 'Retraso de dos meses', 'value': 2},
                {'label': 'Retraso de tres meses', 'value': 3},
                {'label': 'Retraso de cuatro meses', 'value': 4},
                {'label': 'Retraso de cinco meses', 'value': 5},
                {'label': 'Retraso de seis meses', 'value': 6}
            ],
            value=-2
        ),
        #JUNIO (X9)
        html.Label('Historial de Pagos en Junio (X9)'),
        dcc.Dropdown(
            id='payment-jun-dropdown',
            options=[
                {'label': 'Pago anticipado', 'value': -2},
                {'label': 'Pago puntual', 'value': -1},
                {'label': 'Retraso de un mes', 'value': 1},
                {'label': 'Retraso de dos meses', 'value': 2},
                {'label': 'Retraso de tres meses', 'value': 3},
                {'label': 'Retraso de cuatro meses', 'value': 4},
                {'label': 'Retraso de cinco meses', 'value': 5},
                {'label': 'Retraso de seis meses', 'value': 6}
            ],
            value=-2
        ),
        #MAYO (X10)
        html.Label('Historial de Pagos en Mayo (X10)'),
        dcc.Dropdown(
            id='payment-may-dropdown',
            options=[
                {'label': 'Pago anticipado', 'value': -2},
                {'label': 'Pago puntual', 'value': -1},
                {'label': 'Retraso de un mes', 'value': 1},
                {'label': 'Retraso de dos meses', 'value': 2},
                {'label': 'Retraso de tres meses', 'value': 3},
                {'label': 'Retraso de cuatro meses', 'value': 4},
                {'label': 'Retraso de cinco meses', 'value': 5},
                {'label': 'Retraso de seis meses', 'value': 6}
            ],
            value=-2
        ),
        #ABRIL (X11)
        html.Label('Historial de Pagos en Abril (X11)'),
        dcc.Dropdown(
            id='payment-abril-dropdown',
            options=[
                {'label': 'Pago anticipado', 'value': -2},
                {'label': 'Pago puntual', 'value': -1},
                {'label': 'Retraso de un mes', 'value': 1},
                {'label': 'Retraso de dos meses', 'value': 2},
                {'label': 'Retraso de tres meses', 'value': 3},
                {'label': 'Retraso de cuatro meses', 'value': 4},
                {'label': 'Retraso de cinco meses', 'value': 5},
                {'label': 'Retraso de seis meses', 'value': 6}
            ],
            value=-2
        ),

        # VARIABLES CONTINUAS (X1, X5 y X18 a X23)
        
        html.Div([
            html.Label('Crédito Otorgado'),
            dcc.Input(id='amount-credit-input', type='number', min=0, max=1000000, step=1, value=0),
        ], style = {'width': '20%', 'display': 'inline-block', 'margin': 'auto'}),
        
        
        html.Div([
            html.Div([
                html.Label('Monto pagado en Septiembre'),
                dcc.Input(id='amount-paid-sept-input', type='number', min=0, max=1000000, step=1, value=0),
            ], style={'width': '16.5%', 'display': 'inline-block', 'margin': 'auto'}),
            html.Div([
                html.Label('Monto pagado en Agosto'),
                dcc.Input(id='amount-paid-aug-input', type='number', min=0, max=1000000, step=1, value=0),
            ], style={'width': '16.5%', 'display': 'inline-block', 'margin': 'auto'}),
            html.Div([
                html.Label('Monto pagado en Julio'),
                dcc.Input(id='amount-paid-jul-input', type='number', min=0, max=1000000, step=1, value=0),
            ], style={'width': '16.5%', 'display': 'inline-block', 'margin': 'auto'}),
            html.Div([
                html.Label('Monto pagado en Junio'),
                dcc.Input(id='amount-paid-jun-input', type='number', min=0, max=1000000, step=1, value=0),
            ], style={'width': '16.5%', 'display': 'inline-block', 'margin': 'auto'}),
            html.Div([
                html.Label('Monto pagado en Mayo'),
                dcc.Input(id='amount-paid-may-input', type='number', min=0, max=1000000, step=1, value=0),
            ], style={'width': '16.5%', 'display': 'inline-block', 'margin': 'auto'}),
            html.Div([
                html.Label('Monto pagado en Abril'),
                dcc.Input(id='amount-paid-apr-input', type='number', min=0, max=1000000, step=1, value=0),
            ], style={'width': '16.5%', 'display': 'inline-block', 'margin': 'auto'}),
        ]),
        html.Button('Predecir', id='button', style={'display': 'block', 'margin': 'auto'}),
        html.Div(id='output-prediction', style = {'fontsize':'24px','textAlign': 'center', 'font-weight': 'bold'})
    ]),

    #Graficos
    
    html.H2("Gráficos de interés", style={'font-family': 'Calibri Light'}),
    html.Div([
        html.Div(dcc.Graph(figure=fig_edades), style={'display': 'inline-block', 'width': '25%'}),
        html.Div(dcc.Graph(figure=gender_fig), style={'display': 'inline-block', 'width': '25%'}),
        html.Div(dcc.Graph(figure=education_fig), style={'display': 'inline-block', 'width': '25%'}),
        html.Div(dcc.Graph(figure=marital_status_fig), style={'display': 'inline-block', 'width': '25%'})
    ])

], style={'font-family': 'Calibri Light'})


# Callback para realizar la predicción
@app.callback(
    Output('output-prediction', 'children'),
    [Input('button', 'n_clicks')],
    [dash.dependencies.State('gender-dropdown', 'value'),
     dash.dependencies.State('education-dropdown', 'value'),
     dash.dependencies.State('marital-status-dropdown', 'value'),
     dash.dependencies.State('payment-sept-dropdown', 'value'),
     dash.dependencies.State('payment-aug-dropdown', 'value'),
     dash.dependencies.State('payment-jul-dropdown', 'value'),
     dash.dependencies.State('payment-jun-dropdown', 'value'),
     dash.dependencies.State('payment-may-dropdown', 'value'),
     dash.dependencies.State('payment-abril-dropdown', 'value'),
     dash.dependencies.State('amount-credit-input', 'value'),
     dash.dependencies.State('age-input', 'value'),
     dash.dependencies.State('amount-paid-sept-input', 'value'),
     dash.dependencies.State('amount-paid-aug-input', 'value'),
     dash.dependencies.State('amount-paid-jul-input', 'value'),
     dash.dependencies.State('amount-paid-jun-input', 'value'),
     dash.dependencies.State('amount-paid-may-input', 'value'),
     dash.dependencies.State('amount-paid-apr-input', 'value')]
    
)

def predict(n_clicks, gender, education, marital_status, 
            payment_sept, payment_aug,payment_jul, payment_jun,payment_may, payment_abr, 
            amount_credit, age, 
            amount_paid_sept, amount_paid_aug, amount_paid_jul, amount_paid_jun, amount_paid_may, amount_paid_apr):
    if n_clicks is None:
        return ''
    
    # Combinar las variables categóricas y continuas en un solo vector de entrada
    input_values = [gender, education, marital_status,
                    payment_sept, payment_aug, payment_jul, payment_jun,payment_may, payment_abr, 
                    amount_credit, age, 
                    amount_paid_sept, amount_paid_aug, amount_paid_jul, amount_paid_jun, amount_paid_may, amount_paid_apr]
    
    # Preprocesar los valores ingresados (por ejemplo, escalarlos)
    input_values = np.array([input_values])  # Formato de entrada esperado por el modelo
    
    
    print("Datos de entrada:", input_values)
    print("Forma de los datos de entrada:", input_values.shape)

    # Realizar la predicción
    prediction = model.predict(input_values)
    predicted_probability = prediction[0, 1]  
    predicted_probability2 = prediction[0, 0]  
    print(predicted_probability)
    print(predicted_probability2)
    
    # Aquí puedes procesar la salida de la predicción según sea necesario
    # Por ejemplo, mostrar el resultado
    if prediction[0][0] < prediction[0][1]:
        result = "Riesgo de incumplimiento de pago"
    else:
        result = "Sin riesgo de incumplimiento de pago"
    
    return f'Predicción: {result}'

if __name__ == '__main__':
    app.run_server(debug=True)
