from dash import Dash, html, dcc,callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go

from veri_getirici import tickers_list, frequency_list
from tahmin_modeli import train_wrapper, test_wrapper
import numpy as np
import yfinance as yf



# creates the Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


ticker_dropdown = html.Div([
    html.P('Ticker:'),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers_list],
        value='MSFT'
    )
])
frequency_dropdown = html.Div([
    html.P('Frequency:'),
    dcc.Dropdown(
        id='frequency-dropdown',
        options=[{'label': frequency, 'value': frequency} for frequency in frequency_list],
        value='1m'
    )
])

num_bars_input = html.Div([
    html.P('Number of Candles:'),
    dbc.Input(id='num-bar-input', type='number', value='20')
])

tahmin_yapma_butonu = html.Div([
    html.P('Tahmin yapmak için tıklayın'),
    dbc.Button("Tahmin yap", color="primary", id="button", className="mb-3", )

])
# creates the layout of the App
app.layout = html.Div([
    html.H1('Eren Alp Borsa Tahmincisi'),

    # dropdownların olduğu satır
    dbc.Row([
        dbc.Col(ticker_dropdown),
        dbc.Col(frequency_dropdown),
        dbc.Col(num_bars_input),
        dbc.Col(tahmin_yapma_butonu),
    ]),
    html.Button('Model eğit', id='btn-nclicks-1', n_clicks=0),
    html.Button('Tahmin yap', id='btn-nclicks-2', n_clicks=0),
    html.Div(id='container-button-timestamp'),
    # ayıran çizgi
    html.Hr(),

    dcc.Interval(id='update', interval=1000),

    html.Div(id='page-content')

], style={'margin-left': '5%', 'margin-right': '5%', 'margin-top': '20px'})

mylist=[]
testlist=[]

@app.callback(
    Output('page-content', 'children'),
    Input('update', 'n_intervals'),
    State('ticker-dropdown', 'value'), State('frequency-dropdown', 'value'), State('num-bar-input', 'value'), State('button', 'value')
)
def grafik_olustur(self, ticker, frequency, num_bars, button):
    # Interval required 1 minute, inetrval=frequency(diğer dosyada karışmasın)
    data = yf.download(tickers=ticker, period='1d', interval=frequency)

    # istenen sayıda mum koymak için kesme işlemiw
    # num_bars = int(num_bars)
    df_kesik = data.iloc[:int(num_bars), :]

    # declare figure
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=df_kesik.index,
                                 open=df_kesik['Open'],
                                 high=df_kesik['High'],
                                 low=df_kesik['Low'],
                                 close=df_kesik['Close'], name='market data'))

    if len(testlist) > 0:
        # x = data.index, y = data['Close']
        fig.add_scatter(x = data.index, y=testlist[1]['predictions'], mode='lines', line_color='#ffe476')
        fig.add_trace(go.Scatter(x = data.index, y=testlist[1]['predictions'], mode='lines', line_color='#ffe476'))


    # Add titles
    fig.update_layout(
        title=ticker+' canlı fiyat grafiği',
        yaxis_title='Hisse Fiyatı (USD)')


    return [
        html.H2(id='chart-details', children=f'{ticker} - {frequency}'),
        dcc.Graph(figure=fig, config={'displayModeBar': False})
        ]



@app.callback(
    Output('container-button-timestamp', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('btn-nclicks-2', 'n_clicks'),
State('ticker-dropdown', 'value')
)
def displayClick(btn1, btn2, ticker):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg = 'Training başladı'
        model, scaled_data, training_data_size, dataset, scaler, data = train_wrapper(ticker)
        mylist.append(model)
        mylist.append(scaled_data)
        mylist.append(training_data_size)
        mylist.append(dataset)
        mylist.append(scaler)
        mylist.append(data)
        msg = 'Training bitti'
    elif 'btn-nclicks-2' in changed_id:
        train, valid, predictions = test_wrapper(mylist[0], mylist[1], mylist[2], mylist[3], mylist[4], mylist[5])
        testlist.append(train)
        testlist.append(valid)
        testlist.append(predictions)
        msg = 'Test hazır'
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div(msg)



if __name__ == '__main__':
    # starts the server
    app.run_server(debug=True)

