import math
from flask import Flask, render_template
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import cycle
import plotly.express as px
import json

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from numpy import array

app = Flask(__name__)

# Load dataset
maindf = pd.read_csv('dataset5.csv')
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')


closedf = maindf[['Date','Close']]
closedf = closedf[closedf['Date'] > '2019-02-28']
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

training_size=int(len(closedf)*0.70)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 

model=Sequential()
model.add(LSTM(10, input_shape=(None,1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)

x_input = test_data[len(test_data)-time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output=[]
n_steps=time_step
i=0
pred_days = 30
while(i<pred_days):
    if(len(temp_input)>time_step):
        x_input=np.array(temp_input[1:])
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]

        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())

        lst_output.extend(yhat.tolist())
        i=i+1

last_days = np.arange(1, time_step + 1)
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)

temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1, -1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step + 1] = scaler.inverse_transform(closedf[len(closedf) - time_step:]).reshape(1, -1).tolist()[0]
next_predicted_days_value[time_step + 1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value': last_original_days_value,
    'next_predicted_days_value': next_predicted_days_value
})
lstmdf = closedf.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1, 1)).tolist())
lstmdf = scaler.inverse_transform(lstmdf).reshape(1, -1).tolist()[0]


# fig = go.Figure()

# # fig.add_trace(go.Scatter(x=new_pred_plot.index, y=new_pred_plot['last_original_days_value'], mode='lines', name='Last 15 days close price'))
# # fig.add_trace(go.Scatter(x=new_pred_plot.index, y=new_pred_plot['next_predicted_days_value'], mode='lines', name='Predicted next 30 days close price'))
# fig.add_trace(go.Scatter(x=new_pred_plot.index, y=new_pred_plot['last_original_days_value'], mode='lines'))
# fig.add_trace(go.Scatter(x=new_pred_plot.index, y=new_pred_plot['next_predicted_days_value'], mode='lines'))

# fig.update_layout(title='Compare last 15 days vs next 30 days', xaxis_title='Timestamp', yaxis_title='Stock price', plot_bgcolor='white', font_size=15, font_color='black')

# plot_div = fig.to_html(full_html=False)

# Slice the DataFrame to include only the past 15 days data for the first graph
past_15_days_data = new_pred_plot.iloc[:time_step + 1]

# Slice the DataFrame to include only the next 30 days data for the second graph
next_30_days_data = new_pred_plot.iloc[time_step + 1:]

# Combine data for both past 15 days and next 30 days into one DataFrame
# Create a single plot for both past 15 days and next 30 days
fig_combined = go.Figure()

# Add trace for past 15 days with blue color
fig_combined.add_trace(go.Scatter(x=past_15_days_data.index, y=past_15_days_data['last_original_days_value'], mode='lines', name='Last 15 days close price', line=dict(color='blue')))

# Add trace for next 30 days with red color
fig_combined.add_trace(go.Scatter(x=next_30_days_data.index, y=next_30_days_data['next_predicted_days_value'], mode='lines', name='Predicted next 30 days close price', line=dict(color='red')))

fig_combined.update_layout(title='Combined Comparison of Last 15 Days vs Next 30 Days', xaxis_title='Timestamp', yaxis_title='Stock price', plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
plot_div_combined = fig_combined.to_html(full_html=False)


@app.route('/')
def index():
    
    return render_template('index.html',plot_div_combined=plot_div_combined,lstmdf=lstmdf)

@app.route('/stock_analysis')
def stock_analysis():
    # Filter data for the year 2019
    y_2019 = maindf.loc[(maindf['Date'] >= '2019-01-01') & (maindf['Date'] <= '2019-12-31')]

    # Calculate monthly averages for open and close prices
    monthvise = y_2019.groupby(y_2019['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    # Calculate monthly high and low prices
    monthvise_high = y_2019.groupby(y_2019['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
    monthvise_low = y_2019.groupby(y_2019['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

    # Create Plotly bar chart for open and close prices
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45,  title='Monthwise comparison between Stock open and close price')

    # Create Plotly bar chart for high and low prices
    fig_high_low = go.Figure()
    fig_high_low.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig_high_low.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    fig_high_low.update_layout(barmode='group', title='Monthwise High and Low stock price')

    # Create Plotly line chart for open, close, high, and low prices
    names = cycle(['Stock Open Price', 'Stock Close Price', 'Stock High Price', 'Stock Low Price'])
    fig_line = px.line(y_2019, x=y_2019['Date'], y=[y_2019['Open'], y_2019['Close'], y_2019['High'], y_2019['Low']],
                       labels={'Date': 'Date', 'value': 'Stock value'})
    fig_line.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_line.for_each_trace(lambda t: t.update(name=next(names)))
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False)

    # Convert Plotly figures to JSON-compatible dictionaries
    fig_bar_json = json.loads(fig_bar.to_json())
    fig_high_low_json = json.loads(fig_high_low.to_json())
    fig_line_json = json.loads(fig_line.to_json())

    return render_template('stock_analysis.html', fig_bar=fig_bar_json, fig_high_low=fig_high_low_json, fig_line=fig_line_json)


@app.route('/stock_analysis_2020')
def stock_analysis_2020():
    # Filter data for the year 2020
    y_2020 = maindf.loc[(maindf['Date'] >= '2020-01-01') & (maindf['Date'] < '2021-01-01')]

    # Calculate monthly averages for open and close prices
    monthvise = y_2020.groupby(y_2020['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    # Calculate monthly high and low prices
    monthvise_high = y_2020.groupby(y_2020['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
    monthvise_low = y_2020.groupby(y_2020['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

    # Create Plotly bar chart for open and close prices
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45,  title='Monthwise comparison between Stock open and close price')

    # Create Plotly bar chart for high and low prices
    fig_high_low = go.Figure()
    fig_high_low.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig_high_low.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    fig_high_low.update_layout(barmode='group', title='Monthwise High and Low stock price')

    # Create Plotly line chart for open, close, high, and low prices
    names = cycle(['Stock Open Price', 'Stock Close Price', 'Stock High Price', 'Stock Low Price'])
    fig_line = px.line(y_2020, x=y_2020['Date'], y=[y_2020['Open'], y_2020['Close'], y_2020['High'], y_2020['Low']],
                       labels={'Date': 'Date', 'value': 'Stock value'})
    fig_line.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_line.for_each_trace(lambda t: t.update(name=next(names)))
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False)

    # Convert Plotly figures to JSON-compatible dictionaries
    fig_bar_json = fig_bar.to_json()
    fig_high_low_json = fig_high_low.to_json()
    fig_line_json = fig_line.to_json()

    return render_template('stock_analysis_2020.html', fig_bar=fig_bar_json, fig_high_low=fig_high_low_json, fig_line=fig_line_json)


@app.route('/stock_analysis_2021')
def stock_analysis_2021():
    # Filter data for the year 2021
    y_2021 = maindf.loc[(maindf['Date'] >= '2021-01-01') & (maindf['Date'] < '2022-01-01')]

    # Calculate monthly averages for open and close prices
    monthvise = y_2021.groupby(y_2021['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    # Calculate monthly high and low prices
    monthvise_high = y_2021.groupby(y_2021['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
    monthvise_low = y_2021.groupby(y_2021['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

    # Create Plotly bar chart for open and close prices
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45,  title='Monthwise comparison between Stock open and close price')

    # Create Plotly bar chart for high and low prices
    fig_high_low = go.Figure()
    fig_high_low.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig_high_low.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    fig_high_low.update_layout(barmode='group', title='Monthwise High and Low stock price')

    # Create Plotly line chart for open, close, high, and low prices
    names = cycle(['Stock Open Price', 'Stock Close Price', 'Stock High Price', 'Stock Low Price'])
    fig_line = px.line(y_2021, x=y_2021['Date'], y=[y_2021['Open'], y_2021['Close'], y_2021['High'], y_2021['Low']],
                       labels={'Date': 'Date', 'value': 'Stock value'})
    fig_line.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_line.for_each_trace(lambda t: t.update(name=next(names)))
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False)

    # Convert Plotly figures to JSON-compatible dictionaries
    fig_bar_json = fig_bar.to_json()
    fig_high_low_json = fig_high_low.to_json()
    fig_line_json = fig_line.to_json()

    return render_template('stock_analysis_2021.html', fig_bar=fig_bar_json, fig_high_low=fig_high_low_json, fig_line=fig_line_json)



@app.route('/stock_analysis_2022')
def stock_analysis_2022():
    # Filter data for the year 2022
    y_2022 = maindf.loc[(maindf['Date'] >= '2022-01-01') & (maindf['Date'] < '2023-01-01')]

    # Calculate monthly averages for open and close prices
    monthvise = y_2022.groupby(y_2022['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    # Calculate monthly high and low prices
    monthvise_high = y_2022.groupby(y_2022['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
    monthvise_low = y_2022.groupby(y_2022['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

    # Create Plotly bar chart for open and close prices
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45,  title='Monthwise comparison between Stock open and close price')

    # Create Plotly bar chart for high and low prices
    fig_high_low = go.Figure()
    fig_high_low.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig_high_low.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    fig_high_low.update_layout(barmode='group', title='Monthwise High and Low stock price')

    # Create Plotly line chart for open, close, high, and low prices
    names = cycle(['Stock Open Price', 'Stock Close Price', 'Stock High Price', 'Stock Low Price'])
    fig_line = px.line(y_2022, x=y_2022['Date'], y=[y_2022['Open'], y_2022['Close'], y_2022['High'], y_2022['Low']],
                       labels={'Date': 'Date', 'value': 'Stock value'})
    fig_line.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_line.for_each_trace(lambda t: t.update(name=next(names)))
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False)

    # Convert Plotly figures to JSON-compatible dictionaries
    fig_bar_json = fig_bar.to_json()
    fig_high_low_json = fig_high_low.to_json()
    fig_line_json = fig_line.to_json()

    return render_template('stock_analysis_2022.html', fig_bar=fig_bar_json, fig_high_low=fig_high_low_json, fig_line=fig_line_json)

@app.route('/stock_analysis_2023')
def stock_analysis_2023():
    # Filter data for the year 2023
    y_2023 = maindf.loc[(maindf['Date'] >= '2023-01-01') & (maindf['Date'] < '2024-01-01')]

    # Calculate monthly averages for open and close prices
    monthvise = y_2023.groupby(y_2023['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    # Calculate monthly high and low prices
    monthvise_high = y_2023.groupby(y_2023['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
    monthvise_low = y_2023.groupby(y_2023['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

    # Create Plotly bar chart for open and close prices
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45,  title='Monthwise comparison between Stock open and close price')

    # Create Plotly bar chart for high and low prices
    fig_high_low = go.Figure()
    fig_high_low.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig_high_low.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    fig_high_low.update_layout(barmode='group', title='Monthwise High and Low stock price')

    # Create Plotly line chart for open, close, high, and low prices
    names = cycle(['Stock Open Price', 'Stock Close Price', 'Stock High Price', 'Stock Low Price'])
    fig_line = px.line(y_2023, x=y_2023['Date'], y=[y_2023['Open'], y_2023['Close'], y_2023['High'], y_2023['Low']],
                       labels={'Date': 'Date', 'value': 'Stock value'})
    fig_line.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_line.for_each_trace(lambda t: t.update(name=next(names)))
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False)

    # Convert Plotly figures to JSON-compatible dictionaries
    fig_bar_json = fig_bar.to_json()
    fig_high_low_json = fig_high_low.to_json()
    fig_line_json = fig_line.to_json()

    return render_template('stock_analysis_2023.html', fig_bar=fig_bar_json, fig_high_low=fig_high_low_json, fig_line=fig_line_json)


@app.route('/stock_analysis_2024')
def stock_analysis_2024():
    # Filter data for the year 2024
    y_2024 = maindf.loc[(maindf['Date'] >= '2024-01-01') & (maindf['Date'] < '2024-02-28')]

    # Calculate monthly averages for open and close prices
    monthvise = y_2024.groupby(y_2024['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']
    monthvise = monthvise.reindex(new_order, axis=0)

    # Calculate monthly high and low prices
    monthvise_high = y_2024.groupby(y_2024['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
    monthvise_low = y_2024.groupby(y_2024['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

    # Create Plotly bar chart for open and close prices
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Open'],
        name='Stock Open Price',
        marker_color='crimson'
    ))
    fig_bar.add_trace(go.Bar(
        x=monthvise.index,
        y=monthvise['Close'],
        name='Stock Close Price',
        marker_color='lightsalmon'
    ))
    fig_bar.update_layout(barmode='group', xaxis_tickangle=-45,  title='Monthwise comparision between Stock open and close price')

    # Create Plotly bar chart for high and low prices
    fig_high_low = go.Figure()
    fig_high_low.add_trace(go.Bar(
        x=monthvise_high.index,
        y=monthvise_high,
        name='Stock high Price',
        marker_color='rgb(0, 153, 204)'
    ))
    fig_high_low.add_trace(go.Bar(
        x=monthvise_low.index,
        y=monthvise_low,
        name='Stock low Price',
        marker_color='rgb(255, 128, 0)'
    ))
    fig_high_low.update_layout(barmode='group', title='Monthwise High and Low stock price')

    # Create Plotly line chart for open, close, high, and low prices
    names = cycle(['Stock Open Price', 'Stock Close Price', 'Stock High Price', 'Stock Low Price'])
    fig_line = px.line(y_2024, x=y_2024['Date'], y=[y_2024['Open'], y_2024['Close'], y_2024['High'], y_2024['Low']],
                       labels={'Date': 'Date', 'value': 'Stock value'})
    fig_line.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_line.for_each_trace(lambda t: t.update(name=next(names)))
    fig_line.update_xaxes(showgrid=False)
    fig_line.update_yaxes(showgrid=False)

    # Filter data for the overall period
    y_overall = maindf.loc[(maindf['Date'] >= '2019-02-28') & (maindf['Date'] <= '2024-02-28')]

    # Calculate monthly averages for open and close prices for the overall period
    monthvise_overall = y_overall.groupby(y_overall['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
    monthvise_overall = monthvise_overall.reindex(new_order, axis=0)

    # Create Plotly line chart for open, close, high, and low prices for the overall period
    fig_overall = px.line(y_overall, x=y_overall['Date'], y=[y_overall['Open'], y_overall['Close'], y_overall['High'], y_overall['Low']],
                          labels={'Date': 'Date', 'value': 'Stock value'})
    fig_overall.update_layout(title_text='Stock analysis chart for the overall period', font_size=15, font_color='black', legend_title_text='Stock Parameters')
    fig_overall.for_each_trace(lambda t: t.update(name=next(names)))
    fig_overall.update_xaxes(showgrid=False)
    fig_overall.update_yaxes(showgrid=False)

    return render_template('stock_analysis_2024.html', fig_bar=fig_bar.to_json(), fig_high_low=fig_high_low.to_json(), fig_line=fig_line.to_json(), fig_overall=fig_overall.to_json())




if __name__ == "__main__":
    app.run(debug=True)
