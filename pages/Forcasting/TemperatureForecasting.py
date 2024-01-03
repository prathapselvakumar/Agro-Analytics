import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

def temperature():
    df = pd.read_csv('dataset/TemperatureMum_1999-2019.csv')
    df_date = []
    for i in range(0, len(df['Month'])):
        date = datetime.date(year=df['Year'][i], month=df['Month'][i], day=df['Day'][i])
        df_date.append(date)
    df['Date'] = pd.to_datetime(df_date)
    df = df.drop(['Month', 'Day', 'Year', 'TempF'], axis=1)
    df.set_index("Date", inplace=True)
    rolmean = df.rolling(window=52).mean()
    rolstd = df.rolling(window=52).std()
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    sarima_model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    sarima_result = sarima_model.fit()
    forecast_steps = len(test)
    forecast = sarima_result.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(test, forecast_mean)
    print(f"Mean Absolute Error: {mae}")
# MODIFIED colour pallete for the graph - CG blue and yellow
    # Set seaborn style with a custom background color
    sns.set_style("whitegrid", {"axes.facecolor": "#E0F7FA"})  # Custom background color (light blue)
    # Set the color palette with autumn vibes (dark blue, orange, white)
    custom_palette = ["#1565C0", "#FFA000", "#000000"]  # Custom colors (dark blue, orange, black)
    sns.set_palette(custom_palette)
    # Plot SARIMA Forecast with seaborn
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=train.index, y=train['TempC'], label='Training Data', color=custom_palette[0])
    sns.lineplot(x=test.index, y=test['TempC'], label='Test Data', color=custom_palette[1])
    sns.lineplot(x=forecast_mean.index, y=forecast_mean, label='SARIMA Forecast', color=custom_palette[2])
    plt.fill_between(
        forecast_mean.index,
        forecast.conf_int()['lower TempC'],
        forecast.conf_int()['upper TempC'],
        color=custom_palette[1],  # Use the orange color for the prediction interval
        alpha=0.2,
        label='95% Prediction Interval',
    )
    plt.xlabel('Date')
    plt.ylabel('Temperature (Celsius)')
    plt.title('SARIMA Weather Forecast')
    plt.legend()
    plt.tight_layout()
    # Display the plot
    # Additional Information

    return mae