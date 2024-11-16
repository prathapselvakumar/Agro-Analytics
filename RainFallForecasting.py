import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

from statsmodels.tsa.statespace.sarimax import SARIMAX

def rainfall():
    #rainfall prediction
    sns.set(style='whitegrid', palette='Set2')
    # Load the dataset
    df = pd.read_csv('dataset/rainfall.csv.xls')
    # Drop unnecessary columns
    df = df.drop(['STATE_UT_NAME', 'DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'], axis=1)
    # Choose a specific column for forecasting, for example, 'ANNUAL'
    endog_variable = 'ANNUAL'
    # Create SARIMAX model
    model = SARIMAX(df[endog_variable], order=(5, 1, 3))
    fit = model.fit(disp=False)  # Set disp=False to suppress convergence messages
    # Forecast
    forecast_steps = 50
    forecast = fit.get_forecast(steps=forecast_steps)
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df[endog_variable], label='Historical Rainfall', color='cornflowerblue', linewidth=2)
    plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='darkorange', linewidth=2)
    plt.fill_between(
        forecast.predicted_mean.index,
        forecast.conf_int()['lower ' + endog_variable],
        forecast.conf_int()['upper ' + endog_variable],
        color='lightcoral',
        alpha=0.3,
        label='95% Prediction Interval',
    )
    plt.title('SARIMAX Forecast with Prediction Interval', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Rainfall (mm)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    