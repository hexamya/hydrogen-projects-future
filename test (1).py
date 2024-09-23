import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
import numpy as np

# Load the dataset
file_path = 'project/hydrogen_projects.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Data Preparation
# Convert the 'Date online' column to datetime format
data['Date online'] = pd.to_datetime(data['Date online'], errors='coerce')
# Filter the dataframe for only relevant columns
data_filtered = data[['Date online', 'Capacity_kt H2/y']].dropna()

# Aggregate data to get yearly capacity sums
data_filtered['Year'] = data_filtered['Date online'].dt.year
annual_capacity = data_filtered.groupby('Year')['Capacity_kt H2/y'].sum().reset_index()

# Rename the columns for the forecasting model
annual_capacity.columns = ['Year', 'Total Capacity (kt H2/y)']

# Time Series Forecasting with Prophet
prophet_data = annual_capacity.rename(columns={'Year': 'ds', 'Total Capacity (kt H2/y)': 'y'})

# Fit the Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(prophet_data)

# Create a dataframe for future dates (10 years ahead)
future = model.make_future_dataframe(periods=10, freq='Y')
forecast = model.predict(future)

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(prophet_data['ds'], prophet_data['y'], label='Historical Data', marker='o')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Data', marker='x')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='b', alpha=0.2)
plt.title('Global Hydrogen Production Capacity Forecast')
plt.xlabel('Year')
plt.ylabel('Total Capacity (kt H2/y)')
plt.legend()
plt.grid()
plt.show()
