import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('1727098302.380757.312974069.csv')

# Handle missing values
df.dropna(subset=['Date online', 'Capacity_kt H2/y'], inplace=True)

# Convert 'Date online' to datetime
df['Date online'] = pd.to_datetime(df['Date online'], format='%Y')

# Extract year from 'Date online'
df['Year online'] = df['Date online'].dt.year

# Aggregate capacity by year
annual_capacity = df.groupby('Year online')['Capacity_kt H2/y'].sum().reset_index()

# Rename columns to meet Prophet's requirements
annual_capacity.rename(columns={'Year online': 'ds', 'Capacity_kt H2/y': 'y'}, inplace=True)

# Initialize the model
model = Prophet()

# Fit the model
model.fit(annual_capacity)

# Create future dataframe
future = model.make_future_dataframe(periods=10, freq='Y')

# Forecast the future capacity
forecast = model.predict(future)

# Plot the forecast
fig, ax = plt.subplots(figsize=(10, 6))
model.plot(forecast, ax=ax)
plt.xlabel('Year')
plt.ylabel('Global Hydrogen Production Capacity (kt H2/y)')
plt.title('Forecast of Global Hydrogen Production Capacity')
plt.show()
