import pandas as pd

# Load the data
df = pd.read_csv('hydrogen_projects.csv')

# Display the first few rows
print(df.head())

# Handle missing values
# For simplicity, we will drop rows with any missing values in the relevant columns
df.dropna(subset=['Date online', 'Capacity_kt H2/y'], inplace=True)

# Convert 'Date online' to datetime
df['Date online'] = pd.to_datetime(df['Date online'], format='%Y')

# Extract year from 'Date online'
df['Year online'] = df['Date online'].dt.year

# Aggregate capacity by year
annual_capacity = df.groupby('Year online')['Capacity_kt H2/y'].sum().reset_index()

print(annual_capacity)

from fbprophet import Prophet

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
model.plot(forecast)
