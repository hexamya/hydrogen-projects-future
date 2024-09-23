import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('project/hydrogen_projects.csv')

df['Date online'] = pd.to_datetime(df['Date online'], errors='coerce', format="%Y")

df['Year'] = df['Date online'].dt.year

annual_capacity = df.groupby('Year')['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]'].sum().reset_index()

X = annual_capacity['Year'].values.reshape(-1, 1)
y = annual_capacity['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

future_years = np.arange(2023, 2031).reshape(-1, 1)
future_predictions = model.predict(future_years)

future_predictions_mt = future_predictions * 24 * 365 * 0.089 / 1e9

plt.figure(figsize=(10, 6))
plt.plot(annual_capacity['Year'], annual_capacity['IEA zero-carbon estimated normalized capacity\n[Nm³ H₂/hour]'], label='Historical')
plt.plot(future_years, future_predictions, label='Predicted')
plt.title('Global Hydrogen Production Capacity Prediction')
plt.xlabel('Year')
plt.ylabel('Capacity [Nm³ H₂/hour]')
plt.legend()
plt.show()

print(f"Predicted global production capacity in 2030: {future_predictions_mt[-1]:.2f} million tons")


df['Renewable'] = df['Technology'].isin(['ALK', 'PEM', 'SOEC', 'Unknown PtX'])
renewable_share = df.groupby('Year')['Renewable'].mean()

X = renewable_share.index.values.reshape(-1, 1)
y = renewable_share.values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(2023, 2031).reshape(-1, 1)
future_share = model.predict(future_years)

plt.figure(figsize=(10, 6))
plt.plot(renewable_share.index, renewable_share.values, label='Historical')
plt.plot(future_years, future_share, label='Predicted')
plt.titl('Share of Renewable-based Projects')
plt.xlabel('Year')
plt.ylabel('Share')
plt.legend()
plt.show()

print(f"Predicted share of renewable-based projects in 2030: {future_share[-1]*100:.2f}%")
