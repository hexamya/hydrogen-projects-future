import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


# Load the dataset
file_path = 'hydrogen_projects.csv'
data = pd.read_csv(file_path)


# Display the first few rows of the dataset
print(data.head())


# Data Preparation
# Filter relevant columns for end-use analysis
end_use_columns = [
    'EndUse_Refining'
    'EndUse_Ammonia',
    'EndUse_Methanol',
    'EndUse_Iron&Steel',
    'EndUse_Other Ind',
    'EndUse_Mobility',
    'EndUse_Power',
    'EndUse_Grid inj.',
    'EndUse_CHP',
    'EndUse_Domestic heat',
    'EndUse_Biofuels',
    'EndUse_Synfuels',
    'EndUse_CH4 grid inj.',
    'EndUse_CH4 mobility'
]

# Convert end-use columns to numerical values
data[end_use_columns] = data[end_use_columns].applymap(lambda x: 0 if pd.isnull(x) else 1)
data['Date online'] = pd.to_datetime(data['Date online'], errors='coerce', format="%Y")

# Sum the hydrogen projects for each end-use sector
end_use_sum = data[end_use_columns].sum().reset_index()
end_use_sum.columns = ['Sector', 'Number of Projects']

# Determine current demand representation as a percentage
total_projects = end_use_sum['Number of Projects'].sum()
end_use_sum['Percentage'] = (end_use_sum['Number of Projects'] / total_projects) * 100

# Visualization of current hydrogen demand by sector
plt.figure(figsize=(12, 6))
plt.bar(end_use_sum['Sector'], end_use_sum['Percentage'], color='skyblue')
plt.title('Current Hydrogen Demand by End-Use Sector')
plt.xlabel('End-Use Sector')
plt.ylabel('Percentage of Total Projects (%)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Predicting future demand growth in sectors
# Create a mock-up for future years, this could come from expert estimations, trends, etc.
future_years = data.groupby("Date online").count()
future_years["Year"] = future_years.index.year
future_years['Number of Projects'] = future_years['Ref']


# Prepare data for regression
X = future_years[["Year"]]
y = future_years['Ref']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X["2000-01-01": "2023-01-01"], y["2000-01-01": "2023-01-01"], test_size=0.2, random_state=42)

# Fit the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict future years
future_years['Predicted Projects'] = model.predict(future_years[['Year']])

# Visualizing the predictions
plt.figure(figsize=(12, 6))
plt.plot(future_years['Year'], future_years['Number of Projects'], label='Current Projects', marker='o')
plt.plot(future_years['Year'], future_years['Predicted Projects'], label='Predicted Projects', marker='x', linestyle='--')
plt.title('Forecast of Hydrogen Projects Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Projects')
plt.legend()
plt.grid()
plt.show()
