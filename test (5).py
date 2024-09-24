import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load hydrogen production dataset
file_path_hydrogen = 'hydrogen_projects.csv'
hydrogen_data = pd.read_csv(file_path_hydrogen)

# Display the first few rows of the hydrogen dataset
print(hydrogen_data.head())

# Assume we have another dataset for renewable energy capacity
# This should include country or region names and their corresponding renewable energy capacities
# Example format of renewable energy data
# renewable_data = pd.DataFrame({
#     'Country': ['Country A', 'Country B', 'Country C'],
#     'RenewableCapacity_MW': [5000, 2000, 1500]
# })

# Load renewable energy capacity data
# Replace this with the actual data source
renewable_data = pd.read_csv('path_to_your_renewable_energy_data.csv')

# Display the first few rows of the renewable energy dataset
print(renewable_data.head())

# Data Preparation
# Convert hydrogen production capacity to a sum by country
hydrogen_data['Country'] = hydrogen_data['Country'].str.strip()  # Clean up country names
hydrogen_capacity = hydrogen_data.groupby('Country')['Capacity_kt H2/y'].sum().reset_index()

# Rename columns for merging
hydrogen_capacity.columns = ['Country', 'HydrogenCapacity_kt']

# Merge hydrogen and renewable data on Country
merged_data = pd.merge(hydrogen_capacity, renewable_data, on='Country', how='inner')

# Display merged data
print(merged_data.head())

# Correlation analysis
correlation = merged_data[['HydrogenCapacity_kt', 'RenewableCapacity_MW']].corr()
print("Correlation Matrix:")
print(correlation)

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix between Hydrogen and Renewable Energy Capacity')
plt.show()

# Visualizing the relationship with a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x='RenewableCapacity_MW', y='HydrogenCapacity_kt', hue='Country', s=100)
plt.title('Hydrogen Production Capacity vs Renewable Energy Capacity')
plt.xlabel('Renewable Energy Capacity (MW)')
plt.ylabel('Hydrogen Production Capacity (kt/y)')
plt.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
