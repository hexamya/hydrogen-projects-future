import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'hydrogen_projects.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Data Preparation
# Convert relevant date columns to datetime format
data['Date online'] = pd.to_datetime(data['Date online'], errors='coerce', format="%Y")
data['Decomission date'] = pd.to_datetime(data['Decomission date'], errors='coerce', format="%Y")

# Clean up the Status column
data['Status'] = data['Status'].str.strip().str.lower()

# Calculate project duration in months for projects that have completed the decommissioning
data['Development Duration (months)'] = np.where(
    data['Decomission date'].notna(),
    ((data['Decomission date'] - data['Date online']) / np.timedelta64(1, 'D')).round()/30,
    ((pd.Timestamp.now() - data['Date online']) / np.timedelta64(1, 'D')).round()/30
)

# Filter out projects with a negative or zero duration for practical purposes
data = data[data['Development Duration (months)'] > 0]

# Analyzing factors influencing project duration
plt.figure(figsize=(12, 6))
sns.boxplot(x='Status', y='Development Duration (months)', data=data)
plt.title('Project Development Duration by Status')
plt.xlabel('Project Status')
plt.ylabel('Development Duration (months)')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Further analysis: Factor by Technology
plt.figure(figsize=(12, 6))
sns.boxplot(x='Technology', y='Development Duration (months)', data=data)
plt.title('Project Development Duration by Technology')
plt.xlabel('Technology Type')
plt.ylabel('Development Duration (months)')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation analysis using numerical project features if available
numeric_columns = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()

# Display the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Project Features')
plt.show()

# Display summary statistics
print(data[['Status', 'Development Duration (months)']].groupby('Status').describe())
