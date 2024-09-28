import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'hydrogen_projects.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Data Preparation
# Convert the 'Date online' column to datetime format
data['Date online'] = pd.to_datetime(data['Date online'], errors='coerce', format="%Y")
data = data[data['Technology_electricity'] == 'Dedicated renewable']

# Clean up the technology column for consistency
technology_column = 'Country'
data[technology_column] = data[technology_column].str.strip()

# Extract the year from 'Date online'
data['Year'] = data['Date online'].dt.year

# Filter out projects with valid technology and year
technology_data = data[['Year', technology_column, 'Country']].dropna(subset=[technology_column, 'Year'])

# Analyze technology adoption over time
technology_trends = technology_data.groupby(['Year', technology_column]).size().reset_index(name='Count')

# Pivot table for easier visualization
technology_pivot = technology_trends.pivot(index='Year', columns=technology_column, values='Count').fillna(0)
technology_pivot = technology_pivot.astype(int)  # Convert to integer for better visualization


# Plot technology adoption trends over time
plt.figure(figsize=(14, 7))
technology_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Trends in Green Hydrogen Sources Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Projects')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
