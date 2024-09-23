import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'hydrogen_projects.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Data Preparation
# Filter relevant columns for geographical analysis
country_data = data[['Country', 'Project name']].dropna(subset=['Country'])

# Count the number of projects by country
project_count_by_country = country_data.groupby('Country').size().reset_index(name='Number of Projects')

# Load world shapefile using Geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge world geometries with project counts
world = world.merge(project_count_by_country, how='left', left_on='name', right_on='Country')

# Fill NaN values with 0 for countries with no projects
world['Number of Projects'] = world['Number of Projects'].fillna(0)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax)
world.plot(column='Number of Projects', ax=ax, legend=True,
           legend_kwds={'label': "Number of Hydrogen Projects by Country",
                        'orientation': "horizontal"},
           cmap='OrRd', missing_kwds={"color": "lightgrey", "label": "No Projects"})
plt.title('Geographical Distribution of Hydrogen Projects')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(False)
plt.show()
