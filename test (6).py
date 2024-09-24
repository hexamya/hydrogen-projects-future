import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'hydrogen_projects.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

def convert_to_mw(value):
    try:
        if 'GW' in value:
            return float(value.split('GW')[0]) * 1000
        elif 'MW' in value:
            return float(value.split('MW')[0])
        elif 'kW' in value:
            return float(value.split('kW')[0]) / 1000
        elif 'W' in value:
            return float(value.split('W')[0]) / 1000000
        elif 't H2/y' in value:
            return float(value.split('t H2/y')[0]) * 0.05  # Rough estimate
        elif 'kt H2/y' in value:
            return float(value.split('kt H2/y')[0]) * 50  # Rough estimate
        elif 'Mt H2/y' in value:
            return float(value.split('Mt H2/y')[0]) * 50000  # Rough estimate
        elif 'kt NH3/y' in value:
            return float(value.split('kt NH3/y')[0]) * 15  # Rough estimate
        elif 'Mt NH3/y' in value:
            return float(value.split('Mt NH3/y')[0]) * 15000  # Rough estimate
        else:
            return None  # Placeholder for non-convertible values
    except:
        return None  # Return -1 if conversion fails


# Data Preparation
# Clean up relevant columns
data['Technology'] = data['Technology'].str.strip()
data['Capacity_kt H2/y'] = pd.to_numeric(data['Capacity_kt H2/y'], errors='coerce')
data['Announced Size MW'] = data['Announced Size'].apply(lambda x: convert_to_mw(x))

data = data[data['Announced Size MW'].notna()]

# Example capital and operating costs (these are hypothetical values for demonstration)
# These values should ideally come from actual project data
cost_data = {
    'Technology': ['Other Electrolysis', 'ALK', 'PEM', 'SOEC', 'NG w CCUS', 'Other', 'Biomass', 'Biomass w CCUS'],
    'Capital Cost ($/kW)': [1800, 1400, 1600, 2200, 1100, 1500, 2000, 2500],
    'Operating Cost ($/ton H2)': [0.0045, 0.0038, 0.0042, 0.0050, 0.0022, 0.0035, 0.0048, 0.0055],
    'Energy Efficiency (%)': [65, 70, 68, 75, 76, 60, 55, 52],
    'CO2 Emissions (kg CO2/ton H2)': [0.0015, 0.0012, 0.0010, 0.0008, 0.0018, 0.0020, 0.0005, 0.0002]
}

cost_df = pd.DataFrame(cost_data)

# Merge cost data with the main dataset
data = data.merge(cost_df, on='Technology', how='left')

# LCOH Calculation
# LCOH Formula: LCOH = (Capital Cost + Operating Cost * Production Time) / Total Hydrogen Production
# Assuming a fixed production time of 8000 hours per year
production_time = 8000  # operational hours per year

data['Total Hydrogen Production (ton/y)'] = data['Capacity_kt H2/y'] * 1000  # converting from kt to tons
data['LCOH ($/ton H2)'] = (data['Capital Cost ($/kW)'] * data['Announced Size MW'] +
                            data['Operating Cost ($/ton H2)'] * production_time) / data['Total Hydrogen Production (ton/y)']

# Visualizing LCOH by Technology
plt.figure(figsize=(12, 6))
sns.boxplot(x='Technology', y='LCOH ($/ton H2)', data=data)
plt.title('Levelized Cost of Hydrogen Production by Technology')
plt.xlabel('Technology Type')
plt.ylabel('LCOH ($/ton H2)')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display the LCOH estimates
print(data[['Technology', 'LCOH ($/ton H2)']].groupby('Technology').describe())
