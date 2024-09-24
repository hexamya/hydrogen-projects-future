import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('hydrogen_projects.csv')

# Initial data overview
print(data.info())
print(data.head())

# Fill missing values if necessary
data.fillna(0, inplace=True)

# Convert date columns to datetime
data['Date online'] = pd.to_datetime(data['Date online'], format='%Y', errors='coerce')
data['Decomission date'] = pd.to_datetime(data['Decomission date'], format='%Y', errors='coerce')

# Feature engineering
data['year'] = data['Date online'].dt.year

plt.figure(figsize=(24, 6))
sns.countplot(x='year', hue='Technology', data=data)
plt.title('Hydrogen Production Technology Adoption Over Time')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Country', hue='Technology', data=data)
plt.title('Hydrogen Production Technology by Country')
plt.xticks(rotation=45)
plt.show()

feature_cols = [
    # 'Country',
    # 'Status',
    # 'Technology_electricity',
    # 'Product',
    # 'EndUse_Refining',
    # 'EndUse_Ammonia',
    # 'EndUse_Methanol',
    # 'EndUse_Iron&Steel',
    # 'EndUse_Other Ind',
    # 'EndUse_Mobility',
    # 'EndUse_Power',
    # 'EndUse_Grid inj.',
    # 'EndUse_CHP',
    # 'EndUse_Domestic heat',
    # 'EndUse_Biofuels',
    # 'EndUse_Synfuels',
    # 'EndUse_CH4 grid inj.',
    # 'EndUse_CH4 mobility',
    'Capacity_MWel',
    'Capacity_Nm³ H₂/h',
    'Capacity_kt H2/y',
    'Capacity_t CO₂ captured/y'
]
X = data[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(0))


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolors='k', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Hydrogen Production Technologies Metrics')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_pca)
labels = kmeans.labels_

plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clustering of Hydrogen Production Technologies')
plt.show()


data['Cluster'] = labels
X['Cluster'] = labels

plt.figure(figsize=(12, 6))
sns.countplot(x='Cluster', hue='Technology', data=data)
plt.title('Technology Distribution within Clusters')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Cluster', hue='Country', data=data)
plt.title('Country Distribution within Clusters')
plt.show()

plt.figure(figsize=(12, 6))
sns.pairplot(data=X, hue='Cluster', palette='viridis', diag_kind='kde', markers='o')
plt.title('Country Distribution within Clusters')
plt.show()
