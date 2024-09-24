import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('hydrogen_projects.csv')

# Display the first few rows of the dataset
print(df.head())

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical features
label_encoders = {}
categorical_cols = ['Project name', 'Country', 'Status', 'Technology', 'Technology_details',
                    'Technology_electricity', 'Technology_electricity_details', 'Product']
for col in categorical_cols:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
    label_encoders[col] = le

# Select numerical features and normalize them
numerical_cols = ['Capacity_MWel', 'Capacity_Nm³ H₂/h', 'Capacity_kt H2/y',
                  'Capacity_t CO₂ captured/y', 'LOWE_CF']

scaler = StandardScaler()
df_imputed[numerical_cols] = scaler.fit_transform(df_imputed[numerical_cols])

# Display the processed dataset
print(df_imputed.head())


features = [
    # 'Country',
    'Status',
    'Technology',
    'Technology_electricity',
    'Product',
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
    'Capacity_t CO₂ captured/y',
    'LOWE_CF'
]

X = df_imputed[features]


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define the number of clusters
n_clusters = 3  # This is an adjustable parameter

# Initialize and fit the k-means model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster assignments back to the dataset
df_imputed['Cluster'] = clusters

# Display the first few rows of the dataset with cluster assignments
print(df_imputed.head())

# Visualize the clusters (using two principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca_result = pca.fit_transform(X)
df_imputed['PCA1'] = pca_result[:, 0]
df_imputed['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = df_imputed[df_imputed['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}")

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters Visualization')
plt.legend()
plt.show()
