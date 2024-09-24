import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns

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

# Define a function to plot silhouette scores
def plot_silhouette_scores(X, max_clusters=10):
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    plt.plot(cluster_range, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Scores for k-means clustering')
    plt.show()

# Plot silhouette scores to find the optimal number of clusters for k-means
plot_silhouette_scores(X, max_clusters=10)

# Optimal number of clusters found from silhouette scores
optimal_clusters = 3  # This could be adjusted based on silhouette scores

# Initialize and fit the k-means model
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster assignments back to the dataset
df_imputed['Cluster'] = clusters
df["Cluster"] = clusters

# Display the first few rows of the dataset with cluster assignments
print(df_imputed.head())

# Visualize the clusters (using two principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df_imputed['PCA1'] = pca_result[:, 0]
df_imputed['PCA2'] = pca_result[:, 1]
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    cluster_data = df_imputed[df_imputed['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}")

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters Visualization')
plt.legend()
plt.show()


#
plt.figure(figsize=(12, 12))
sns.pairplot(
    df_imputed[[
        'Country',
        'Status',
        'Technology',
        'Technology_electricity',
        'Product',
        'Cluster'
    ]],
    hue="Cluster"
)
plt.title('Pair plot Visualization')
plt.show()



#
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate linkage matrix
Z = linkage(X, 'ward')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

df.to_csv("cluster.csv", index=False)