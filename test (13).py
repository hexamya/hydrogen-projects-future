import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('hydrogen_projects.csv')

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical features with LabelEncoder
label_encoders = {}
categorical_cols = ['Project name', 'Country', 'Status', 'Technology', 'Technology_details',
                    'Technology_electricity', 'Technology_electricity_details', 'Product']
for col in categorical_cols:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
    label_encoders[col] = le


# Select numerical features and normalize them with StandardScaler
numerical_cols = ['Capacity_MWel', 'Capacity_Nm³ H₂/h', 'Capacity_kt H2/y',
                  'Capacity_t CO₂ captured/y',
                  'IEA zero-carbon estimated normalized capacity\r\n[Nm³ H₂/hour]', 'LOWE_CF']

scaler = StandardScaler()
df_imputed[numerical_cols] = scaler.fit_transform(df_imputed[numerical_cols])

# Display the processed dataset
print(df_imputed.head())

features = ['Capacity_MWel', 'Capacity_Nm³ H₂/h', 'Capacity_kt H2/y',
            'Capacity_t CO₂ captured/y', 'IEA zero-carbon estimated normalized capacity\r\n[Nm³ H₂/hour]', 'LOWE_CF']
X = df_imputed[features]

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
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

# Apply k-means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Apply hierarchical clustering
agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters)
agg_labels = agg_clustering.fit_predict(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Compute silhouette scores for each method
kmeans_score = silhouette_score(X, kmeans_labels)
agg_score = silhouette_score(X, agg_labels)
dbscan_score = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

print(f"K-Means Silhouette Score: {kmeans_score}")
print(f"Agglomerative Clustering Silhouette Score: {agg_score}")
print(f"DBSCAN Silhouette Score: {dbscan_score}")



# Choose the best clustering method based on silhouette score
best_labels = kmeans_labels if kmeans_score >= agg_score and kmeans_score >= dbscan_score else (agg_labels if agg_score >= dbscan_score else dbscan_labels)



