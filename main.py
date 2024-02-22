import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import scipy.cluster.hierarchy as sch
import featuretools as ft
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load the customer data
customer_data = pd.read_csv('customer_data.csv')

# Drop irrelevant columns if any
customer_data.drop(columns=['CustomerID'], inplace=True)

# Handle missing values if any
customer_data.dropna(inplace=True)

# Separate features and target variable
X = customer_data.drop(columns=['Gender'])  # Exclude 'Gender' for now
y = customer_data['Gender']

# Feature Engineering
es = ft.EntitySet(id='customer_data')
es = es.entity_from_dataframe(entity_id='data', dataframe=customer_data, index='index')
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='data', max_depth=2, verbose=1)
X_engineered = feature_matrix.drop(columns=['Gender'])
X_original = X.copy()
X_combined = pd.concat([X_engineered, X_original], axis=1)

# Feature Scaling and Encoding
numerical_features = X_combined.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_features = ['Gender']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
X_processed = preprocessor.fit_transform(X_combined)

# Dimensionality Reduction using Kernel PCA
kpca = KernelPCA(kernel='rbf')
grid_search_kpca = GridSearchCV(kpca, param_grid={"gamma": np.linspace(0.03, 0.3, 10)}, cv=5)
grid_search_kpca.fit(X_processed)
best_kpca = grid_search_kpca.best_estimator_

# Clustering pipeline
cluster_pipeline = Pipeline([
    ('kpca', best_kpca),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
    ('clustering', KMeans())
])

# Hyperparameter Tuning for KMeans
param_grid = {'clustering__n_clusters': np.arange(2, 11)}
grid_search = GridSearchCV(cluster_pipeline, param_grid, cv=5)
grid_search.fit(X_processed)
best_clusterer = grid_search.best_estimator_

# Fit clustering pipeline
cluster_labels = best_clusterer.fit_predict(X_processed)

# Determine the optimal number of clusters using silhouette score and Davies-Bouldin index
silhouette_avg = silhouette_score(X_processed, cluster_labels)
db_index = davies_bouldin_score(X_processed, cluster_labels)

# Perform ensemble clustering using K-means, hierarchical clustering, DBSCAN, and Spectral Clustering
optimal_n_clusters = best_clusterer.named_steps['clustering'].n_clusters
kmeans_labels = KMeans(n_clusters=optimal_n_clusters, random_state=42).fit_predict(X_processed)
hc_labels = AgglomerativeClustering(n_clusters=optimal_n_clusters, affinity='euclidean', linkage='ward').fit_predict(X_processed)
dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_processed)
spectral_labels = SpectralClustering(n_clusters=optimal_n_clusters, affinity='nearest_neighbors').fit_predict(X_processed)

# Combine cluster labels from different clustering algorithms
ensemble_labels = np.vstack((kmeans_labels, hc_labels, dbscan_labels, spectral_labels)).mean(axis=0).round().astype(int)

# Semi-supervised learning for clustering with label propagation
label_propagation = LabelPropagation()
semi_supervised_labels = label_propagation.fit_predict(X_processed, y)

# Visualize clusters in 2D space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=cluster_data[:, 0], y=cluster_data[:, 1], hue=ensemble_labels, palette='viridis', legend='full')
plt.title('Customer Segmentation based on Kernel PCA')
plt.xlabel('Kernel PCA Component 1')
plt.ylabel('Kernel PCA Component 2')
plt.show()

# Analyze cluster characteristics
cluster_means = pd.DataFrame(cluster_data, columns=['PCA1', 'PCA2'])
cluster_means['Cluster'] = ensemble_labels
cluster_means = cluster_means.groupby('Cluster').mean()

# Visualize cluster characteristics
plt.figure(figsize=(12, 8))
sns.barplot(data=cluster_means.reset_index(), x='Cluster', y='PCA1', palette='muted')
plt.title('Average PCA Component 1 by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average PCA Component 1')
plt.show()

# Repeat visualization for PCA component 2 and other features

