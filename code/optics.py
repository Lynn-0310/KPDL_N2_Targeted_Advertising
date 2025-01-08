import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Hàm chuẩn hóa dữ liệu
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Hàm giảm chiều dữ liệu
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# Đọc dữ liệu
df = pd.read_csv('data\clean_data_final.csv')
data = df.drop(columns=['Gender']).head(5000).values  # Giới hạn dữ liệu cho bài test
print(data[:20])

# Chuẩn hóa và giảm chiều dữ liệu
scaler_data = preprocess_data(data)
perform_pca_data = perform_pca(scaler_data)

# Định nghĩa hàm sử dụng thuật toán OPTICS
#1
# def optics_clustering(data, min_samples=10, xi=0.05, min_cluster_size=0.05):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#2
# def optics_clustering(data, min_samples=10, xi=0.1, min_cluster_size=0.1):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#3
# def optics_clustering(data, min_samples=15, xi=0.02, min_cluster_size=0.02):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#4
# def optics_clustering(data, min_samples=8, xi=0.03, min_cluster_size=0.05):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#5
# def optics_clustering(data, min_samples=20, xi=0.05, min_cluster_size=0.1):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#6
# def optics_clustering(data, min_samples=12, xi=0.07, min_cluster_size=0.03):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#7
# def optics_clustering(data, min_samples=6, xi=0.04, min_cluster_size=0.04):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#8
# def optics_clustering(data, min_samples=25, xi=0.02, min_cluster_size=0.07):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model

#9
# def optics_clustering(data, min_samples=10, xi=0.06, min_cluster_size=0.08):
#     optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
#     optics_model.fit(data)
#     return optics_model.labels_, optics_model


#10
def optics_clustering(data, min_samples=7, xi=0.03, min_cluster_size=0.03):
    optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics_model.fit(data)
    return optics_model.labels_, optics_model



# Định nghĩa hàm sử dụng thuật toán OPTICS
def optics_clustering(data, min_samples=10, xi=0.001, min_cluster_size=0.05):
    optics_model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics_model.fit(data)
    return optics_model.labels_, optics_model

# Áp dụng OPTICS
labels, optics_model = optics_clustering(perform_pca_data)

# Đánh giá phân cụm
def evaluate_clustering(data, labels):
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)

    return silhouette, calinski_harabasz, davies_bouldin

silhouette, calinski_harabasz, davies_bouldin = evaluate_clustering(perform_pca_data, labels)

# Visualization
def plot_clusters(data, labels):
    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(12, 8))
    for cluster_id in np.unique(labels):
        cluster_data = reduced_data[labels == cluster_id]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {int(cluster_id)}', s=50, alpha=0.7)

    plt.title('OPTICS Clustering Visualization', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig('results\optics\optics_clusters.png')
    plt.show()

plot_clusters(perform_pca_data, labels)

# Hiển thị thông số đánh giá
def display_metrics(silhouette, calinski_harabasz, davies_bouldin):
    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'],
        'Value': [silhouette, calinski_harabasz, davies_bouldin]
    })
    print(metrics_df)
    metrics_df.to_csv('results\optics\optics_evaluation_metrics.csv', index=False)

display_metrics(silhouette, calinski_harabasz, davies_bouldin)
