import pandas as pd
import numpy as np
from pyclustering.cluster.clarans import clarans
# from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# # Đọc dữ liệu
# def load_data():
#     movies = pd.read_csv('./input/data_dat/movies.dat', sep='::', header=None, engine='python',
#                          names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
#     ratings = pd.read_csv('./input/data_dat/ratings.dat', sep='::', header=None, engine='python',
#                           names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')
#     users = pd.read_csv('./input/data_dat/users.dat', sep='::', header=None, engine='python',
#                         names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')
#
#     return movies, ratings, users
#
# movies, ratings, users = load_data()

#Hàm chuẩn hóa dữ liệu
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

#Hàm giảm chiều dữ liệu
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

df = pd.read_csv('./input/data_csv/clean_data_final.csv')
data = df.drop(columns = ['Gender']).head(2000).values
print(data[:20])

scaler_data = preprocess_data(data)
perform_pca_data = perform_pca(scaler_data)

# # Xử lý dữ liệu
# def preprocess_data(movies, ratings):
#     # Kết hợp movie với rating
#     data = ratings.merge(movies, on='MovieID')
#
#     # Giới hạn n dữ liệu sử dụng để tiến hành phân cụm (theo test hiện tại có thể xử lý 15000 dữ liệu trong 2 phút)
#     data = data.head(2000)
#
#     # Tạo pivot
#     pivot_table = data.pivot_table(index='UserID', columns='Title', values='Rating', fill_value=0)
#
#     # Standardize data
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(pivot_table.values)
#
#     return standardized_data
# #     return pivot_table
# #
# # pivot_table = preprocess_data(movies, ratings)
# # data_matrix = pivot_table.values
# data_matrix = preprocess_data(movies, ratings)

# Định nghĩa hàm sử dụng thuật toán Clarans
def clarans_clustering(data, num_clusters, num_local, max_neighbors):
    # metric = distance_metric(type_metric.EUCLIDEAN)
    # clarans_instance = clarans(data.tolist(), num_clusters, num_local, max_neighbors)
    clarans_instance = clarans(perform_pca_data, num_clusters, num_local, max_neighbors)
    clarans_instance.process()

    clusters = clarans_instance.get_clusters()
    medoids = clarans_instance.get_medoids()
    return clusters, medoids

# Áp dụng Clarans
num_clusters = 5
num_local = 1
max_neighbors = 5
# clusters, medoids = clarans_clustering(data_matrix, num_clusters, num_local, max_neighbors)
clusters, medoids = clarans_clustering(perform_pca_data, num_clusters, num_local, max_neighbors)

# Đánh giá phân cụm
def evaluate_clustering(data, clusters):
    labels = np.zeros(data.shape[0])
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id

    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)

    return silhouette, calinski_harabasz, davies_bouldin, labels

# silhouette, calinski_harabasz, davies_bouldin, labels = evaluate_clustering(data_matrix, clusters)
silhouette, calinski_harabasz, davies_bouldin, labels = evaluate_clustering(perform_pca_data, clusters)

# Visualization
def plot_clusters(data, labels):
    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(12, 8))
    for cluster_id in np.unique(labels):
        cluster_data = reduced_data[labels == cluster_id]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {int(cluster_id)}', s=50, alpha=0.7)

    medoids_points = np.array([perform_pca_data[i] for i in medoids])
    plt.scatter(medoids_points[:, 0], medoids_points[:, 1], s=200, c='yellow', marker='X', label='Medoids')

    plt.title('CLARANS Clustering Visualization', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    plt.legend(fontsize=10)
    plt.savefig('./output/png/clarans_clusters.png')
    plt.show()

# plot_clusters(data_matrix, labels)
plot_clusters(perform_pca_data, labels)

# Hiển thị thông số đánh giá
def display_metrics(silhouette, calinski_harabasz, davies_bouldin):
    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'],
        'Value': [silhouette, calinski_harabasz, davies_bouldin]
    })
    print(metrics_df)
    metrics_df.to_csv('./output/metrics/clarans_evaluation_metrics.csv', index=False)

display_metrics(silhouette, calinski_harabasz, davies_bouldin)
