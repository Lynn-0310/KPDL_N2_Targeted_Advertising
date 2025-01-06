import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pyclustering.cluster.clarans import clarans
# from pyclustering.cluster import cluster_visualizer_multidim
# from pyclustering.utils.metric import type_metric, distance_metric
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu và lấy n điểm đầu tiên
data_path = './input/data_csv/clean_data_final.csv'
data = pd.read_csv(data_path)
data = data.head(500)

# # Hiển thị dữ liệu trước xử lý
data.info()
print(data.describe())

# 2. Chuẩn hóa dữ liệu
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)
# #Loại bỏ các cột không phải là số
# data = data.select_dtypes(include=[np.number])

# # Hiển thị dữ liệu sau khi chuẩn hóa
# data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns)
# print(data_scaled_df.head())

# 3. Phân cụm với CLARANS
# # Thiết lập metric khoảng cách
# metric = distance_metric(type_metric.EUCLIDEAN)

# Thiết lập thuật toán CLARANS
num_clusters = 2  # Số cụm muốn phân
num_local = 2  # Số lượng tối ưu hóa địa phương
max_neighbors = 5  # Số lượng hàng xóm tối đa

# clarans_instance = clarans(data_scaled.tolist(), num_clusters, num_local, max_neighbors)
clarans_instance = clarans(data, num_clusters, num_local, max_neighbors)
clarans_instance.process()

# Kết quả phân cụm
clusters = clarans_instance.get_clusters()
medoids = clarans_instance.get_medoids()

# 4. Đánh giá cụm
# Gán nhãn cho từng điểm dữ liệu
labels = np.full(data.shape[0], -1, dtype=int)
for cluster_id, cluster_points in enumerate(clusters):
    # for point in cluster_points:
    #     labels[point] = cluster_id
    labels[np.array(cluster_points)] = cluster_id

# Tính Silhouette Score
silhouette_avg = silhouette_score(data, labels)

# Tính Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(data, labels)

# sample_indices = np.random.choice(data_scaled.shape[0], size=500, replace=False)
# silhouette_avg = silhouette_score(data_scaled[sample_indices], labels[sample_indices])
# calinski_harabasz = calinski_harabasz_score(data_scaled[sample_indices], labels[sample_indices])

# # Tính Dunn Index
# def dunn_index(data, clusters):
#     def intercluster_distances(data, clusters):
#         distances = []
#         for i, cluster_a in enumerate(clusters):
#             for cluster_b in clusters[i+1:]:
#                 distances.append(
#                     np.min([np.linalg.norm(data[p1] - data[p2]) for p1 in cluster_a for p2 in cluster_b])
#                 )
#         return min(distances)
#
#     def intracluster_distances(data, clusters):
#         return max([np.max([np.linalg.norm(data[p1] - data[p2]) for p1 in cluster for p2 in cluster]) for cluster in clusters])
#
#     return intercluster_distances(data, clusters) / intracluster_distances(data, clusters)
#
# dunn = dunn_index(data_scaled, clusters)

# 5. Trực quan hóa cụm
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'olive']
for idx, cluster in enumerate(clusters):
    # cluster_data = np.array([data_scaled[point] for point in cluster_points])
    cluster_points = np.array([data[i] for i in cluster])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {idx + 1}', color=colors[idx])

# Đánh dấu medoids
medoids_points = np.array([data[i] for i in medoids])
plt.scatter(medoids_points[:, 0], medoids_points[:, 1], s=200, c='yellow', marker='X', label='Medoids')

plt.title('CLARANS Clustering Visualization')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend()
plt.grid(True)
plt.show()

# 6. Biểu đồ đánh giá cụm
scores = {
    'Silhouette Score': silhouette_avg,
    'Calinski-Harabasz Index': calinski_harabasz,
    # 'Dunn Index': dunn
}

# Hiển thị bảng đánh giá
score_table = pd.DataFrame(scores.items(), columns=['Metric', 'Value'])
print(score_table)

# Vẽ biểu đồ đánh giá
# plt.figure(figsize=(10, 6))
# plt.bar(scores.keys(), scores.values(), color=['blue', 'green', 'orange'])
# plt.title('Cluster Evaluation Metrics')
# plt.ylabel('Score')
# plt.show()
plt.figure(figsize=(8, 4))
plt.bar(['Silhouette Score', 'Calinski-Harabasz Index'], [silhouette_avg, calinski_harabasz], color=['blue', 'green'])
for i, v in enumerate([silhouette_avg, calinski_harabasz]):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
plt.title('Cluster Evaluation Metrics')
plt.ylabel('Score')
plt.show()

# # Tìm thể loại với rating cao nhất
# original_data = pd.read_csv(data_path).head(1000)  # Lấy lại dữ liệu gốc
# rating_by_genre = original_data.groupby('Thể loại')['Ratings'].max()
#
# # Vẽ biểu đồ
# rating_by_genre.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), color='skyblue')
# plt.title('Maximum Ratings by Genre')
# plt.xlabel('Genre')
# plt.ylabel('Maximum Rating')
# plt.xticks(rotation=45, ha='right')
# plt.show()

# 7. Hiển thị cụm sau tiền xử lý
clustered_data = data.copy()
clustered_data['Cluster'] = labels
print(clustered_data.head())

# 8. Lưu dữ liệu cụm vào file CSV
import os
os.makedirs('./output', exist_ok=True)
clustered_data.to_csv('./output/clustered_data.csv', index=False)
