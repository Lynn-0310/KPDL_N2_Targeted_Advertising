import pandas as pd
import numpy as np
from pyclustering.cluster.clarans import clarans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
import os
from datetime import datetime
import matplotlib.pyplot as plt

#Hàm chuẩn hóa dữ liệu
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

#Hàm giảm chiều dữ liệu
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


# Đọc dữ liệu
df = pd.read_csv("data\clean_data_final.csv")
data = df.drop(columns = ['Gender']).head(500).values
#data = df[['Occupation','Rating','Age']].head(5000).values
print(data[:20])

#Chuẩn hóa và giảm chiều
scaler_data = preprocess_data(data)
perform_pca_data = perform_pca(scaler_data)

# Cấu hình clarans
k = 5
num_local = 1
max_neighbor =10
clarans_instance = clarans(perform_pca_data, k, num_local, max_neighbor)
clarans_instance.process()

# Lấy kết quả
clusters = clarans_instance.get_clusters()
medoids = clarans_instance.get_medoids()

# Tạo danh sách nhãn từ các cụm
labels = np.zeros(perform_pca_data.shape[0], dtype=int)
for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        labels[index] = cluster_id
#Các chỉ số đánh giá thuật toán
silhouette_avg = silhouette_score(perform_pca_data, labels)
calinski_harabasz = calinski_harabasz_score(perform_pca_data, labels)
davies_bouldin = davies_bouldin_score(perform_pca_data, labels)


# Vẽ biểu đồ
plt.figure(figsize=(10, 7))

for idx, cluster in enumerate(clusters):
    cluster_points = np.array([perform_pca_data[i] for i in cluster])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {idx+1}')

# Vẽ Medoids
medoids_points = np.array([perform_pca_data[i] for i in medoids])
plt.scatter(medoids_points[:, 0], medoids_points[:, 1], s=200, c='yellow', marker='X', label='Medoids')

plt.title("CLARANS Clustering")
plt.legend()
plt.grid(True)
# Tạo timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Đường dẫn lưu biểu đồ
output_dir1 = 'results\clarans'
output_file1 = os.path.join(output_dir1, f'plot_{timestamp}.png')

# Lưu biểu đồ
plt.savefig(output_file1, format='png', dpi=300)
print(f"Biểu đồ đã được lưu tại: {output_file1}")

# Hiển thị biểu đồ (tùy chọn)
plt.show()

# Tạo DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Calinski-Harabasz Index','Davies-Bouldin Index'],
    'Value': [silhouette_avg,calinski_harabasz, davies_bouldin]
})
# Hiển thị nội dung DataFrame
print(metrics_df)
# Đường dẫn lưu file CSV
output_dir = 'results\clarans'
output_file = os.path.join(output_dir, 'clarans_evaluation_metrics.csv')


# Kiểm tra xem file đã tồn tại chưa
if os.path.exists(output_file):
    # Nếu file tồn tại, thêm dữ liệu vào
    metrics_df.to_csv(output_file, mode='a', index=False, header=False)
    print(f"Đã thêm dữ liệu mới vào file: {output_file}")
else:
    # Nếu file chưa tồn tại, tạo file mới và thêm dữ liệu
    metrics_df.to_csv(output_file, mode='w', index=False, header=True)
    print(f"File chưa tồn tại. Đã tạo mới và thêm dữ liệu vào file: {output_file}")