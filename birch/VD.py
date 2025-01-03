import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Đọc dữ liệu
data = pd.read_csv('clean_data_final.csv')

# Tách các cột đặc trưng (loại bỏ Rating và Gender vì chúng có thể không liên quan đến phân cụm)
X = data.drop(columns=['Rating', 'Gender'])

# Lấy mẫu 200 bản ghi ngẫu nhiên
sample_data = X.sample(n=5000, random_state=42)
# Hiển thị một số bản ghi mẫu
print(sample_data.head())

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X_scaled = scaler.fit_transform(sample_data)

# Áp dụng thuật toán BIRCH
birch = Birch(threshold=1.75, branching_factor=50)
labels = birch.fit_predict(X_scaled)
centroids = birch.subcluster_centers_

# Sử dụng PCA để giảm chiều dữ liệu xuống 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_2d = pca.transform(centroids)

print(X_pca[:5])
#Danh gia mo hinh
silhouette = silhouette_score(X_scaled, labels)
print("Chi so danh gia mo hinh: ", silhouette)

# In ra nhãn cụm
print("Cluster Labels:")
print(np.unique(labels))

# In ra tâm cụm
#print("\nCluster Centroids (in scaled space):")
#for i, centroid in enumerate(centroids_2d):
 #   print(f"Centroid {i}: {centroid}")

# Tạo biểu đồ
plt.figure(figsize=(10, 7))

# Vẽ điểm dữ liệu theo các cụm
for cluster in np.unique(labels):
    cluster_points = X_pca[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.6)

# Vẽ tâm cụm (centroids) dưới dạng dấu X màu đỏ
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=100, label='Centroids')

# Tùy chỉnh biểu đồ
plt.title('Cluster Visualization with BIRCH (PCA Reduced to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
