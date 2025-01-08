import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 1. Đọc và tiền xử lý dữ liệu
print("==> Bước 1: Đọc dữ liệu")
df = pd.read_csv('data\\clean_data_final.csv').head(5000)  # Lấy 5000 bản ghi đầu tiên
print("Dữ liệu ban đầu (5 dòng đầu):")
print(df.head())

# Loại bỏ cột 'Gender' không cần thiết cho phân cụm
data = df.drop(columns=['Gender'])

# 2. Chuẩn hóa dữ liệu
print("\n==> Bước 2: Chuẩn hóa dữ liệu")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 3. Giảm chiều dữ liệu bằng PCA
print("\n==> Bước 3: Giảm chiều dữ liệu với PCA")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print("Dữ liệu sau khi giảm chiều (5 dòng đầu):")
print(pca_data[:5])

# 4. Áp dụng thuật toán BIRCH
print("\n==> Bước 4: Phân cụm dữ liệu với BIRCH")
birch_model = Birch(threshold=0.5, branching_factor=50)
birch_model.fit(pca_data)

# Lấy nhãn cụm và centroids
labels = birch_model.labels_
centroids = birch_model.subcluster_centers_

# Thêm nhãn cụm vào DataFrame gốc
df['Cluster'] = labels

# 5. Đánh giá mô hình
print("\n==> Bước 5: Đánh giá mô hình phân cụm")
silhouette = silhouette_score(pca_data, labels)
calinski_harabasz = calinski_harabasz_score(pca_data, labels)
dbi = davies_bouldin_score(pca_data, labels)
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
print(f"Davies-Bouldin Index (DBI): {dbi:.4f}")

# Lưu kết quả đánh giá vào tệp CSV
metrics_df = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'],
    'Value': [silhouette, calinski_harabasz, dbi]
})
metrics_df.to_csv('results\\birch_clustering_metrics.csv', index=False)

# 6. Phân tích đặc điểm từng cụm
print("\n==> Bước 6: Phân tích đặc điểm cụm")
genres_columns = ['War', 'Horror', 'Musical', 'Crime', 'Mystery', 'Film-Noir', 'Animation', 'Thriller',
                  'Documentary', 'Comedy', 'Fantasy', 'Western', 'Adventure', 'Action', 'Sci-Fi', 'Romance',
                  'Childrens', 'Drama']

for cluster_id in np.unique(labels):
    cluster_data = df[df['Cluster'] == cluster_id]
    print(f"\nCụm {cluster_id}:")
    print(f"- Số lượng bản ghi: {len(cluster_data)}")
    print(f"- Độ tuổi trung bình: {cluster_data['Age'].mean():.2f}")
    print(f"- Điểm đánh giá trung bình: {cluster_data['Rating'].mean():.2f}")
    
    # Tính tần suất các thể loại phim
    genre_frequencies = cluster_data[genres_columns].mean()
    print("\n- Tần suất các thể loại phim:")
    print(genre_frequencies.to_string(index=True))
    
    # Thể loại phim phổ biến nhất
    favorite_genre = genre_frequencies.idxmax()
    print(f"- Thể loại phổ biến nhất: {favorite_genre} ({genre_frequencies[favorite_genre]:.4f})")

# Hiển thị tọa độ centroids
print("\nTọa độ centroid của các cụm:")
for i, centroid in enumerate(centroids):
    print(f"Cụm {i}: Centroid (x, y) = ({centroid[0]:.4f}, {centroid[1]:.4f})")

# 7. Vẽ biểu đồ phân cụm
print("\n==> Bước 7: Vẽ biểu đồ phân cụm")
plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap("tab10", len(set(labels)))

# Vẽ từng cụm
for cluster_id in np.unique(labels):
    cluster_points = pca_data[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')

# Vẽ centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow',
            marker='X', edgecolors='black', label='Centroids')

# Trang trí biểu đồ
plt.title("BIRCH Clustering with Centroids")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)

# Lưu biểu đồ vào tệp
plt.savefig('results\\birch_clustering_with_centroids.png')
plt.show()
