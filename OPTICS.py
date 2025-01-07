# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Đọc dữ liệu
data = pd.read_csv('clean_data_final.csv', nrows=100000)
np.seterr(divide='ignore', invalid='ignore')

# Kiểm tra đầu vào dữ liệu
print("\nDữ liệu đầu vào:")
print(data.head(10))   

df = pd.DataFrame(data)

# Tiến hành chuẩn hóa dữ liệu 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(columns=['Gender']))

# Áp dụng OPTICS
optics = OPTICS(min_samples=2, xi=0.05, min_cluster_size=0.1)
optics.fit(scaled_data)

# Lấy nhãn của các cụm
labels = optics.labels_

# Thêm nhãn vào DataFrame để phân tích
df['Cluster'] = labels

# Tóm tắt kết quả phân cụm
cluster_summary = df['Cluster'].value_counts().reset_index()
cluster_summary.columns = ['Cluster', 'Count']
print("\nTóm tắt kết quả phân cụm:")
print(cluster_summary)

 # Printing detailed clustering results with 2D coordinates
print("\nCluster Details with 2D Coordinates (Rating, Age):")


# In chi tiết kết quả phân cụm
print("\nChi tiết kết quả phân cụm:")
for cluster in np.unique(labels):
    if cluster == -1:
        print(f"\nCluster -1 (Điểm nhiễu):")
    else:
        print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster])  

# Vẽ biểu đồ phân cụm với legend nằm ngoài đồ thị
plt.figure(figsize=(10, 6))
unique_clusters = np.unique(labels)

for cluster in unique_clusters:
    if cluster == -1:  # Điểm nhiễu
        plt.scatter(
            df[df['Cluster'] == cluster]['Rating'],
            df[df['Cluster'] == cluster]['Age'],
            label='Noise',
            color='black',
            marker='x'
        )
    else:
        plt.scatter(
            df[df['Cluster'] == cluster]['Rating'],
            df[df['Cluster'] == cluster]['Age'],
            label=f'Cluster {cluster}'
        )


# Xem xét các đặc điểm của Cluster 0 và Cluster 1
cluster_0 = df[df['Cluster'] == 0]
cluster_1 = df[df['Cluster'] == 1]



# Thống kê mô tả
print("\nCluster 0: Đặc điểm")
print(cluster_0.describe())

print("\nCluster 1: Đặc điểm")
print(cluster_1.describe())

# Xem xét các thể loại phim yêu thích (tính trung bình của các thể loại phim)
print("\nTần suất các thể loại phim trong Cluster 0:")

genres_columns = ['War', 'Horror', 'Musical', 'Crime', 'Mystery', 'Film-Noir', 'Animation', 'Thriller', 'Documentary', 'Comedy', 'Fantasy', 'Western', 'Adventure', 'Action', 'Sci-Fi', 'Romance', 'Childrens', 'Drama']  
genre_frequencies_0 = cluster_0[genres_columns].mean()
print(genre_frequencies_0.to_string(index=True))

# Xác định thể loại phim yêu thích nhất trong Cluster 0
favorite_genre_cluster_0 = genre_frequencies_0.idxmax()
print(f"Thể loại phim yêu thích nhất trong Cluster 0: {favorite_genre_cluster_0} với giá trị trung bình {genre_frequencies_0[favorite_genre_cluster_0]:.4f}")

print("\nTần suất các thể loại phim trong Cluster 1:")
genre_frequencies_1 = cluster_1[genres_columns].mean()
print(genre_frequencies_1.to_string(index=True))

# Xác định thể loại phim yêu thích nhất trong Cluster 1
favorite_genre_cluster_1 = genre_frequencies_1.idxmax()
print(f"Thể loại phim yêu thích nhất trong Cluster 1: {favorite_genre_cluster_1} với giá trị trung bình {genre_frequencies_1[favorite_genre_cluster_1]:.4f}")

# In thêm thông tin về độ tuổi trung bình và điểm đánh giá trung bình của Cluster 0 và Cluster 1
print("\nThông tin thêm về Cluster 0:")
print(f"Độ tuổi trung bình: {cluster_0['Age'].mean():.2f}")
print(f"Điểm đánh giá trung bình: {cluster_0['Rating'].mean():.2f}")

print("\nThông tin thêm về Cluster 1:")
print(f"Độ tuổi trung bình: {cluster_1['Age'].mean():.2f}")
print(f"Điểm đánh giá trung bình: {cluster_1['Rating'].mean():.2f}")

# Kiểm tra số cụm hợp lệ
valid_clusters = len(np.unique(labels[labels != -1])) > 1

if valid_clusters:
    silhouette_avg = silhouette_score(scaled_data, labels)
    print(f"\nSilhouette Score: {silhouette_avg:.4f}")
    dbi_score = davies_bouldin_score(scaled_data, labels)
    print(f"Davis-Bouldin Index: {dbi_score:.4f}")
else:
    print("\nKhông đủ cụm hợp lệ để tính Silhouette Score hoặc Davis-Bouldin Index.")


# Vẽ biểu đồ phân cụm
plt.title('Clustering Result using OPTICS')
plt.xlabel('Rating')
plt.ylabel('Age')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Clusters")  
plt.tight_layout(rect=[0, 0, 0.8, 1])  

plt.show()
