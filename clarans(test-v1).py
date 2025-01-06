import pandas as pd
import numpy as np
from pyclustering.cluster.clarans import clarans
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("D:\Code\Python\KPDL\Clarans_data_movie\Iris.csv")
data = df[['SepalLengthCm', 'SepalWidthCm']].values
print(data[:10])

# Giảm chiều dữ liệu xuống 2D bằng PCA trước khi phân cụm
#pca = PCA(n_components=2)
#data_2d = pca.fit_transform(data)


# Cấu hình clarans
k = 3
num_local = 3
max_neighbor = 6
clarans_instance = clarans(data, k, num_local, max_neighbor)
clarans_instance.process()

# Lấy kết quả
clusters = clarans_instance.get_clusters()
medoids = clarans_instance.get_medoids()

# Vẽ biểu đồ
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green']

for idx, cluster in enumerate(clusters):
    cluster_points = np.array([data[i] for i in cluster])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {idx+1}', color=colors[idx])

# Vẽ Medoids
medoids_points = np.array([data[i] for i in medoids])
plt.scatter(medoids_points[:, 0], medoids_points[:, 1], s=200, c='yellow', marker='X', label='Medoids')

plt.title("CLARANS Clustering")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.legend()
plt.grid(True)
plt.show()
