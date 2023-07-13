from sklearn.cluster import KMeans
import numpy as np

# 准备数据
data = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# 创建 K-means 对象
kmeans = KMeans(n_clusters=2)

# 拟合数据
kmeans.fit(data)

# 获取聚类结果
labels = kmeans.labels_
print(labels)  # 输出每个数据点所属的簇的标签

# 获取聚类中心点
centroids = kmeans.cluster_centers_
print(centroids)  # 输出每个簇的中心点坐标