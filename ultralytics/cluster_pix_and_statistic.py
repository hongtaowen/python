import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取图像
image_path = r'D:\code pro1\pythonProject\gitbase\datasets\mydata\test\images_masked\TY1795_1_jpg.rf.0dff741a0fc24734a50ce79211a40dd6.jpg'  # 替换为您的图像路径
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

# 转换为RGB颜色空间
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 重塑图像为二维数组
pixels = image_rgb.reshape((-1, 3))

# 使用KMeans进行聚类
n_clusters = 3  # 可以根据需要调整聚类数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_

# 获取每个聚类的中心颜色
centers = kmeans.cluster_centers_
centers = np.uint8(centers)

# 定义每个聚类的可视化颜色为HSV
visual_colors_hsv = [
    [154, 100, 63],  # 孔雀绿
    [0, 47, 94],  # 亮珊瑚
    [60, 100, 100],  # 黄色
    [60, 100, 50],  # 橄榄色
    [0, 191, 165],  # 棕色
]

# 确保颜色数组是uint8类型
visual_colors_hsv = np.array(visual_colors_hsv, dtype=np.uint8)

# 转换回RGB颜色空间
visual_colors_rgb = cv2.cvtColor(visual_colors_hsv.reshape(1, -1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)

# 显示聚类中心的颜色
for i, center in enumerate(centers):
    print(f'Cluster {i} center in RGB: {center}')

# 显示每个聚类的颜色范围
ranges = {}
for i in range(n_clusters):
    mask = (labels == i)
    cluster_pixels = pixels[mask]
    r_min, g_min, b_min = np.min(cluster_pixels, axis=0)
    r_max, g_max, b_max = np.max(cluster_pixels, axis=0)
    ranges[f'Cluster {i}'] = {'R': (r_min, r_max), 'G': (g_min, g_max), 'B': (b_min, b_max)}

for cluster, range_values in ranges.items():
    print(f'{cluster} range:')
    print(f"R: {range_values['R']}")
    print(f"G: {range_values['G']}")
    print(f"B: {range_values['B']}")

# 计算每个聚类的百分比
total_pixels = pixels.shape[0]
cluster_percentages = {}
for i in range(n_clusters):
    cluster_count = np.sum(labels == i)
    cluster_percentages[f'Cluster {i}'] = (cluster_count / total_pixels) * 100

# 打印每个聚类的百分比
for cluster, percentage in cluster_percentages.items():
    print(f'{cluster} percentage: {percentage:.2f}%')

# 创建可视化的图像
segmented_image = visual_colors_rgb[labels].reshape(image_rgb.shape)

# 可视化原图像和聚类结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
