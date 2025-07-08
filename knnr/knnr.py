#导入所需模块
import numpy as np
import os
import cv2
from collections import Counter
import matplotlib.pyplot as plt

def histogram_equalization(image):

    # 计算直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累积直方图
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # 创建映射表
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # 应用映射表
    equalized_image = cdf[image]

    return equalized_image

# 返回的是480张100*100的矩阵，和index
listdir = os.listdir('./face_train')
names = [d for d in listdir if not d.startswith('.')]
images = []
target = []
for index,dir in enumerate(names):
    for i in range(1, 11):
        gray = cv2.imread('./face_train/%s/%d.jpg' % (dir, i))  # 三维图片
        
        #图像均衡化
        gray_equalized = histogram_equalization(gray)
        
        # Display both original and processed images side by side
        plt.subplot(1, 2, 1)  # First subplot
        plt.imshow(gray)
        plt.title(f"Original Image {i}")

        plt.subplot(1, 2, 2)  # Second subplot
        plt.imshow(gray_equalized, cmap='gray')  # Display in grayscale
        plt.title(f"Processed Image {i}")

        plt.show()  # Show the combined plot
        
        gray_ = gray_equalized[:, :, 0]  # 二维数组
        gray_ = cv2.resize(gray_, dsize=(100, 100))
        images.append(gray_)
        target.append(index)
images = np.asarray(images)
target = np.asarray(target)

#图像数据转换特征矩阵
image_data = []

for image in images:
    #转换成一维的数组
    data = image.flatten()
    image_data.append(data)

#print(image_data)
    
#转换为numpy数组
X_train = np.array(image_data)
y_train = target
print(type(X_train))
print(X_train.shape)
print(X_train)

listdir_test = os.listdir('./face_test')
names_test = [d for d in listdir_test if not d.startswith('.')]
print(names_test)
images_test = []
target_test = []
for index_test,dir_test in enumerate(names_test):
    for i in range(1, 31):
        gray_test = cv2.imread('./face_test/%s/%d.jpg' % (dir_test, i))  # 三维图片
        
        #图像均衡化
        gray_test = histogram_equalization(gray_test)
        
        gray_test_ = gray_test[:, :, 0]  # 二维数组
        gray_test_ = cv2.resize(gray_test_, dsize=(100, 100))
        
        images_test.append(gray_test_)
        target_test.append(index_test)
images_test = np.asarray(images_test)
target_test = np.asarray(target_test)

#图像数据转换特征矩阵
image_data_test = []

for image_test in images_test:
    #转换成一维的数组
    data_test = image_test.flatten()
    image_data_test.append(data_test)

#print(image_data)
    
#转换为numpy数组
X_test = np.array(image_data_test)
y_test = target_test
print(type(X_test))
print(X_test.shape)

# 对数据进行中心化，即每一列减去该列的均值
X_train_centered = X_train - np.mean(X_train, axis=0)

# 主成分数量
k = 100
# 对训练矩阵进行SVD
U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)
# 选择前100个主成分
X_train_pca = U[:, :k]
# 构建变换矩阵P
P = np.dot(X_train.T, X_train_pca)
print(P)

true_count = 0

# 计算每个测试数据和训练数据之间的欧几里得范数
for i in range(len(X_test)):
    test_vector = X_test[i]
    
    # 归一化测试向量
    test_vector = test_vector - np.mean(test_vector, axis=0)

    # 对测试向量PCA
    test_vector = np.dot(test_vector,P)
    
    
    # 初始化一个列表来存储每个训练向量与当前测试向量的欧几里得范数
    distances = []
    
    for j in range(len(X_train_pca)):
        train_vector = X_train_pca[j]
        
        dist = np.sqrt(np.sum((train_vector-test_vector) ** 2))
        distances.append(dist)
    
    # 找到最近的k个邻居
    nearest_indices = np.argsort(distances)[:4]
    
    # 找到最频繁的索引
    counts = Counter(nearest_indices//10 + 1)
    most_common_index = counts.most_common(1)[0][0]
    
    # 检查索引是否相同
    if most_common_index == (i // 30) +1:
        true_count += 1
        print(i)

print(f"True的数量：{true_count}")
print(f"True的概率：{true_count/(len(X_test))}")