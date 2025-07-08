#导入所需模块
import numpy as np
import os
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

listdir_test = os.listdir('./face_test')
names_test = [d for d in listdir_test if not d.startswith('.')]
print(names_test)
images_test = []
target_test = []
for index_test,dir_test in enumerate(names_test):
    for i in range(1, 31):
        gray_test = cv2.imread('./face_test/%s/%d.jpg' % (dir_test, i))  # 三维图片
        
        #图像均衡化
        gray_test_equalized = histogram_equalization(gray_test)
        
        gray_test_ = gray_test_equalized[:, :, 0]  # 二维数组
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
print(Vt.shape)
print(P.shape)

# 计算平均向量

# 将向量切分成每30个一组
group_size = 10
num_groups = 20
grouped_vectors = X_train_pca.reshape(num_groups, group_size, -1)

# 计算每组的平均向量
X_train_average = np.mean(grouped_vectors, axis=1)

print(X_train_average.shape)

true_count = 0

# Calculate the distance between each test vector and the average vector
for i in range(len(X_test)):
    test_vector = X_test[i]
    
    # Normalize the test vector
    test_vector = test_vector - np.mean(test_vector, axis=0)

    # PCA for the test vector
    test_vector = np.dot(test_vector,P)
    
    # Initialize a list to store the euclidean norm between each training vector and the current test vector
    distances = []
    
    # Calculate the euclidean norm and add it to the list
    for j in range(len(X_train_average)):
        train_vector = X_train_average[j]
        
        # Calculate the euclidean norm
        distance = np.linalg.norm(train_vector - test_vector)
        distances.append(distance)
        
    # Find the training vector with the smallest euclidean norm
    min_distances_index = np.argmin(distances)
    
    # Check if the indexes are the same
    if min_distances_index + 1 == (i // 30) +1:
        true_count += 1
        print(i)