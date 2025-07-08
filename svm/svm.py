#导入所需模块
import numpy as np
import os
import cv2

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

class MultiClassSVM:
    def __init__(self, X, y, k, C=1, tol=1e-5, max_iter=1000, learning_rate=0.01):
        self.X = X
        self.y = y
        self.k = k
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        self.W = np.random.rand(self.n_features, self.n_classes)
        self.b = np.random.rand(self.n_classes)
        self.losses = []

    def hinge_loss(self, X, y, W, b):
        scores = np.dot(X, W) + b
        correct_class_score = scores[np.arange(self.n_samples), y]
        margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
        margins[np.arange(self.n_samples), y] = 0
        loss = np.sum(margins) / self.n_samples
        return loss

    def fit(self):
        for i in range(self.max_iter):
            scores = np.dot(self.X, self.W) + self.b
            correct_class_score = scores[np.arange(self.n_samples), self.y]
            margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
            margins[np.arange(self.n_samples), self.y] = 0
            margins[margins > 0] = 1
            row_sum = np.sum(margins, axis=1)
            margins[np.arange(self.n_samples), self.y] = -row_sum
            dW = np.dot(self.X.T, margins) / self.n_samples
            db = np.sum(margins, axis=0) / self.n_samples
            dW += self.C * self.W
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            loss = self.hinge_loss(self.X, self.y, self.W, self.b)
            self.losses.append(loss)
            if i % 10 == 0:
                print('Iter %d / %d: loss %f' % (i, self.max_iter, loss))

    def predict(self, X):
        scores = np.dot(X, self.W) + self.b
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def get_params(self):
        return self.W, self.b
    
    def get_losses(self):
        return self.losses
    
# 对数据进行中心化，即每一列减去该列的均值
X_test_centered = X_test - np.mean(X_train, axis=0) # 注意这里使用的是训练数据的均值

# 使用训练集的Vt进行PCA
X_test_pca = np.dot(X_test_centered, Vt.T[:, :k])

svm = MultiClassSVM(X_train_pca, y_train, k)
svm.fit()

y_pred = svm.predict(X_test_pca)
true_count = np.sum(y_pred == y_test)
print('Number of true predictions:', true_count)
test_accuracy = svm.score(X_test_pca, y_test)  # 注意这里使用的是PCA处理后的测试数据
print('Test accuracy:', test_accuracy)