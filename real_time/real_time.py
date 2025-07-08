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
listdir = os.listdir('Assignment/face_train')
names = [d for d in listdir if not d.startswith('.')]
#print(names)
images = []
target = []
for index,dir in enumerate(names):
    for i in range(1, 11):
        gray = cv2.imread('Assignment/face_train/%s/%d.jpg' % (dir, i))  # 三维图片

        #图片均衡化
        gray_equalized = histogram_equalization(gray)
        
        gray_ = gray_equalized[:, :, 0]  # 二维数组
        gray_ = cv2.resize(gray_, dsize=(100, 100))      
        images.append(gray_)
        target.append(index)
images = np.asarray(images)
target = np.asarray(target)

# 转换成矩阵
#图像数据转换特征矩阵
image_data = []

for image in images:
    #转换成一维的数组
    data = image.flatten()
    image_data.append(data)

#转换为numpy数组
X = np.array(image_data)
print(type(X))
print(X.shape)
print(X)

# 对数据进行中心化，即每一列减去该列的均值
X_train_centered = X - np.mean(X, axis=0)

# 主成分数量
k = 100
# 对训练矩阵进行SVD
U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)
# 选择前100个主成分
X_train_pca = U[:, :k]
print(X_train_pca.shape)
# 构建变换矩阵P
P = np.dot(X.T, X_train_pca)
print(P)

cap = cv2.VideoCapture(0)
# 人脸检测
face_detector = cv2.CascadeClassifier('E:/tools/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
while True:
    flag,frame = cap.read()
    if not flag:
        break
   
    gray = cv2.cvtColor(frame,code = cv2.COLOR_BGR2GRAY)
    
    # 提取测试向量
    faces = face_detector.detectMultiScale(gray, minNeighbors=12)
    for x,y,w,h in faces:       
        
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,dsize=(100,100))
        
        face_equalized = histogram_equalization(face)
        
        imgs = []
        imgs.append(face_equalized)
        imgs = np.asarray(imgs)
        
        #图像数据转换特征矩阵
        image_data = []

        data = imgs.flatten()
        image_data.append(data)
        
        #转换为numpy数组
        test_vector = np.array(image_data)
        #print(test_vector)

        # 归一化测试向量
        test_vector_centered = test_vector - np.mean(test_vector)
        #print(test_vector_centered)
        
        # 对测试向量PCA
        test_vector_pca = np.dot(test_vector_centered,P)
        
        #print(test_vector_pca)

        # 初始化一个列表来存储每个训练向量与当前测试向量的欧几里得范数
        distances = []

        for j in range(len(X_train_pca)):
            train_vector = X_train_pca[j]

            dist = np.sqrt(np.sum((train_vector-test_vector_pca) ** 2))
            distances.append(dist)

        # 找到最近的k个邻居
        nearest_indices = np.argsort(distances)[:3]
        print(nearest_indices)
        # 找到最频繁的索引
        counts = Counter(nearest_indices//10)
        most_common_index = counts.most_common(1)[0][0]
        
        print(most_common_index)

        # 人脸辨识，返回index和置信度
        y_ = most_common_index
        
        label = names[y_]
        
        print('这个人是：%s。'%(label))
        cv2.rectangle(frame,pt1 = (x,y),pt2 = (x+w,y+h),color=[0,0,255], thickness = 2)
        cv2.putText(frame,text = label, org = (x, y-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5,
                    color=[0,0,255],
                    thickness=2)
    cv2.imshow('face',frame)
    key = cv2.waitKey(1000//30)
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
