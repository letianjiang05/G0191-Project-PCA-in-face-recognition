import cv2
import os
import numpy as np
def load_data():
    listdir = os.listdir('./faces')
    names = [d for d in listdir if not d.startswith('.')]
    #print(names)
    faces = []
    target = []
    for index,dir in enumerate(names):
        for i in range(1, 31):
            gray = cv2.imread('./faces/%s/%d.jpg' % (dir, i))  # 三维图片
            gray_ = gray[:, :, 0]  # 二维数组
            gray_ = cv2.resize(gray_, dsize=(100, 100))
            #gray_ = cv2.equalizeHist(gray_) # 图片均衡化处理，颜色鲜明
            faces.append(gray_)
            target.append(index)
    faces = np.asarray(faces)
    target = np.asarray(target)

    return faces,target,names


def split_data(faces,target):
    index = np.arange(390)
    np.random.shuffle(index)  # 洗牌
    faces = faces[index]
    target = target[index]
    # 训练数据和测试数据
    X_train, X_test = faces[:270], faces[270:]
    # 目标值
    y_train, y_test = target[:270], target[270:]
    return X_train,X_test,y_train,y_test


def take_photo(path):

    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('E:/tools/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    filename = 1
    flag_write = False
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, minNeighbors=10)
        for x, y, w, h in faces:
            if flag_write:
                face = gray[y:y+h,x:x+w]
                face = cv2.resize(face, dsize=(100, 100))
                cv2.imwrite('./faces_dynamic/%s/%d.jpg' % (path,filename), face)
                filename += 1

            cv2.rectangle(frame,pt1 = (x,y),pt2= (x+w,y+h),color=[0,0,255],thickness = 2)

        if filename > 40:
            break
        cv2.imshow('face', frame)
        key = cv2.waitKey(int(1000 / 60))

        if key == ord('q'):
            break
        if key == ord('w'):
            flag_write = True

    cv2.destroyAllWindows()
    cap.release()


def take_faces():
    global key
    while True:
        key = input('请输入文件夹的名字，姓名拼音，如果Q，程序退出!')
        if key == 'Q':
            break
        # 在faces_dynamic下创建子文件夹
        os.makedirs('./faces/%s' % (key), exist_ok=True)
        take_photo(key)


def dynamic_recognizer_face(face_recognizer,names):
    cap = cv2.VideoCapture(0)
    # 人脸检测
    face_detector = cv2.CascadeClassifier('E:/tools/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    while True:
        flag,frame = cap.read()
        if not flag:
            break
        gray = cv2.cvtColor(frame,code = cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, minNeighbors=10)
        for x,y,w,h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,dsize=(100,100))
            #face = cv2.equalizeHist(face)
            y_, confidence = face_recognizer.predict(face) # 人脸辨识
            label = names[y_]
            print('这个人是：%s。置信度是：%0.1f'%(label,confidence))
            cv2.rectangle(frame,pt1 = (x,y),pt2 = (x+w,y+h),color=[0,0,255], thickness = 2)
            cv2.putText(frame,text = label, org = (x, y-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=[0,0,255],
                        thickness=2)
        cv2.imshow('face',frame)
        key = cv2.waitKey(1000//60)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # 动态采集人脸
    #take_faces()
    # 加载数据，返回目标值
    faces, target, names = load_data()
    # 加载算法
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 算法训练，找到数据和目标值之间的规律
    face_recognizer.train(faces,target)
    # 动态加载数据
    dynamic_recognizer_face(face_recognizer,names)