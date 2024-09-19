import os
import cv2
from PIL import Image
import numpy as np

# 人脸训练集路径
path = './Facedata/'
# 初始化识别器
# opencv-contrib-python和opencv-python库版本要一致，否则运行会报错
recognizer = cv2.face.LBPHFaceRecognizer_create()
# 获取分类器
detector = cv2.CascadeClassifier(r'./cv2data/haarcascade_frontalface_default.xml')

# 获取图像及标签
def getImagesAndLabels(path):
    # join函数的作用
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids

# 主程序入口
if __name__ == "__main__":
    print('Training faces. It will take a few seconds. Waiting...')
    faces, ids = getImagesAndLabels(path)
    # 开始训练
    print(ids)
    recognizer.train(faces, np.array(ids))         
    print('Training has finished!')
    # 保存文件
    recognizer.write(r'./Model/trainer20221230.yml')  
    print("{0} faces trained. Exiting Program.".format(len(np.unique(ids))))

    
