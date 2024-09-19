import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PIL import ImageFont, ImageDraw, Image

class FaceRecogition(QObject):
    change_pixmap_signal = pyqtSignal(QImage)
    end_recogition = False

    def __init__(self):
        super().__init__()


    def cv2AddChineseText(self,frame, name, position, fill):
        font = ImageFont.truetype('simsun.ttc', 30)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, name, font=font, fill=fill)
        return np.array((img_pil))

    def capture(self):
        # 初始化识别器
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('./Model/trainer.yml')
        faceCascade = cv2.CascadeClassifier(r'./cv2data/haarcascade_frontalface_default.xml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        names = []  # 这里应该加载名字与ID的对应关系
        with open('id.txt', 'r', encoding='utf-8') as file:
            for line in file:
                name, id_ = line.strip().split(':')
                names.append(name)

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        minW = 0.1 * camera.get(3)
        minH = 0.1 * camera.get(4)
        print('请正对着摄像头...')
        confidence = 150.00
        score = 0
        while True:
            success, img = camera.read()
            if not success:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(int(minW), int(minH)))
            for (x, y, w, h) in faces:
                # 画一个矩形
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 图像预测 https://www.py.cn/jishu/jichu/26805.html
                # predict()函数返回两个元素的数组：第一个元素是所识别 个体的标签，第二个是置信度评分。
                #  图像预测predict函数，返回值一个是id，一个是置信度confidence，置信度值越小越好
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                # 设置匹配指数，200是根据confidence的结果估算出来的
                score = int("{0}".format(round(200 - confidence)))
                # 匹配指数大于等于95即可验证通过人脸
                if score > 95:
                    name = names[idnum]
                else:
                    name = "unknown"
                # cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (230, 250, 100), 1)
                img= self.cv2AddChineseText(img, name, (x, y - 30),(0, 255, 0))
                cv2.putText(img, str(score), (x + 5, y + h - 5), font, 1, (255, 0, 0), 1)
            if  score > 95:
                cv2.imwrite('result/result_image.jpg', img)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(convert_to_Qt_format)

            cv2.waitKey(10)  #重要


            if self.end_recogition :
                break

        camera.release()
        # cv2.destroyAllWindows()


