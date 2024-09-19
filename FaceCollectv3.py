import cv2
import numpy as np
import os
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PIL import ImageFont, ImageDraw, Image


class FaceCapture(QObject):
    change_pixmap_signal = pyqtSignal(QImage)

    def cv2AddChineseText(self,frame, name, position, fill):
        font = ImageFont.truetype('simsun.ttc', 30)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, name, font=font, fill=fill)
        return np.array((img_pil))

    def get_max_id(self):
        max_id = -1
        if os.path.exists('id.txt'):
            with open('id.txt', 'r',encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        try:
                            id_ = int(parts[1])
                            if id_ > max_id:
                                max_id = id_
                        except ValueError:
                            pass
        return max_id + 1

    def get_padding_size(self, shape):
        h, w = shape
        longest = max(h, w)
        result = (np.array([longest] * 4, int) - np.array([h, h, w, w], int)) // 2
        return result.tolist()

    def deal_with_image(self, img, h=64, w=64):
        top, bottom, left, right = self.get_padding_size(img.shape[0:2])
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = cv2.resize(img, (h, w))
        return img

    def get_face(self, name, coll_num):
        face_id = self.get_max_id()
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        face_detector = cv2.CascadeClassifier('./cv2data/haarcascade_frontalface_default.xml')
        count = 1
        while count <= coll_num:
            success, img = camera.read()
            if not success:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 8)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = self.deal_with_image(face)
                cv2.imwrite(f"Facedata/User.{face_id}.{count}.jpg", face)
                # cv2.putText(img, name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                img= self.cv2AddChineseText(img, name, (x, y - 30),(0, 255, 0))
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            key = cv2.waitKey(30) & 0xff
            if  key == 27:
                break
        camera.release()
        # cv2.destroyAllWindows()
        with open('id.txt', 'a', encoding='utf-8') as file:
            file.write(f"{name}:{face_id}\n")

# 测试代码
if __name__ == '__main__':
    app = FaceCapture()
    app.get_face("test_name", 10)