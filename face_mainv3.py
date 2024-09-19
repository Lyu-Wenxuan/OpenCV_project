# -*- coding: utf-8 -*-

import sys


from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from face_recognition_ui import Ui_MainWindow
from FaceCollectv3 import FaceCapture
from FaceRecognitionv3 import *
from FaceTrain import  *

class FaceRecognitionApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.face_capture = FaceCapture()
        self.face_recoginition = FaceRecogition()

        self.face_capture.change_pixmap_signal.connect(self.update_image)
        self.face_recoginition.change_pixmap_signal.connect(self.update_image)
        self.pushButton.clicked.connect(self.collect_faces)
        self.pushButton_2.clicked.connect(self.train_model)
        self.pushButton_3.clicked.connect(self.recognize_face)
        self.pushButton_4.clicked.connect(self.stop_recognition)
        self.pushButton_5.clicked.connect(self.stop_recognition)
        self.pushButton_5.clicked.connect(QtCore.QCoreApplication.instance().quit)

        self.label.setPixmap(QPixmap("UI_images/renlian1.jpg"))

    def collect_faces(self):
        self.face_recoginition.end_recogition = True
        try :
            face_name = self.lineEdit.text()
        except :
            QMessageBox.information(self, "请输入名字", "请输入采集图片的名字")
            return
        try :
            coll_num = int(self.lineEdit_2.text())
            face_name = self.lineEdit.text()
        except :
            QMessageBox.information(self, "采集数量有误", "采集数量建议在1-200之间")
            return

        if face_name == '' :
            QMessageBox.information(self, "请输入名字", "请输入采集图片的名字")
            return
        if coll_num> 100 or coll_num < 1 :
            QMessageBox.information(self, "采集数量有误", "采集数量建议在1-100之间")
            return
        self.label_2.setText("开始采集,请正对摄像头，稍微变换角度,使系统能框出人脸")
        self.face_capture.get_face(face_name, coll_num)
        self.label_2.setText("采集完成")
        self.label.setPixmap(QPixmap("UI_images/renlian1.jpg"))

    def train_model(self):
        self.label_2.setText("开始训练模型")
        print("")
        path = './Facedata/'

        faces, ids = getImagesAndLabels(path)
        # 开始训练
        #print(ids)
        recognizer.train(faces, np.array(ids))
        xinxi = "{0} faces trained. 模型已保存.".format(len(np.unique(ids)))
        self.label_2.setText(f"{xinxi}")
        # 保存文件
        recognizer.write(r'./Model/trainer.yml')



    def recognize_face(self):
        self.face_recoginition.end_recogition = False
        self.label_2.setText("开始识别，结果保存在result文件夹")
        self.face_recoginition.capture()
    def stop_recognition(self):
        # 这里添加结束识别的代码
        print("结束识别")
        self.face_recoginition.end_recogition = True
        self.label.setPixmap(QPixmap("UI_images/renlian1.jpg"))

    def update_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))



def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    mainWin = FaceRecognitionApp()
    mainWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()