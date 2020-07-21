import sys, cv2
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui, QtWidgets, QtCore
import ProjectUi
import ImageProcess


class UiMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.ui = ProjectUi.Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer_camera = QtCore.QTimer()    # 计时器用来固定时长取帧
        self.timer_getpic = QtCore.QTimer()    # 计时器用来固定时长取识别的图片
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.slot_init()
        self.flag_show_detected = 0
        self.detected = 0    # 是否识别完毕
        self.image = None    # 读取到的帧
        self.detected_pic = None    # 识别后的图片
        self.result_pics = None    # 识别到的每个单塑件图片list
        self.result_infos = None    # 识别到的每个单塑件信息list [中心点坐标，倾斜角度，类型，区别度]

    def slot_init(self):
        self.ui.button_camera.clicked.connect(self.button_display_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_getpic.timeout.connect(self.process_pic)
        self.ui.button_show_detected.clicked.connect(self.show_detected)
        self.ui.button_exit.clicked.connect(self.close)

    def button_display_camera_click(self):
        if not self.timer_camera.isActive():
            # flag = self.cap.open(self.CAM_NUM)

            # ***************测试使用**********************
            flag = self.cap.open("./Object.avi")

            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # self.show_camera()
                self.timer_camera.start(30)  # 每30ms取一次图像
                self.timer_getpic.start(2000)
                self.ui.button_camera.setText(u'关闭检测')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.ui.camera.clear()
            self.ui.button_camera.setText(u'启动检测')
            self.ui.button_show_detected.setEnabled(False)
            self.ui.camera.setText("未开启")

    def show_camera(self):
        flag, self.image = self.cap.read()

        # ***************测试使用**********************
        if self.image is None:
            self.cap.open("./Object.avi")
            flag, self.image = self.cap.read()
        # 识别完后允许点击显示识别后按钮
        if self.detected:
            self.ui.button_show_detected.setEnabled(True)  # 允许点击识别按钮
        else:
            self.ui.button_show_detected.setEnabled(False)

        if self.flag_show_detected:
            show_image = self.detected_pic
        else:
            show_image = self.image
        if self.detected:
            three_pic_label = [self.ui.dict_1, self.ui.dict_2, self.ui.dict_3, self.ui.dict_4]     # 把三个label放进list便于遍历
            three_info_label = [self.ui.dict_1_info, self.ui.dict_2_info, self.ui.dict_3_info, self.ui.dict_4_info]
            for i, j in enumerate(self.result_pics):
                dict_pic = cv2.resize(j, (204, 114))
                cv2.cvtColor(dict_pic, cv2.COLOR_BGR2RGB)
                show_pic = QtGui.QImage(dict_pic.data, dict_pic.shape[1], dict_pic.shape[0], QtGui.QImage.Format_RGB888)
                three_pic_label[i].setPixmap(QtGui.QPixmap.fromImage(show_pic))
            for i in range(0, 4):
                three_info_label[i].setText(f"position:({self.result_infos[i][0][0]}, {self.result_infos[i-1][0][1]})\n"
                                            f"angle:{self.result_infos[i][1]}\n"
                                            f"shape type:{self.result_infos[i][2]}\n"
                                            f"distinction:{self.result_infos[i][3]}")

        show = cv2.resize(show_image, (800, 600), 0, 0, cv2.INTER_LINEAR)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def show_detected(self):
        if self.flag_show_detected:
            self.flag_show_detected = 0
            self.ui.button_show_detected.setText("识别图像")
        else:
            self.flag_show_detected = 1
            self.ui.button_show_detected.setText("未识别图像")

    def process_pic(self):
        ip = ImageProcess.detect(self.image)
        cont = ip.get_three_cont()
        self.result_pics, self.result_infos = ip.get_pic_info(cont)
        self.detected, self.detected_pic = ip.get_classified_pic()
        # self.detected = 1

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = UiMainWindow()
    win.show()

    sys.exit(app.exec_())
