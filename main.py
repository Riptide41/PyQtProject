import os
import sys
import threading
import time
import random

import cv2
from PyQt5 import QtGui, QtWidgets, QtCore, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

import ImageProcess
import MainWindowUi
from StackPage import ModifyBarPage, PicInfoPage


class UiMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.ui = MainWindowUi.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("./ICON/Icon.jpg"))
        self.timer_camera = QtCore.QTimer()  # 计时器用来固定时长取帧
        self.timer_get_pic = QtCore.QTimer()  # 计时器用来固定时长取识别的图片
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.flag_show_detected = 0  # 是否显示识别后的整图
        self.detected_flag = 0  # 是否识别完毕
        self.image = None  # 读取到的帧
        self.detected_pic = None  # 识别后的图片
        self.result_infos_pics = None  # 识别到的每个单塑件信息list [中心点坐标，倾斜角度，类型，区别度，图片]
        self.detect_min_square = 50
        self.detect_max_square = 20000
        self.display_page_flag = 0
        self.updated_result_flag = False

        self.switch_display_widget = Qt.QStackedLayout(self.ui.switch_frame)
        self.modify_bar_page = ModifyBarPage(self.detect_min_square, self.detect_max_square)
        self.pic_info_page = PicInfoPage()
        self.switch_display_widget.addWidget(self.pic_info_page)
        self.switch_display_widget.addWidget(self.modify_bar_page)
        self.switch_display_widget.setCurrentIndex(self.display_page_flag)

        # 把显示检测结果的四个label放进list便于之后遍历更改
        self.three_pic_label = [self.pic_info_page.ui.dict_1, self.pic_info_page.ui.dict_2,
                                self.pic_info_page.ui.dict_3, self.pic_info_page.ui.dict_4]
        self.three_info_label = [self.pic_info_page.ui.dict_1_info, self.pic_info_page.ui.dict_2_info,
                                 self.pic_info_page.ui.dict_3_info, self.pic_info_page.ui.dict_4_info]
        # **********************
        self.refresh_flag = 0

        self.slot_init()
        self.detecter = ImageProcess.Detect()

    # 连接UI中事件与运行的函数
    def slot_init(self):
        self.ui.button_camera.clicked.connect(self.button_display_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)

        # self.timer_get_pic.timeout.connect(self.thread_process_pic)
        self.timer_get_pic.timeout.connect(self.process_pic)
        self.ui.button_show_detected.clicked.connect(self.show_detected)
        self.ui.button_exit.clicked.connect(self.close)
        self.ui.button_modify.clicked.connect(self.switch_content)

        self.modify_bar_page.ui.low_limit_slider.valueChanged.connect(self.low_limit_slider_change)
        self.modify_bar_page.ui.high_limit_slider.valueChanged.connect(self.high_limit_slide_change)

    # 面积检测下限滑动条事件函数
    def low_limit_slider_change(self):
        slider = self.modify_bar_page.ui.low_limit_slider
        self.detect_min_square = slider.value()
        self.modify_bar_page.ui.low_limit_value.setText(str(slider.value()))
        self.modify_bar_page.ui.high_limit_slider.setMinimum(slider.value())

    # 面积检测上限滑动条事件函数
    def high_limit_slide_change(self):
        slider = self.modify_bar_page.ui.high_limit_slider
        self.detect_max_square = slider.value()
        self.modify_bar_page.ui.high_limit_value.setText(str(slider.value()))
        self.modify_bar_page.ui.low_limit_slider.setMaximum(slider.value())

    # 切换滑动条页面和识别结果页面
    def switch_content(self):
        if self.display_page_flag:
            self.display_page_flag = 0
            self.switch_display_widget.setCurrentIndex(self.display_page_flag)
            self.ui.button_modify.setText(u"Modify\n Area")

        else:
            self.display_page_flag = 1
            self.switch_display_widget.setCurrentIndex(self.display_page_flag)
            self.ui.button_modify.setText(u"Detected\n pieces")

    # 启动摄像头函数
    def button_display_camera_click(self):
        if not self.timer_camera.isActive():
            # flag = self.cap.open(self.CAM_NUM)

            # ***************测试使用**********************
            flag = self.cap.open("./Object.mp4")

            # ##############################################
            # list = os.listdir("C:/Users/WangTC/Documents/Basedcam2 Files/Picture")
            # self.image = cv2.imread("C:/Users/WangTC/Documents/Basedcam2 Files/Picture" + '/' + list[0], 1)
            # os.remove("C:/Users/WangTC/Documents/Basedcam2 Files/Picture" + '/' + list[0])

            if not flag:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.show_camera()
                self.timer_camera.start(30)  # 每30ms取一次图像
                self.timer_get_pic.start(2000)  # 每2s处理一次图像
                self.ui.button_camera.setText(u'Stop Detect')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.ui.camera.clear()
            self.ui.button_camera.setText(u'Start Detect')
            self.ui.button_show_detected.setEnabled(False)
            self.ui.camera.setText("UNOPENED")

    # 在camera中显示图片
    def show_camera(self):
        # flag, self.image = self.cap.read()

        # ***************测试使用**********************
        # self.image = cv2.imread("./test.jpg")

        # if flag is False:
        #     self.cap.open("./Object.mp4")
        #     flag, self.image = self.cap.read()
        # ******************************************
        list = os.listdir("C:/Users/WangTC/Documents/Basedcam2 Files/Picture")
        if len(list):
            time.sleep(0.5)
            self.image = cv2.imread("C:/Users/WangTC/Documents/Basedcam2 Files/Picture" + '/' + list[0], 1)
            os.remove("C:/Users/WangTC/Documents/Basedcam2 Files/Picture" + '/' + list[0])
            self.refresh_flag = 1
        # *******************************************
        self.image = cv2.resize(self.image, (1440, 1080), 0, 0, cv2.INTER_LINEAR)
        # 识别完后允许点击显示识别后按钮
        if self.updated_result_flag:
            self.ui.button_show_detected.setEnabled(True)  # 允许点击识别按钮

        if self.flag_show_detected:
            show_image = self.detected_pic
        else:
            show_image = self.image

        # 更新结果显示区域
        if self.updated_result_flag:
            # 显示结果图片和信息
            for i, j in enumerate(self.result_infos_pics):
                # 显示单塑件图片信息
                if i == 4:
                    break
                self.three_info_label[i].setText(f"position:({j[0][0]}, {j[0][1]})\n"
                                                 f"angle:{'%.2f' % float(j[1])}\n"
                                                 f"shape type:{j[2]}\n"
                                                 f"score:{'%.2f' % float(j[3])}")
                # 显示单塑件图片
                if j[4] is not None:
                    pic = cv2.cvtColor(j[4], cv2.COLOR_BGR2RGB)
                    show_pic = QtGui.QImage(pic.data.tobytes(), pic.shape[1], pic.shape[0], 3 * pic.shape[1],
                                            QtGui.QImage.Format_RGB888)
                    show_pix = QtGui.QPixmap.fromImage(show_pic).scaled(self.three_pic_label[i].size(),
                                                                        QtCore.Qt.KeepAspectRatio)
                    self.three_pic_label[i].setPixmap(show_pix)
                else:
                    self.three_pic_label[i].setText("Unrecognized")
            self.updated_result_flag = False

        # 变换图片的颜色排列方式
        show = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)  # 转换为QImage
        # 变换图片大小
        pixmap = QtGui.QPixmap.fromImage(showImage).scaled(
            self.ui.camera.width() - 20, self.ui.camera.height() - 20, QtCore.Qt.KeepAspectRatio)
        self.ui.camera.setPixmap(pixmap)
        # self.ui.camera.setScaledContents(True)

    def show_detected(self):
        if self.flag_show_detected:
            self.flag_show_detected = 0
            self.ui.button_show_detected.setText("Recognized\n Picture")
            self.ui.button_show_detected.setIcon(QtGui.QIcon(Qt.QPixmap(":/icon/window_icon/切换1.svg")))

        else:
            self.flag_show_detected = 1
            self.ui.button_show_detected.setText("Original\n Picture")
            self.ui.button_show_detected.setIcon(QtGui.QIcon(Qt.QPixmap(":/icon/window_icon/切换开.svg")))

    def thread_process_pic(self):
        t = threading.Thread(target=self.process_pic)
        t.start()

    def process_pic(self):
        start = time.time()
        self.detecter.image_process(self.image, self.detect_min_square, self.detect_max_square)
        self.updated_result_flag, self.result_infos_pics, self.detected_pic = self.detecter.get_four_pic_info()
        end = time.time()
        print("time:", end - start)
        # ********************************
        # if self.refresh_flag:
        #     for i in self.detecter.all_pics:
        #         dict_pic = cv2.resize(i, (204, 114))
        #         dict_pic = cv2.cvtColor(dict_pic, cv2.COLOR_BGR2GRAY)
        #         # print("save pic")
        #         cv2.imwrite(f"./conts/{time.time()+random.random()}.jpg", dict_pic)
        #     self.refresh_flag = 0

    # 重写退出提醒函数，弹出退出询问框
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit?',
                                     "Are you sure?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = UiMainWindow()
    win.show()

    sys.exit(app.exec_())
