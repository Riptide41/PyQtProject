import cv2
import sys, os

from PyQt5 import QtGui, QtWidgets, QtCore, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow
import time

import ImageProcess
import ProjectUi
from StackPage import ModifyBarPage, PicInfoPage


class UiMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(UiMainWindow, self).__init__(parent)
        self.ui = ProjectUi.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("./ICON/Icon.jpg"))
        self.timer_camera = QtCore.QTimer()    # 计时器用来固定时长取帧
        self.timer_get_pic = QtCore.QTimer()    # 计时器用来固定时长取识别的图片
        self.timer_save_origin_pic = QtCore.QTimer()     # 计时器用来定时保存图片
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.flag_show_detected = 0
        self.detected_flag = 0    # 是否识别完毕
        self.image = None    # 读取到的帧
        self.detected_pic = None    # 识别后的图片
        self.result_pics = None    # 识别到的每个单塑件图片list
        self.result_infos = None    # 识别到的每个单塑件信息list [中心点坐标，倾斜角度，类型，区别度]
        self.detect_min_square = 36000
        self.detect_max_square = 100000
        # self.child_window = ChildWindow()
        self.display_page_flag = 0
        self.updated_result_flag = False

        self.switch_display_frame = Qt.QStackedLayout(self.ui.frame)
        self.modify_bar_page = ModifyBarPage(self.detect_min_square, self.detect_max_square)
        self.pic_info_page = PicInfoPage()
        self.switch_display_frame.addWidget(self.pic_info_page)
        self.switch_display_frame.addWidget(self.modify_bar_page)
        self.switch_display_frame.setCurrentIndex(self.display_page_flag)
        # **********************
        self.refresh_flag = 0

        self.slot_init()
        self.detecter = ImageProcess.Detect()

    def slot_init(self):
        self.ui.button_camera.clicked.connect(self.button_display_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_get_pic.timeout.connect(self.process_pic)
        self.timer_save_origin_pic.timeout.connect(self.save_pic)
        self.ui.button_show_detected.clicked.connect(self.show_detected)
        self.ui.button_exit.clicked.connect(self.close)
        self.ui.button_modify.clicked.connect(self.switch_content)

        self.modify_bar_page.ui.low_limit_slider.valueChanged.connect(self.low_limit_slider_change)
        self.modify_bar_page.ui.high_limit_slider.valueChanged.connect(self.high_limit_slide_change)

    def low_limit_slider_change(self):
        slider = self.modify_bar_page.ui.low_limit_slider
        self.detect_min_square = slider.value()
        self.modify_bar_page.ui.low_limit_value.setText(str(slider.value()))
        self.modify_bar_page.ui.high_limit_slider.setMinimum(slider.value())

    def high_limit_slide_change(self):
        slider = self.modify_bar_page.ui.high_limit_slider
        self.detect_max_square = slider.value()
        self.modify_bar_page.ui.high_limit_value.setText(str(slider.value()))
        self.modify_bar_page.ui.low_limit_slider.setMaximum(slider.value())

    def switch_content(self):
        if self.display_page_flag:
            self.display_page_flag = 0
            self.switch_display_frame.setCurrentIndex(self.display_page_flag)
            self.ui.button_modify.setText(u"更改面积")
        else:
            self.display_page_flag = 1
            self.switch_display_frame.setCurrentIndex(self.display_page_flag)
            self.ui.button_modify.setText(u"检测结果")

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
                self.timer_get_pic.start(2000)
                self.timer_save_origin_pic.start(1000)    # 每1s保存一次图像
                self.ui.button_camera.setText(u'关闭检测')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.ui.camera.clear()
            self.ui.button_camera.setText(u'启动检测')
            self.ui.button_show_detected.setEnabled(False)
            self.ui.camera.setText("未开启")

    def save_pic(self):
        pass
        # filename = time.time()
        # cv2.imwrite(f"./Origin_Pic/{filename}.jpg", self.image)
        # print("save success!")

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


            self.image = cv2.resize(self.image, (1623, 1080), 0, 0, cv2.INTER_LINEAR)
        # 识别完后允许点击显示识别后按钮
        if self.updated_result_flag:
            self.ui.button_show_detected.setEnabled(True)  # 允许点击识别按钮


        if self.flag_show_detected:
            show_image = self.detected_pic
        else:
            show_image = self.image

        # 更新结果显示区域
        if self.updated_result_flag:
            three_pic_label = [self.pic_info_page.ui.dict_1, self.pic_info_page.ui.dict_2,
                               self.pic_info_page.ui.dict_3, self.pic_info_page.ui.dict_4]     # 把三个label放进list便于遍历
            three_info_label = [self.pic_info_page.ui.dict_1_info, self.pic_info_page.ui.dict_2_info,
                                self.pic_info_page.ui.dict_3_info, self.pic_info_page.ui.dict_4_info]
            # try:
            for i, j in enumerate(self.result_pics):
                # self.j = j
                dict_pic = cv2.resize(j, (204, 114))
                cv2.cvtColor(dict_pic, cv2.COLOR_BGR2RGB)
                show_pic = QtGui.QImage(dict_pic.data, dict_pic.shape[1], dict_pic.shape[0],
                                        QtGui.QImage.Format_RGB888)
                three_pic_label[i].setPixmap(QtGui.QPixmap.fromImage(show_pic))
            for j, i in enumerate(self.result_infos):
                three_info_label[j].setText(f"position:({i[0][0]}, {i[0][1]})\n"
                                            f"angle:{i[1]}\n"
                                            f"shape type:{i[2]}\n"
                                            f"distinction:{i[3]}")
            self.updated_result_flag = False

            # except Exception as e:
            #     print("LABEL ERROR:", str(e))
            # for i in range(0, 4):
            #     three_info_label[i].setText(f"position:({self.result_infos[i][0][0]}, {self.result_infos[i-1][0][1]})\n"
            #                                 f"angle:{self.result_infos[i][1]}\n"
            #                                 f"shape type:{self.result_infos[i][2]}\n"
            #                                 f"distinction:{self.result_infos[i][3]}")

        show = cv2.resize(show_image, (640, 480), 0, 0, cv2.INTER_LINEAR)
        # **************测试使用************************
        # show = show[140:740, 200:1000]

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
        self.detecter.image_process(self.image, self.detect_min_square, self.detect_max_square)
        self.updated_result_flag, self.result_pics, self.result_infos, self.detected_pic = self.detecter.get_four_pic_info()
        # ********************************
        # if self.refresh_flag:
        #     for i in self.ip.all_pics:
        #         dict_pic = cv2.resize(i, (204, 114))
        #         dict_pic = cv2.cvtColor(dict_pic, cv2.COLOR_BGR2GRAY)
        #         show_pic = QtGui.QImage(dict_pic.data, dict_pic.shape[1], dict_pic.shape[0],
        #                                 QtGui.QImage.Format_RGB888)
        #         cv2.imshow("ssssda", dict_pic)
        #         cv2.imwrite(f"./conts/{time.time()}.jpg", dict_pic)
        #     self.refresh_flag = 0

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            # sys.exit(0)    # 关闭主窗口时也关闭子窗口


# class ChildWindow(QtWidgets.QWidget):
#     def __init__(self):
#         super(QtWidgets.QWidget, self).__init__()
#         ui = ModifyUi.Ui_widget()
#         ui.setupUi(self)
#
#     def show_window(self):
#         self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
#         self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = UiMainWindow()
    win.show()

    sys.exit(app.exec_())
