# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ProjectUi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(809, 820)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(81)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(809, 820))
        MainWindow.setMaximumSize(QtCore.QSize(809, 820))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(10, -1, 10, -1)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.button_camera = QtWidgets.QPushButton(self.centralwidget)
        self.button_camera.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.button_camera.sizePolicy().hasHeightForWidth())
        self.button_camera.setSizePolicy(sizePolicy)
        self.button_camera.setMinimumSize(QtCore.QSize(0, 50))
        self.button_camera.setObjectName("button_camera")
        self.verticalLayout.addWidget(self.button_camera)
        self.button_show_detected = QtWidgets.QPushButton(self.centralwidget)
        self.button_show_detected.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.button_show_detected.sizePolicy().hasHeightForWidth())
        self.button_show_detected.setSizePolicy(sizePolicy)
        self.button_show_detected.setMinimumSize(QtCore.QSize(0, 50))
        self.button_show_detected.setObjectName("button_show_detected")
        self.verticalLayout.addWidget(self.button_show_detected)
        self.button_exit = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(50)
        sizePolicy.setHeightForWidth(self.button_exit.sizePolicy().hasHeightForWidth())
        self.button_exit.setSizePolicy(sizePolicy)
        self.button_exit.setMinimumSize(QtCore.QSize(0, 50))
        self.button_exit.setObjectName("button_exit")
        self.verticalLayout.addWidget(self.button_exit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.camera = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera.sizePolicy().hasHeightForWidth())
        self.camera.setSizePolicy(sizePolicy)
        self.camera.setMinimumSize(QtCore.QSize(640, 480))
        font = QtGui.QFont()
        font.setPointSize(40)
        self.camera.setFont(font)
        self.camera.setAlignment(QtCore.Qt.AlignCenter)
        self.camera.setObjectName("camera")
        self.verticalLayout_2.addWidget(self.camera)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.dict_1 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dict_1.sizePolicy().hasHeightForWidth())
        self.dict_1.setSizePolicy(sizePolicy)
        self.dict_1.setMinimumSize(QtCore.QSize(204, 114))
        self.dict_1.setObjectName("dict_1")
        self.verticalLayout_5.addWidget(self.dict_1)
        self.dict_1_info = QtWidgets.QLabel(self.centralwidget)
        self.dict_1_info.setMinimumSize(QtCore.QSize(204, 114))
        self.dict_1_info.setObjectName("dict_1_info")
        self.verticalLayout_5.addWidget(self.dict_1_info)
        self.horizontalLayout_5.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.dict_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dict_2.sizePolicy().hasHeightForWidth())
        self.dict_2.setSizePolicy(sizePolicy)
        self.dict_2.setMinimumSize(QtCore.QSize(204, 114))
        self.dict_2.setObjectName("dict_2")
        self.verticalLayout_6.addWidget(self.dict_2)
        self.dict_2_info = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dict_2_info.sizePolicy().hasHeightForWidth())
        self.dict_2_info.setSizePolicy(sizePolicy)
        self.dict_2_info.setMinimumSize(QtCore.QSize(204, 114))
        self.dict_2_info.setObjectName("dict_2_info")
        self.verticalLayout_6.addWidget(self.dict_2_info)
        self.horizontalLayout_5.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.dict_3_info = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dict_3_info.sizePolicy().hasHeightForWidth())
        self.dict_3_info.setSizePolicy(sizePolicy)
        self.dict_3_info.setMinimumSize(QtCore.QSize(204, 114))
        self.dict_3_info.setObjectName("dict_3_info")
        self.verticalLayout_7.addWidget(self.dict_3_info)
        self.dict_3 = QtWidgets.QLabel(self.centralwidget)
        self.dict_3.setMinimumSize(QtCore.QSize(204, 114))
        self.dict_3.setObjectName("dict_3")
        self.verticalLayout_7.addWidget(self.dict_3)
        self.horizontalLayout_5.addLayout(self.verticalLayout_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 809, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_camera.setText(_translate("MainWindow", "打开摄像头"))
        self.button_show_detected.setText(_translate("MainWindow", "塑件识别"))
        self.button_exit.setText(_translate("MainWindow", "关闭"))
        self.camera.setText(_translate("MainWindow", "未开启"))
        self.label.setText(_translate("MainWindow", "实时图像"))
        self.dict_1.setText(_translate("MainWindow", "TextLabel"))
        self.dict_1_info.setText(_translate("MainWindow", "TextLabel"))
        self.dict_2.setText(_translate("MainWindow", "TextLabel"))
        self.dict_2_info.setText(_translate("MainWindow", "TextLabel"))
        self.dict_3_info.setText(_translate("MainWindow", "TextLabel"))
        self.dict_3.setText(_translate("MainWindow", "TextLabel"))

