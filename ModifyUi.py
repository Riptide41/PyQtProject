# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ModifyUi_Horizon.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_widget(object):
    def setupUi(self, widget):
        widget.setObjectName("widget")
        widget.setEnabled(True)
        widget.resize(841, 197)
        self.horizontalLayout = QtWidgets.QHBoxLayout(widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(widget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.max_high_limit = QtWidgets.QLabel(self.frame)
        self.max_high_limit.setGeometry(QtCore.QRect(750, 130, 41, 22))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.max_high_limit.setFont(font)
        self.max_high_limit.setObjectName("max_high_limit")
        self.min_low_limit = QtWidgets.QLabel(self.frame)
        self.min_low_limit.setGeometry(QtCore.QRect(60, 60, 31, 22))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.min_low_limit.setFont(font)
        self.min_low_limit.setObjectName("min_low_limit")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(80, 30, 664, 126))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.low_limit_value = QtWidgets.QLabel(self.layoutWidget)
        self.low_limit_value.setAlignment(QtCore.Qt.AlignCenter)
        self.low_limit_value.setObjectName("low_limit_value")
        self.verticalLayout.addWidget(self.low_limit_value)
        self.low_limit_slider = QtWidgets.QSlider(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(45)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.low_limit_slider.sizePolicy().hasHeightForWidth())
        self.low_limit_slider.setSizePolicy(sizePolicy)
        self.low_limit_slider.setMinimumSize(QtCore.QSize(660, 0))
        self.low_limit_slider.setMaximumSize(QtCore.QSize(662, 16777215))
        self.low_limit_slider.setMinimum(50)
        self.low_limit_slider.setMaximum(20000)
        self.low_limit_slider.setProperty("value", 10000)
        self.low_limit_slider.setOrientation(QtCore.Qt.Horizontal)
        self.low_limit_slider.setObjectName("low_limit_slider")
        self.verticalLayout.addWidget(self.low_limit_slider)
        spacerItem = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.high_limit_value = QtWidgets.QLabel(self.layoutWidget)
        self.high_limit_value.setAlignment(QtCore.Qt.AlignCenter)
        self.high_limit_value.setObjectName("high_limit_value")
        self.verticalLayout.addWidget(self.high_limit_value)
        self.high_limit_slider = QtWidgets.QSlider(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.high_limit_slider.sizePolicy().hasHeightForWidth())
        self.high_limit_slider.setSizePolicy(sizePolicy)
        self.high_limit_slider.setMinimumSize(QtCore.QSize(662, 0))
        self.high_limit_slider.setMaximumSize(QtCore.QSize(662, 16777215))
        self.high_limit_slider.setMaximum(500000)
        self.high_limit_slider.setOrientation(QtCore.Qt.Horizontal)
        self.high_limit_slider.setObjectName("high_limit_slider")
        self.verticalLayout.addWidget(self.high_limit_slider)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "Change Square"))
        self.max_high_limit.setText(_translate("widget", "200000"))
        self.min_low_limit.setText(_translate("widget", "50"))
        self.low_limit_value.setText(_translate("widget", "TextLabel"))
        self.high_limit_value.setText(_translate("widget", "TextLabel"))
