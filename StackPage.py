import DictPicInfoUi_Vertical, ModifyUi_Vertical
from PyQt5.Qt import QWidget


class PicInfoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = DictPicInfoUi_Vertical.Ui_Form()
        self.ui.setupUi(self)


class ModifyBarPage(QWidget):
    def __init__(self, low_limit, high_limit, parent=None):
        super().__init__(parent)
        self.ui = ModifyUi_Vertical.Ui_Form()
        self.ui.setupUi(self)
        self.ui.high_limit_value.setText(str(high_limit))
        self.ui.low_limit_value.setText(str(low_limit))
        self.ui.high_limit_slider.setMinimum(low_limit)
        self.ui.low_limit_slider.setMaximum(high_limit)
        self.ui.low_limit_slider.setValue(low_limit)
        self.ui.high_limit_slider.setValue(high_limit)


