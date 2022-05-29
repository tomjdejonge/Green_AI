import sys
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QPushButton, QDesktopWidget,
    QLabel, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider
)
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up basic Window
        self.setWindowTitle('Green AI')
        self.resize(640, 480)
        self.setStyleSheet("background-color: lightGray;")
        self.center()

        # Test Button
        test_button = QPushButton('Click me', self)
        test_button.clicked.connect(self.clickMethod)
        test_button.resize(160, 80)
        test_button.move(0, 32)

        # Test Label
        label = QLabel('GREEN AI', self)
        label.resize(500, 80)
        font = label.font()
        font.setPointSize(30)
        label.setFont(font)
        label.setAlignment(Qt.AlignVCenter)


    def clickMethod(self):
        print('Clicked Pyqt button.')

    def center(self):
        frameGm = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()