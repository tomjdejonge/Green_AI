import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QToolBar, QWidget, QHBoxLayout, QPushButton

class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle('GreenAI')
        self.central_widget()
        self.createMenu()
        self.createToolBar()
        self.createStatusBar()
        self.setGeometry(500,200,350,400)


    def central_widget(self):
        #self.central_widget = self.setCentralWidget(QLabel("Central Widget"))
        self.central_widget = Widget.addbutton(self)

    def createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Exit', self.close)

    def createToolBar(self):
        tools = QToolBar()
        self.addToolBar(tools)
        tools.addAction('Exit', self.close)

    def createStatusBar(self):
        status = QStatusBar()
        status.showMessage("GreenAI")
        self.setStatusBar(status)

class Widget(QWidget):
    def __init__(self):
        super().__init__(parent)

    def addbutton(self):
        layout = QHBoxLayout()
        layout.addWidget(QPushButton("Left-Most"))
        layout.addWidget(QPushButton("Center"), 1)
        layout.addWidget(QPushButton("Right-Most"), 2)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())