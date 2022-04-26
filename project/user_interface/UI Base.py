import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QToolBar, QWidget, QHBoxLayout, QPushButton, QVBoxLayout

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

        x = QHBoxLayout()  # new
        b1 = QPushButton("Test1")  # new
        b2 = QPushButton("Test2")  # new
        x.addWidget(b1)  # new

        self.setCentralWidget(Widget.button(self) )

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

    def __init__(self, parent=Window):
        """Initializer.
        """
        super().__init__(parent)


    def button(self):
        QPushButton('klik')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

