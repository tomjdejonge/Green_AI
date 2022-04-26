from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Form.ui', self)
        self.show()

        #knoppen
        self.startknop.clicked.connect(self.startklik)
        self.berekenknop.clicked.connect(self.berekenklik)

    def startklik(self):
        print('start')

    def berekenklik(self):
            print('bereken')


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()