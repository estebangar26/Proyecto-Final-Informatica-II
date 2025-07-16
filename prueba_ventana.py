from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import sys

app = QApplication(sys.argv)
ventana = QDialog()
loadUi("VentanaLogin_fondo_local.ui", ventana)
ventana.show()
sys.exit(app.exec_())
