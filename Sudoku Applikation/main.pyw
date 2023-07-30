# main.pyw
import sys
import cv2
import style
import numpy as np
import imutils
from PyQt5.QtCore import QSize, Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QInputDialog
from ui.main_ui import Ui_main_ui

from sudoku_solver import SudokuSolver

# 4k display with high dpi resolution
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class Main(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.ui = Ui_main_ui()
        self.ui.setupUi(self)
        self.setWindowTitle('Sudoku Solver mit OpenCV und PyQt5')
        # self.move(-5, 0)
        self.resize(1900, 650)
        self.setMinimumSize(QSize(100, 50))

        self.solver = SudokuSolver('img/sudoku1.jpeg')
        self.solver.state_update_callback = self.update_status
        self.solver_running = False

        self.frame = None
        self.pixmap = None

        self.cap = None
        self.connected = False

        # Setup settings widget
        self.ui.settings.solver = self.solver
        self.ui.settings.get_values_from_instance()
        self.ui.settings.set_callback_after_update(self.run_solver)
        self.ui.settings.ui.cb_detail.clicked.connect(self.update_ui)
        self.ui.settings.ui.btn_solve.clicked.connect(self.run_solver)

        # connect ui events to methods
        self.ui.tabWidget.tabBarClicked.connect(self.update_ui)
        self.ui.actionSource_image.triggered.connect(self.set_src_image)
        self.ui.actionSource_webcam.triggered.connect(self.set_src_webcam)
        self.ui.actionSource_ipcam.triggered.connect(self.set_src_ipcam)
        self.ui.actionVisualize_solver.triggered.connect(self.visualize_solver)
        self.ui.statusBar.showMessage('Sudoku Solver Status: INIT')

    def set_src_image(self):
        """
        Lets user choose new img in FileDialog and gives path to solver
        :return:
        """
        file_name, ok = QFileDialog.getOpenFileUrl()
        file_name = file_name.toString()
        file_name = file_name[8:]
        if ok:
            self.solver.set_source(file_name, 'image')
            self.run_solver()

    def set_src_webcam(self):
        """
        Lets user choose device webcam as source
        :return:
        """
        num, ok = QInputDialog.getInt(self, 'Lokale Webcam', 'Wähle eine Webcam:')
        if ok:
            self.solver.set_source(num, 'webcam')
            self.run_solver()

    def set_src_ipcam(self):
        """
        Lets user input URL to IP Webcam stream
        :return:
        """
        url, ok = QInputDialog.getText(self, 'IP Webcam', 'URL eingeben:')
        if ok:
            self.solver.set_source(str(url), 'ipcam')
            self.run_solver()

    def run_solver(self):
        """
        Run the solver in separate thread
        :return:
        """
        if not self.solver_running:
            self.solver_running = True
            solver_thread = Thread(self, self.solver.run, self.solver_callback)
            solver_thread.start()

    def visualize_solver(self):
        """
        Run solver with visualisation of the solving algorithm
        :return:
        """
        self.solver.visualize_solver = True
        self.run_solver()

    def solver_callback(self):
        """
        Callback method gets run after solver completed
        :return:
        """
        self.solver_running = False
        self.update_ui()
        if self.solver.src_type == 'webcam':
            self.run_solver()

    def update_ui(self):
        """
        Shows the images generated by the solver in the UI
        :return:
        """
        if self.ui.settings.ui.cb_detail.isChecked():
            self.show_img_on_tab([self.solver.img, self.solver.gray, self.solver.th_adaptive], self.ui.pixmap_1)
            self.show_img_on_tab([self.solver.img_conts, self.solver.img_corners, self.solver.img_angle_corr], self.ui.pixmap_2)
            self.show_img_on_tab([self.solver.th_angle_corr, self.solver.boxes_stacked], self.ui.pixmap_3)
            self.show_img_on_tab([self.solver.img, self.solver.virtual_grid], self.ui.pixmap_4)
            self.show_img_on_tab([self.solver.img, self.solver.ar_grid], self.ui.pixmap_5)
        else:
            self.show_img_on_tab(self.solver.img, self.ui.pixmap_1)
            self.show_img_on_tab(self.solver.img_angle_corr, self.ui.pixmap_2)
            self.show_img_on_tab(self.solver.boxes_stacked, self.ui.pixmap_3)
            self.show_img_on_tab(self.solver.virtual_grid, self.ui.pixmap_4)
            self.show_img_on_tab(self.solver.ar_grid, self.ui.pixmap_5)

    def show_img_on_tab(self, img, tab):  # img can be grayscale, BGR or list of imgs
        """
        Generic Mmethod for showing image or images on UI tab
        :param img: Grayscale image (2D nparray), BGR image (3D nparray) or list of images
        :param tab: Destination tab of UI
        :return: 0 - SUCCESS
        """
        if type(img) == list:  # if method gets a list of images, they will be stacked horizontally
            img = self.stack_imgs(img)
        elif len(img.shape) == 2:  # all images assumed to be BGR to be shown in pixmap
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        qformat = QImage.Format_Indexed8  # single color image
        if len(img.shape) == 3:  # rows[0],cols[1],channels[2]
            if (img.shape[2]) == 4:  # alfa channel
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        image = image.rgbSwapped()  # opencv works by default in a bgr colorspace
        pixmap = QPixmap.fromImage(image)

        if self.ui.actionAutoskalierung.isChecked():
            pixmap_resized = pixmap.scaled(tab.width(), tab.height(), Qt.KeepAspectRatio)
            tab.setPixmap(pixmap_resized)
        else:
            tab.setPixmap(pixmap)
        return 0

    @staticmethod
    def stack_imgs(imgs):
        """
        Method stacks any list of Images horizontally
        :param imgs: list of Grayscale or BGR images (nparrays)
        :return: new BGR image of stacked input
        """
        # all images mus be the same color format to be stacked
        imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in imgs]
        heights = [img.shape[1] for img in imgs]
        min_height = sorted(heights)[-1]
        imgs = [imutils.resize(img, height=min_height) for img in imgs]  # resize all images to same height

        img = np.hstack(imgs)
        return img

    def on_cb_changed(self):
        """Checkbox Event Detailansicht"""
        self.update_ui()

    def resizeEvent(self, *args):
        """Event bei einer Grössenänderung des Fensters"""
        super().resizeEvent(*args)
        self.update_ui()

    def keyPressEvent(self, event):
        """Event Erfassung und Auswertung der gedrückten Tasten"""
        super().keyPressEvent(event)
        # print(event.key())
        if event.key() == Qt.Key_Q:
            self.close()

    def on_main_started(self):
        """Aufruf der Funktion erfolgt nach vollständiger Initialisierung."""
        self.run_solver()

    def update_status(self):
        state = self.solver.state
        state_str = 'Sudoku Solver Status: ' + state
        self.ui.statusBar.showMessage(state_str)

        if state == 'INIT':
            self.ui.settings.ui.progressBar.setValue(0)
        elif state == 'LOADING IMG':
            self.ui.settings.ui.progressBar.setValue(10)
        elif state == 'FINDING CONTOURS':
            self.ui.settings.ui.progressBar.setValue(20)
        elif state == 'APPROXIMATE GRID':
            self.ui.settings.ui.progressBar.setValue(30)
        elif state == 'SORTING CORNERS':
            self.ui.settings.ui.progressBar.setValue(40)
        elif state == 'PERSPECTIVE TRANSFORM':
            self.ui.settings.ui.progressBar.setValue(50)
        elif state == 'GUESSING NUMBERS':
            self.ui.settings.ui.progressBar.setValue(60)
        elif state == 'CHECKING SUDOKU SCAN':
            self.ui.settings.ui.progressBar.setValue(70)
        elif state == 'CALCULATING SOLUTION':
            self.ui.settings.ui.progressBar.setValue(80)
        elif state == 'VISUALIZING SOLUTION':
            self.ui.settings.ui.progressBar.setValue(90)
        elif state == 'DONE':
            self.ui.settings.ui.progressBar.setValue(100)


def except_hook(cls, exception, traceback):
    """Fehlerausgabe in der Python-Konsole anstelle des Terminals."""
    sys.__excepthook__(cls, exception, traceback)


class Thread(QThread):
    signal_counter = pyqtSignal(int)

    def __init__(self, parent=None, function=None, callback=None):
        super().__init__(parent)
        self.function = function
        self.callback = callback

    def run(self):
        self.function()
        if self.callback:
            self.callback()
        print('[MAIN] terminateing thread...')

    def start_thread(self):
        print('[MAIN] starting thread...')
        self.start(QThread.NormalPriority)

    def stop(self):
        print('[MAIN] stopping thread...')
        self.terminate()


if __name__ == '__main__':
    sys.excepthook = except_hook
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('assets/icon/abbts.ico'))
    style.set_style(app)
    main = Main()
    main.show()
    t = QTimer()
    t.singleShot(100, main.on_main_started)
    sys.exit(app.exec_())