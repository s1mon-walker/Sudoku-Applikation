from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget
from sudoku_solver import SudokuSolver
from .ui.settings_ui import Ui_settings


class Settings(QWidget):
    """Subwidget in welchem die Funtkionskapselung exemplarisch aufgezeigt wird."""
    event_settings = pyqtSignal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.solver: SudokuSolver = None
        self.ui = Ui_settings()
        self.ui.setupUi(self)
        self.value_change_connected = False
        self.ui.cb_detail.setChecked(True)

        self.callback = None

    def on_button_clicked(self):
        obj = self.sender()
        obj_name = obj.objectName()

        self.get_values_from_instance()

    def connect_value_change_event(self):
        self.ui.slider_blur.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_blocksize.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_C.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_th.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_min_num_area.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_erode.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_dilate.valueChanged.connect(self.on_slider_value_changed)
        self.ui.slider_mask_size.valueChanged.connect(self.on_slider_value_changed)
        self.value_change_connected = True

    def disconnect_value_change_event(self):
        if self.value_change_connected:
            self.ui.slider_blur.valueChanged.disconnect()
            self.ui.slider_blocksize.valueChanged.disconnect()
            self.ui.slider_C.valueChanged.disconnect()
            self.ui.slider_th.valueChanged.disconnect()
            self.ui.slider_min_num_area.valueChanged.disconnect()
            self.ui.slider_smin.valueChanged.disconnect()
            self.ui.slider_smax.valueChanged.disconnect()
            self.ui.slider_vmin.valueChanged.disconnect()
            self.ui.slider_vmax.valueChanged.disconnect()
            self.ui.slider_erode.valueChanged.disconnect()
            self.ui.slider_dilate.valueChanged.disconnect()
            self.ui.slider_mask_size.valueChanged.disconnect()
        self.value_change_connected = False

    def on_slider_value_changed(self):
        self.set_values_to_instance()
        if self.callback:
            self.callback()

    def set_values_to_instance(self):
        self.solver.blur_kernel = self.ui.slider_blur.value() * 2 + 1

        self.solver.th_blocksize = self.ui.slider_blocksize.value() * 2 + 1
        self.solver.th_C = self.ui.slider_C.value()

        self.solver.box_th_val = self.ui.slider_th.value()
        self.solver.box_erode_iter = self.ui.slider_erode.value()
        self.solver.box_dilate_iter = self.ui.slider_dilate.value()
        self.solver.mask_size = self.ui.slider_mask_size.value()

        self.solver.min_num_area = self.ui.slider_min_num_area.value()

        self.set_values_to_labels()

    def get_values_from_instance(self):
        self.disconnect_value_change_event()

        self.ui.slider_blur.setValue((self.solver.blur_kernel-1)/2)

        self.ui.slider_blocksize.setValue((self.solver.th_blocksize-1)/2)
        self.ui.slider_C.setValue(self.solver.th_C)

        self.ui.slider_th.setValue(self.solver.box_th_val)
        self.ui.slider_erode.setValue(self.solver.box_erode_iter)
        self.ui.slider_dilate.setValue(self.solver.box_dilate_iter)
        self.ui.slider_mask_size.setValue(self.solver.mask_size)

        self.ui.slider_min_num_area.setValue(self.solver.min_num_area)

        self.connect_value_change_event()

        self.set_values_to_labels()

    def set_values_to_labels(self):
        self.ui.lbl_blur.setText(f'Kernel Grösse: {self.solver.blur_kernel:2d}')

        self.ui.lbl_blocksize.setText(f'Blocksize: {self.solver.th_blocksize:3d}')
        self.ui.lbl_C.setText(f'C: {self.solver.th_C}')

        self.ui.lbl_th.setText(f'Threshold: {self.solver.box_th_val:3d}')
        self.ui.lbl_erode.setText(f'Erode: {self.solver.box_erode_iter:3d}')
        self.ui.lbl_dilate.setText(f'Dilate: {self.solver.box_dilate_iter:3d}')
        self.ui.lbl_mask_size.setText(f'Linienbreite: {self.solver.mask_size:3d}')

        self.ui.lbl_min_area.setText(f'Min. Fläche: {self.solver.min_num_area:3d}')

    def set_callback_after_update(self, callback):
        self.callback = callback

"""
self.event_display.emit('MEASUREMENT_SETTINGS')
self.ui.display.event_display.connect(self.on_event_display)
def on_event_display(self, event):
    if isinstance(event, str):
        self.statusbar.status_message(event)
"""