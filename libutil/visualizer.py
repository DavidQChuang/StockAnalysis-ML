import sys
import time
from typing import Callable
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QObject
import numpy as np
import pyqtgraph as pg

class nparraylist:
    def __init__(self, shape=(0,), dtype=float):
        """First item of shape is ingnored, the rest defines the shape"""
        self.shape = shape
        self.data = np.zeros((100,*shape[1:]),dtype=dtype)
        self.capacity = 100
        self.size = 0

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity,*self.data.shape[1:]))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        return self.data[:self.size]

class VApp(QtWidgets.QMainWindow):
    def __init__(self, main_fn: Callable[['VWorker', 'VApp'],None], parent=None):
        super(VApp, self).__init__(parent)

        #### Create Gui Elements ###########
        self.mainbox = QtWidgets.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtWidgets.QVBoxLayout())
        main_layout: QtWidgets.QLayout = self.mainbox.layout() # type: ignore

        # canvas
        self.canvas = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.canvas)

        # labels
        main_layout.addWidget(QtWidgets.QTabWidget())
        
        self.iter_label = QtWidgets.QLabel()
        main_layout.addWidget(self.iter_label)
        self.loss_label = QtWidgets.QLabel()
        main_layout.addWidget(self.loss_label)
        self.err_max_label = QtWidgets.QLabel()
        main_layout.addWidget(self.err_max_label)
        self.err_label = QtWidgets.QLabel()
        main_layout.addWidget(self.err_label)

        # view box on canvas
        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 200, 64))

        #  image plot in view box
        # self.loss_mat_graph = pg.ImageItem(border='w')
        # self.view.addItem(self.loss_mat_graph)

        #  line plot in canvas
        self.price_plot = self.canvas.addPlot(row=0, col=0, colspan=2)
        self.price_graph = self.price_plot.plot(pen='y')
        self.price_plot.setLabel(axis='left', text='Closing Price ($)')
        self.price_plot.setLabel(axis='bottom', text='Date')
        
        #####################
        # Error PLOT
        self.err_plot = self.canvas.addPlot(row=1, col=0)
        self.err_graph = self.err_plot.plot(pen='y')
        self.err_plot.setLabel(axis='left', text='Error ($)')
        self.err_plot.setLabel(axis='bottom', text='Iterations')
        
        #####################
        # Max Error PLOT
        self.err_max_plot = self.canvas.addPlot(row=1, col=1)
        self.err_max_graph = self.err_max_plot.plot(pen='r')
        self.err_max_plot.setLabel(axis='left', text='Max Error ($)')
        self.err_max_plot.setLabel(axis='bottom', text='Iterations')
        
        #####################
        # 

        #####################
        # MAIN EVENT LOOP
        self.worker = VWorker(main_fn, self)
        self.worker.sig_iter.connect(self.update_iter)
        self.worker.sig_loss.connect(self.update_loss)
        self.worker.sig_err_max.connect(self.update_err_max)
        self.worker.sig_err.connect(self.update_err)
        
        self.button = QtWidgets.QPushButton("Start")
        self.button.clicked.connect(self.worker.start)
        
        main_layout.addWidget(self.button)

        #### Set Data  #####################

        self.err_data = nparraylist()
        self.err_max_data = nparraylist()
        self.loss_mat_data = np.meshgrid(np.arange(100), np.arange(16))

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        # if main_fn != None:
            # main_fn(self)
        # self._update()
        
    @QtCore.pyqtSlot(int)
    def update_iter(self, iter):
        self.iter_label.setText("Iteration %d" % iter)
        
        if iter == 0:
            self.loss_data = nparraylist()
        
    @QtCore.pyqtSlot(float)
    def update_loss(self, loss):
        self.loss_label.setText("Loss: " + str(round(loss, 2)))
        
    @QtCore.pyqtSlot(float)
    def update_err_max(self, err_max):
        self.err_max_data.add(err_max)
        self.err_max_graph.setData(self.err_max_data.finalize())
        
        self.err_max_label.setText("Error max($): " + str(round(err_max, 2)))
        
    @QtCore.pyqtSlot(float)
    def update_err(self, err):
        self.err_data.add(err)
        self.err_graph.setData(self.err_data.finalize())
        
        self.err_label.setText("Error($): " + str(round(err, 2)))
            # self.loss_mat_graph.setData(self.loss_mat_data)
    
class VWorker(QtCore.QThread):
    sig_iter = QtCore.pyqtSignal(int)
    sig_loss = QtCore.pyqtSignal(float)
    sig_err_max = QtCore.pyqtSignal(float)
    sig_err = QtCore.pyqtSignal(float)
    
    def __init__(self, main_fn: Callable[['VWorker','VApp'],None], app: VApp, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.main_fn = main_fn
        self.app = app
    
    def run(self):
        self.main_fn(self, self.app)
        
    def iter_callback(self):
        def callback(iter, x, y, y_hat, loss, err_max, err):
            if iter % 5 == 0:
                self.sig_iter.emit(iter)
                self.sig_loss.emit(loss)
                self.sig_err_max.emit(err_max)
                self.sig_err.emit(err)
        return callback
    
    def valid_callback(self):
        def callback(iter, output, loss, err_max, err):
            if iter % 5 == 0:
                self.sig_iter.emit(iter)
                self.sig_loss.emit(loss)
                self.sig_err_max.emit(err_max)
                self.sig_err.emit(err)
        return callback
        
def run_app(main_fn):
    qapp = QtWidgets.QApplication(sys.argv)
    app = VApp(main_fn=main_fn)
    app.show()
    exit(qapp.exec())
    
import torch
import inspect
def visualize_module(module: torch.nn.Module):
    fields = inspect.getmembers(module,
        lambda x: isinstance(x, torch.nn.Module))
    
    print(fields)
        
if __name__ == "__main__":
    run_app(None)