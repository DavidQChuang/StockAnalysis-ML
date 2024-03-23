import sys
import time
from typing import Callable
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QObject
import numpy as np
import pyqtgraph as pg

from datasets.Common import AdvancedTimeSeriesDataset, TimeSeriesDataset

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
    dataset: AdvancedTimeSeriesDataset
    
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
        # self.view = self.canvas.addViewBox()
        # self.view.setAspectLocked(True)
        # self.view.setRange(QtCore.QRectF(0,0, 200, 64))

        #  image plot in view box
        # self.loss_mat_graph = pg.ImageItem(border='w')
        # self.view.addItem(self.loss_mat_graph)

        #  line plot in canvas
        self.price_plot = self.canvas.addPlot(row=0, col=0, colspan=2)
        self.price_graph = self.price_plot.plot(pen='y')
        self.price_pred_graph = self.price_plot.plot(pen='r')
        self.price_real_graph = self.price_plot.plot(pen='g')
        self.price_plot.setLabel(axis='left', text='Close Price ($)')
        
        self.macd_plot = self.canvas.addPlot(row=1, col=0, colspan=2)
        # self.macd_graph = self.macd_plot.plot(pen='b')
        self.macd_graph = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.macd_plot.addItem(self.macd_graph)
        self.macd_plot.setLabel(axis='left', text='MACD')
        self.macd_plot.setLabel(axis='bottom', text='Date')
        
        self.macd_plot.setXLink(self.price_plot)
        
        self.dir_graph = pg.BarGraphItem(x=[], height=[], width=0.6)
        self.macd_plot.addItem(self.dir_graph)
        
        #####################
        # Error PLOT
        self.err_plot = self.canvas.addPlot(row=3, col=0)
        self.err_graph = self.err_plot.plot(pen='y')
        self.err_plot.setLabel(axis='left', text='Error ($)')
        self.err_plot.setLabel(axis='bottom', text='Iterations')
        
        #####################
        # Max Error PLOT
        self.err_max_plot = self.canvas.addPlot(row=3, col=1)
        self.err_max_graph = self.err_max_plot.plot(pen='r')
        self.err_max_plot.setLabel(axis='left', text='Max Error ($)')
        self.err_max_plot.setLabel(axis='bottom', text='Iterations')
        
        #####################
        # 

        #####################
        # MAIN EVENT LOOP
        self.worker = VWorker(main_fn, self)
        self.worker.sig_iter.connect(self.update_iter)
        # self.worker.sig_loss.connect(self.update_loss)
        self.worker.sig_err_max.connect(self.update_err_max)
        self.worker.sig_err.connect(self.update_err)
        self.worker.sig_x.connect(self.update_x)
        self.worker.sig_y.connect(self.update_y)
        self.worker.sig_y_hat.connect(self.update_y_hat)
        
        self.worker.sig_dataset.connect(self.update_dataset)
        
        self.button = QtWidgets.QPushButton("Start")
        self.button.clicked.connect(self.worker.start)
        
        main_layout.addWidget(self.button)

        #### Set Data  #####################

        self.err_data = nparraylist()
        self.err_max_data = nparraylist()
        self.loss_mat_data = np.meshgrid(np.arange(100), np.arange(16))
        self.price_data: np.ndarray = None # type: ignore
        self.price_pred_data: np.ndarray = None # type: ignore
        self.price_real_data: np.ndarray = None # type: ignore

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        # if main_fn != None:
            # main_fn(self)
        # self._update()
        
    #### Initialization slots  #####################
    @QtCore.pyqtSlot(AdvancedTimeSeriesDataset)
    def update_dataset(self, dataset):
        self.dataset = dataset
    
    #### Per-iteration slots  #####################
        
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
            
    @QtCore.pyqtSlot(np.ndarray)
    def update_x(self, x):
        self.update_price_graph(x, self.price_real_data, self.price_pred_data)
        
    @QtCore.pyqtSlot(np.ndarray)
    def update_y(self, y):
        self.update_price_graph(self.price_data, y, self.price_pred_data)
        
    @QtCore.pyqtSlot(np.ndarray)
    def update_y_hat(self, y_hat):
        self.update_price_graph(self.price_data, self.price_real_data, y_hat)
        
    def update_price_graph(self, x, y, y_hat):
        close_idx = self.dataset.column_names.index("close")
        # print(close_idx)
        # raise "x"
        
        price_data_len: int
        last_close_price: float
        if x is not self.price_data:
            self.price_data = x
            
            # get sequence of closing prices in [batch 0, (all rows), feature 0]
            close_data = x[0, -50:, close_idx]
            
            price_data_len = len(close_data)
            last_close_price = x[0, -1, close_idx]
            
            # Render historical prices
            self.price_graph.setData(x=range(price_data_len), y=close_data)
            
            # Render MACD bar graph
            if 'close_macd26' in self.dataset.column_names:
                macd_data = x[0, -50:, self.dataset.column_names.index('close_macd26')]
            
                colors = ['g' if e >= 0 else 'r' for e in macd_data]
                self.macd_graph.setOpts(x=range(price_data_len), height=macd_data, brushes=colors)
        else:
            price_data_len = len(self.price_data[0, -50:])
            last_close_price = self.price_data[0, -1, close_idx]
            
        if y is not self.price_real_data and self.price_data is not None:
            self.price_real_data = y
            
            start_idx = price_data_len - 1
            real_len = len(y[0]) + 1
            
            # Render real future prices line graph
            # (plus last historical price to connect to historical line)
            data = [last_close_price] + list(y[0])
            self.price_real_graph.setData(x=range(start_idx, start_idx+real_len), y=data)
            
        if y_hat is not self.price_pred_data and self.price_data is not None:
            self.price_pred_data = y_hat
            
            start_idx = price_data_len - 1
            pred_len = len(y_hat[0]) + 1
            
            # Render inferred future prices line graph
            # (plus last historical price to connect to historical line)
            data = [last_close_price] + list(y_hat[0])
            self.price_pred_graph.setData(x=range(start_idx, start_idx+pred_len), y=data)
            
            y_data = [last_close_price] + list(y[0])
            y_hat_data = data
            
            # Get the error of the signs of the predicted differences
            # and map 0 (good) to +1 and -1 or -2 (bad) to -1.
            close_diff = np.sign(np.abs(np.sign(np.diff(y_data))
                              - np.sign(np.diff(y_hat_data))))
            close_dir = (close_diff * -2 + 1) * 0.1
            
            # Render directional accuracy bar graph
            colors = [ 'g' if d >= 0 else 'r' for d in close_dir ]
            self.dir_graph.setOpts(x=range(start_idx+1, start_idx+pred_len), height=close_dir, brushes=colors)
    
class VWorker(QtCore.QThread):
    sig_iter = QtCore.pyqtSignal(int)
    sig_loss = QtCore.pyqtSignal(float)
    sig_err_max = QtCore.pyqtSignal(float)
    sig_err = QtCore.pyqtSignal(float)
    sig_x = QtCore.pyqtSignal(np.ndarray)
    sig_y = QtCore.pyqtSignal(np.ndarray)
    sig_y_hat = QtCore.pyqtSignal(np.ndarray)
    
    sig_dataset = QtCore.pyqtSignal(AdvancedTimeSeriesDataset)
    
    def __init__(self, main_fn: Callable[['VWorker','VApp'],None], app: VApp, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.main_fn = main_fn
        self.app = app
    
    def run(self):
        self.main_fn(self, self.app)
        
    def data_callback(self):
        def callback(iter, x: torch.Tensor, y):
            if iter % 10 == 0:
                self.sig_x.emit(x.detach().clone().cpu().numpy())
                self.sig_y.emit(y.detach().clone().cpu().numpy())
                
        return callback
        
    def iter_callback(self):
        def callback(iter, y_hat, err_max, err):
            if iter % 2 == 0:
                self.sig_iter.emit(iter)
                # self.sig_loss.emit(loss)
                self.sig_err_max.emit(err_max)
                self.sig_err.emit(err)
            if iter % 10 == 0:
                self.sig_y_hat.emit(y_hat.detach().clone().cpu().numpy())
                
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