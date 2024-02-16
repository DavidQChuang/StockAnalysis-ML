import signal
import sys, getopt
from PyQt6 import QtCore, QtGui, QtNetwork, QtWidgets
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from sympy import true

QApplication.setApplicationName('foobar')
QApplication.setApplicationVersion('0.1')

class Window(QtWidgets.QLabel):
    def __init__(self, name):
        super(Window, self).__init__()
        self.server = QtNetwork.QLocalServer(self)
        self.server.newConnection.connect(self.handleMessage)
        if not self.server.listen(name):
            print('!!! name: ' + self.server.fullServerName())
            raise RuntimeError(self.server.errorString())

    def closeEvent(self, event):
        self.server.close()
        self.server.removeServer(self.server.fullServerName())

    def handleMessage(self, message=None):
        socket = self.server.nextPendingConnection()
        if socket is not None:
            if socket.waitForReadyRead(2000):
                message = socket.readAll().data().decode('utf-8')
                socket.disconnectFromServer()
            socket.deleteLater()
        if message == 'stop':
            self.close()
        else:
            self.setText(message)

def usage():
    print("""
usage: %s [opts] [message]

options:
 -h  display this help and exit
 -V  display version information
 -s  stop the server
""" % QApplication.applicationName())

def main():
    keys = 'hVs'
    try:
        options, args = getopt.getopt(sys.argv[1:], keys)
    except getopt.GetoptError as exception:
        print('ERROR: %s' % exception)
        usage()
        return 2
    else:
        options = dict(options)
        if '-h' in options:
            usage()
        elif '-V' in options:
            print('%s-%s' % (
                QApplication.applicationName(),
                QApplication.applicationVersion(),
                ))
        else:
            if '-s' in options:
                message = 'stop'
            else:
                message = args[0] if args else None
            name = '%s_server' % QApplication.applicationName()
            
            print("> Connecting server %s" % name)
            socket = QtNetwork.QLocalSocket()
            socket.connectToServer(name, QtCore.QIODeviceBase.OpenModeFlag.WriteOnly)
            
            # if socket.state() == QLocalSocket.LocalSocketState.ConnectedState:
            #     print("The socket is connected to a server.")
            # else:
            #     print("The socket is not connected to a server.")
            
            if socket.waitForConnected(500):
                socket.write(message.encode('utf-8'))
                if not socket.waitForBytesWritten(2000):
                    print('ERROR: could not write to socket: %s' %
                          socket.errorString())
                socket.disconnectFromServer()
            elif socket.error() != QLocalSocket.LocalSocketError.ServerNotFoundError and False:
                print('ERROR: could not connect to server with error of type %s: %s' %
                      (socket.error(), socket.errorString()))
            elif message is not None:
                print('ERROR: server is not running')
            else:
                print("> Creating server %s" % name)
                
                app = QApplication(sys.argv)
                window = Window(name)
                window.setGeometry(50, 50, 200, 30)
                window.show()
                signal.signal(signal.SIGINT, lambda *a: app.quit())
                
                return app.exec()
    return 0

if __name__ == '__main__':

    sys.exit(main())