import socketserver
from comps.utils import *

class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(f"Connected to {self.client_address}")
        try:
            while True:
                message = self.request.recv(1024).decode()
                if not message:
                    break

                if message.startswith('shake'):
                    tag = message.split('-')[1]
                    if tag == SOCKET.sonic1[0]:
                        SOCKET.sonic1[1] = self.request
                        self.my_device_tag = SOCKET.sonic1
                print(f"Received message from {self.client_address}: {message}")
            
        except Exception as e:
            pass
        finally:
            self.my_device_tag[1] = None
            print(f"Connection closed with {self.client_address}")




def start_server():
    server_address = ('localhost', SOCKET.SERVER_PORT)
    with socketserver.ThreadingTCPServer(server_address, RequestHandler) as server:
        print("Server is running...")
        server.serve_forever()

if __name__ == '__main__':
    start_server()
