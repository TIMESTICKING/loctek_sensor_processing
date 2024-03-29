import socket
import os
import sys
sys.path.append('.')
sys.path.append('./comps')
sys.path.append('../comps')
from comps.utils import *
from argparse import ArgumentParser

def main(args):
    # 创建一个套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接到服务器
    client_socket.connect(('localhost', args.port))
    client_socket.send(f'shake-{SOCKET.__getattribute__(SOCKET, args.socket_name)[0]}'.encode())

    while True:
        # 发送消息
        message = client_socket.recv(1024).decode()
        if not message:
            break

        print(message)

    # 关闭套接字
    client_socket.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('socket_name', type=str)
    parser.add_argument('--port', type=int, default=SOCKET.SERVER_PORT)

    args = parser.parse_args()

    main(args)
