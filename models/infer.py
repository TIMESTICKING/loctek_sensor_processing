import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import *
from model import *


def get_label(net: MyMLP, IR_data, distance_data, label=['idle', 'sit', 'sit2stand', 'stand', 'stand2sit']):
    IR_data = IR_data.cuda()
    distance_data = distance_data.cuda()

    outputs = net(IR_data, distance_data)
    outputs = outputs.cpu()

    _, predicted = torch.max(outputs.data, 1)

    return outputs, label[predicted]


def get_action():
    pass


if __name__ == '__main__':
    low_net = MyMLP().cuda()
    # low_net.load_state_dict()


