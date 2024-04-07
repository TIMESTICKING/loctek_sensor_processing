import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import *
from model import *
from comps.IRdataCollect import IR_byte_decoder


def get_label(net: MyMLP, IR_data, distance_data, label=['idle', 'sit', 'sit2stand', 'stand', 'stand2sit']):
    IR_data = IR_data.cuda()
    distance_data = distance_data.cuda()

    outputs = net(IR_data, distance_data)
    outputs = outputs.cpu()

    _, predicted = torch.max(outputs.data, 1)

    return outputs, label[predicted]


def get_action(position, low_net, high_net, *args):
    assert position in [0, 1], 'wrong position type'

    label_raw, label = get_label(low_net if position == 0 else high_net, *args)

    '''
    0: descend the table
    1: no action
    2: rise the table
    '''
    action = 1
    return label, action



if __name__ == '__main__':
    low_net = MyMLP().cuda()
    # low_net.load_state_dict()

    high_net = MyMLP().cuda()
    # high_net.load_state_dict()

    # prepare fake data


    label = get_action(0, low_net, high_net, IR_data, distance_data)



