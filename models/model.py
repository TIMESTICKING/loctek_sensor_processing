"""
@Description :   
@Author      :   李佳宝
@Time        :   2024/04/24 11:03:05
"""

import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from preprocess import *


FRAME_IR = 9
FRAME_DISTANCE = 14
FEATURE_DIM_IR = 4
FEATURE_DIM_SONIC = 24
LABEL_NUM = 5
BATCH = 5

IR_DIM = 1

mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPFormula:

    IR_encoded = collections.deque(maxlen=FRAME_IR)

    def __init__(self, pth) -> None:
        self.parameters = torch.load(os.path.normpath(pth), map_location=mydevice)
        '''打印某个矩阵'''
        print(self.parameters['mlp1.0.weight'])
        '''把矩阵转为列表，直接拷贝到C++文件？'''
        print(self.parameters['mlp1.0.weight'].tolist())


    def encodeIR(self, IR):
        IR = IR.view(1, 64)
        res = (IR @ self.parameters['mlp1.0.weight'].T) + self.parameters['mlp1.0.bias']
        '''激活函数，max(res, 0)，其实可以在矩阵逐元素运算时完成'''
        res = torch.relu(res)
        res = (res @ self.parameters['mlp1.2.weight'].T) + self.parameters['mlp1.2.bias']
        res = torch.relu(res)

        res = res.squeeze(0)
        MLPFormula.IR_encoded.append(res)

    
    def clear_data_ready(self):
        """call it every time you change the
        """        
        MLPFormula.IR_encoded.clear()


    def is_data_ready(self):
        return len(MLPFormula.IR_encoded) == FRAME_IR


    def __call__(self, ir_encoded, distance_data):
        distance_data = distance_data.view(-1, FRAME_DISTANCE*IR_DIM)
        distance_encoded = (distance_data @ self.parameters['mlp2.0.weight'].T) + self.parameters['mlp2.0.bias']
        '''激活函数，max(res, 0)，其实可以在矩阵逐元素运算时完成'''
        distance_encoded = torch.relu(distance_encoded)
        distance_encoded = (distance_encoded @ self.parameters['mlp2.2.weight'].T) + self.parameters['mlp2.2.bias']
        distance_encoded = torch.relu(distance_encoded)

        '''把编码后的IR数据和超声数据拼接起来'''
        combined_output = torch.cat((ir_encoded.view(1, -1), distance_encoded), dim=1)

        label = (combined_output @ self.parameters['mlp3.0.weight'].T) + self.parameters['mlp3.0.bias']
        '''softmax归一化指数函数，exp(xi) / sum[exp(xj)]'''
        label = torch.softmax(label, dim=1)

        return label



# 神经网络类
class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, FEATURE_DIM_IR),
            nn.ReLU(),
            # nn.Linear(16, FEATURE_DIM),
            # nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(FRAME_DISTANCE*IR_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, FEATURE_DIM_SONIC),
            nn.ReLU(),
            # nn.Linear(FEATURE_DIM*2, 1*FEATURE_DIM),
            # nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(FRAME_IR*FEATURE_DIM_IR + FEATURE_DIM_SONIC, LABEL_NUM),
            nn.Softmax(dim=1)
        )
    

    @torch.no_grad()
    def encode_IR(self, ir_data):
        ir_data = ir_data.view(-1, 64)
        ir_output = self.mlp1(ir_data)
        ir_output_one_batch = ir_output.view(1, -1) # 1 x (9x4)

        return ir_output_one_batch

    @torch.no_grad()
    def infer(self, ir_data, distance_data):
        distance_data = distance_data.view(1, -1)
        distance_output = self.mlp2(distance_data) # B x 24
        combined_output = torch.cat((ir_data, distance_output), dim=1)

        final_output = self.mlp3(combined_output)
        return final_output

    def clear_data_ready(self):
        pass


    def forward(self, ir_data, distance_data):
        ir_data = ir_data.view(-1, 64)
        distance_data = distance_data.view(-1, FRAME_DISTANCE*IR_DIM)

        ir_output = self.mlp1(ir_data)
        ir_output_per_batch = ir_output.view(-1, FRAME_IR*FEATURE_DIM_IR) # B x (9x4)

        distance_output = self.mlp2(distance_data) # B x 24
        combined_output = torch.cat((ir_output_per_batch, distance_output), dim=1)

        final_output = self.mlp3(combined_output)
        return final_output



class MyCNN(MyMLP):
    def __init__(self):
        super(MyCNN, self).__init__()

        self.mlp1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        # 自动计算全连接层的输入特征数
        self.fc_input_dim = self._get_conv_output_dim(8, 8)
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, FEATURE_DIM)


    @torch.no_grad()
    def _get_conv_output_dim(self, input_height, input_width):
        # 创建一个假的输入张量以计算卷积层输出的大小
        dummy_input = torch.zeros(1, 1, input_height, input_width)
        output = self.mlp1(dummy_input)
        return int(torch.numel(output) / output.size(0))  # 返回展平后的特征数


    def forward(self, ir_data, distance_data):
        ir_data = ir_data.view(-1, 8, 8).unsqueeze(1)
        distance_data = distance_data.view(-1, FRAME_DISTANCE*2)

        # 卷积层后接ReLU激活函数
        x = F.relu(self.mlp1(x))
        # 展平
        x = x.view(-1, self.fc_input_dim)
        # 全连接层
        ir_output = self.fc1(x)
        ir_output_per_batch = ir_output.view(-1, FRAME_IR*FEATURE_DIM) # 10x16

        distance_output = self.mlp2(distance_data) # 10x4
        combined_output = torch.cat((ir_output_per_batch, distance_output), dim=1)

        final_output = self.mlp3(combined_output)
        return final_output