import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import *


FRAME_IR = 9
FRAME_DISTANCE = 14
FEATURE_DIM_IR = 4
FEATURE_DIM_SONIC = 24
LABEL_NUM = 5
BATCH = 5

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
            nn.Linear(FRAME_DISTANCE*2, 16),
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
    


    def forward(self, ir_data, distance_data):
        ir_data = ir_data.view(-1, 64)
        distance_data = distance_data.view(-1, FRAME_DISTANCE*2)

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