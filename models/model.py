import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import *

# 转换为PyTorch张量并缩放数据
def to_tensor_and_scale(dataset):
    tensor = torch.tensor(dataset, dtype=torch.float32)
    max_val = tensor.max()
    min_val = tensor.min()
    scaled_tensor = (tensor - min_val) / (max_val - min_val)
    return scaled_tensor, max_val, min_val

sampled_distance_dataset, sampled_IR_dataset, groudtruth = make_dataset()
# 示例数据转换
distance_tensor, _, _ = to_tensor_and_scale(np.stack(sampled_distance_dataset, axis=0))
IR_tensor, _, _ = to_tensor_and_scale(np.concatenate(sampled_IR_dataset, axis=0))

# 神经网络类
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(4*64, 4*4),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(8*2, 1*4),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(4*5, 1*20),
            nn.ReLU(),
            nn.Linear(1*20, 1*5),
            nn.Softmax(dim=1)
        )

    def forward(self, ir_data, distance_data):
        ir_data = ir_data.view(-1, 4*64)
        distance_data = distance_data.view(-1, 8*2)
        ir_output = self.mlp1(ir_data)
        distance_output = self.mlp2(distance_data)
        combined_output = torch.cat((ir_output, distance_output), dim=1)
        combined_output = combined_output.view(-1, 4*5)
        final_output = self.mlp3(combined_output)
        return final_output

# 实例化网络并查看输出
net = MyNetwork()

# 假设我们有一批数据
batch_ir_data = IR_tensor[:10].view(10, 4, 64)
batch_distance_data = distance_tensor[:10].view(10, 8, 2)

# 通过网络传递数据
output = net(batch_ir_data, batch_distance_data)
print(output.shape)  # 应该是 (10, 1, 5)


if __name__ == '__main__':
    make_dataset()
