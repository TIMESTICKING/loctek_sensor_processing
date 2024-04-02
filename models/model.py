import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import *
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


FRAME_IR = 4
FRAME_DISTANCE = 8
FEATURE_DIM = 4
LABEL_NUM = 5


class CustomDataset(Dataset):
    def __init__(self, IR_dataset, distance_dataset, ground_truth):
        self.IR_dataset = IR_dataset
        self.distance_dataset = distance_dataset
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.IR_dataset)

    def __getitem__(self, idx):
        IR_data = self.IR_dataset[idx]
        distance_data = self.distance_dataset[idx]
        label = self.ground_truth[idx]
        # 将标签转换为one-hot编码
        label_one_hot = torch.zeros(LABEL_NUM)
        label_one_hot[label] = 1
        return IR_data, distance_data, label_one_hot



# 转换为PyTorch张量并缩放数据
def scale_IR(dataset):
    tensor = torch.tensor(dataset, dtype=torch.float32)
    max_val = 35. # tensor.max()
    min_val = 15. # tensor.min()
    scaled_tensor = (tensor - min_val) / (max_val - min_val)
    return scaled_tensor, max_val, min_val



sampled_distance_dataset, sampled_IR_dataset, groudtruth = make_dataset()
# 示例数据转换
distance_tensor = torch.tensor(np.stack(sampled_distance_dataset, axis=0), dtype=torch.float32)
IR_tensor, max_IR, min_IR = scale_IR(np.stack(sampled_IR_dataset, axis=0))

# 神经网络类
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(64, FEATURE_DIM),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(FRAME_DISTANCE*2, 1*FEATURE_DIM),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(1*FEATURE_DIM*(FRAME_IR+1), 1*5),
            nn.Softmax(dim=1)
        )

    def forward(self, ir_data, distance_data):
        ir_data = ir_data.view(-1, 64)
        distance_data = distance_data.view(-1, FRAME_DISTANCE*2)

        ir_output = self.mlp1(ir_data)
        distance_output = self.mlp2(distance_data) # 10x4

        ir_output_per_batch = ir_output.view(-1, FRAME_IR*4) # 10x16
        combined_output = torch.cat((ir_output_per_batch, distance_output), dim=1)

        final_output = self.mlp3(combined_output)
        return final_output


# 创建数据集和数据加载器
dataset = CustomDataset(IR_tensor, distance_tensor, groudtruth)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 实例化网络
net = MyNetwork().cuda()


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# 训练循环示例
for epoch in range(1000):
    running_loss = 0.0
    for i, (IR_data, distance_data, labels) in enumerate(dataloader, 0):
        IR_data = IR_data.cuda()
        distance_data = distance_data.cuda()
        labels = labels.cuda()
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = net(IR_data, distance_data)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

        # 打印统计信息
        running_loss += loss.cpu().item()
        if i % 50 == 49:  # 每10个批次打印一次
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 50:.3f}")
            running_loss = 0.0


# if __name__ == '__main__':
#     make_dataset()
