import torch
from model import mydevice

def save_params_to_txt(model_path, output_file):
    # 加载模型参数
    model_params = torch.load(model_path, mydevice)
    
    # 打开文件准备写入
    with open(output_file, 'w') as f:
        # 遍历模型的每个参数
        for name, param in model_params.items():
            # 将参数的张量转换为numpy数组，然后转换为列表
            param_data = param.numpy().tolist()
            res = str(param_data).replace('[', '{').replace(']', '}')
            # 格式化为C语言中的数组形式
            formatted_param = f"{name}:\n{res}\n\n\n"
            # 写入到文件
            f.write(formatted_param)


if __name__ == '__main__':


    # 使用示例
    model_path = 'models\checkpoints_v2\high\AllData_v2_balanced_0d90.pth'  # 模型文件路径
    output_file = f'{model_path[:-3]}txt'  # 输出文件路径
    save_params_to_txt(model_path, output_file)

