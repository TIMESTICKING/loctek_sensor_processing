#
# Author Jiabao Li
#
# Created on Mon May 27 2024
#
# Copyright (c) 2024 Loctek
#


import torch
from model import mydevice



def save_params_to_txt(model_path, output_file, pos,op_declares:bool=False):
    # 加载模型参数
    # model_params = torch.load(model_path, mydevice)
    model_params = torch.load(model_path, torch.device("cpu"))
    
    declares = ''
    # 打开文件准备写入
    with open(output_file, 'w') as f:
        # 遍历模型的每个参数
        for name, param in model_params.items():
            if 'weight' in name:
                param = param.T
            if param.ndim != 2:
                param = param.reshape(1, -1)
            var_name = str(name).replace('.', '')
            definition = f'const Eigen::Matrix<float, {param.shape[0]}, {param.shape[1]}> {pos}_{var_name} '
            declare = f'const Eigen::Matrix<float, {param.shape[0]}, {param.shape[1]}> * p_{var_name}; \n'
            declares += declare
            # 将参数的张量转换为numpy数组，然后转换为列表
            # param_data = param.cpu().numpy().tolist()
            param_data = param.numpy().tolist()
            res = str(param_data).replace('[', '{').replace(']', '}')
            # 格式化为C语言中的数组形式
            formatted_param = f"{definition}\n{res};\n\n\n"
            # 写入到文件
            f.write(formatted_param)

        if op_declares:
            f.write(declares)

        # declares = ''
        # for name, param in model_params.items():
        #     if 'weight' in name:
        #         param = param.T
        #     if param.ndim != 2:
        #         param = param.reshape(1, -1)
        #     var_name = str(name).replace('.', '')
        #     declare = f'Eigen::Matrix<float, {param.shape[0]}, {param.shape[1]}> {var_name} ;\n'
        #     declares += declare
        #     # 将参数的张量转换为numpy数组，然后转换为列表
        #     param_data = param.numpy().reshape(-1).tolist()
        #     res = str(param_data).replace('[', '').replace(']', '')
        #     # 格式化为C语言中的数组形式
        #     formatted_param = f"{var_name} << \n{res};\n\n\n"
        #     # 写入到文件
        #     f.write(formatted_param)
        

        # f.write(declares)


if __name__ == '__main__':


    # 使用示例
    model_path = 'models\checkpoints\low\AllData_6_12_low_balanced_1.pth'  # 模型文件路径
    output_file = f'{model_path[:-3]}txt'  # 输出文件路径
    save_params_to_txt(model_path, output_file, 'low')


