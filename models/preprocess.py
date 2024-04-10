import os
import numpy as np
import pandas as pd
import scipy.io
import glob
import random
from scipy.ndimage import affine_transform
import math
DATA_TYPE = 'all'

def load_preprocess(data_dir='data/', pre_keywords='low-posi*'):

    if DATA_TYPE == 'all':
        fs = glob.glob(os.path.join(data_dir, pre_keywords))
        folders = [os.path.basename(folder) for folder in fs]
    else:
        _pre = pre_keywords.split('-')[0]
        folders = [
            f'{_pre}-position-nobody',
            f'{_pre}-position-passenger',
            f'{_pre}-position-sit',
            f'{_pre}-position-stand'
        ]

    # Prepare to collect data and labels
    filename_dataset = []
    distance_dataset = []
    IR_dataset = []
    groudtruth = []
    if DATA_TYPE == 'all':
        labels = ['idle', 'sit', 'sit2stand', 'stand', 'stand2sit']
    else:
        labels = ['idle', 'sit', 'stand']

    # Process each folder
    for folder in folders:
        # Determine the label based on folder name
        label = 'idle' if folder.endswith('nobody') or folder.endswith('passenger') else folder.split('-')[-1]
        
        # Path to the current folder
        folder_path = os.path.join(data_dir, folder)
        
        # Get all file names in the current folder
        files = os.listdir(folder_path)
        
        # Group files by sample ID
        sample_ids = set(file.split('_')[1][:-4] for file in files if file.endswith('.mat'))
        
        # Process each sample
        for sample_id in sample_ids:
            
            # Load CSV and MAT files for the current sample
            csv_file = os.path.join(folder_path, f"{folder}_{sample_id}.csv")
            mat_file = os.path.join(folder_path, f"{folder}_{sample_id}.mat")
            
            if os.path.exists(csv_file):
                distance_dataset.append(distance_preprocess(pd.read_csv(csv_file, header=None, dtype=float).values))
            if os.path.exists(mat_file):
                IR_dataset.append(scipy.io.loadmat(mat_file)['IR_video'])
                filename_dataset.append(f"{folder}_{sample_id}")
            else:
                pass

            groudtruth.append(labels.index(label))


    print('distance length is', len(distance_dataset))
    print('IR length is', len(IR_dataset))
    print('gt length is', len(groudtruth))

    return distance_dataset, IR_dataset, groudtruth, filename_dataset

def skew_data_limited(data, skew_angle):
    """
    对数据进行水平倾斜操作，倾斜角度限制在左右45度以内。
    
    参数:
    - data: 输入数据，形状为(N, H, W)的NumPy数组。
    - skew_angle: 倾斜角度，单位为度。
    
    返回:
    - 倾斜后的数据。
    """
    # 限制倾斜角度在-45到45度之间
    skew_angle = np.clip(skew_angle, -45, 45)
    # 计算倾斜因子
    skew_factor = math.tan(math.radians(skew_angle))
    
    # 定义仿射变换矩阵
    affine_matrix = np.array([[1, skew_factor, 0], [0, 1, 0], [0, 0, 1]])
    
    skewed_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        # 对每一帧应用仿射变换进行倾斜
        skewed_data[i] = affine_transform(data[i], affine_matrix, order=1, mode='reflect', prefilter=False)
    return skewed_data


def distance_preprocess(distances):
    # 创建一个新的 2*n 的 ndarray
    new_array = np.empty((2, distances.shape[1]))

    # 复制原始数组到新数组的第一行
    new_array[0, :] = distances

    # 对于第一行中的每个元素
    for i in range(distances.shape[1]):
        if distances[0, i] == 38000.0:
            # 如果元素是 38000.0，则替换为 50~120 的随机数
            new_array[0, i] = np.random.randint(50, 121)
            # 在新行对应的位置写入 0.001
            new_array[1, i] = 0.001
        else:
            # 否则在新行对应的位置写入 1.0
            new_array[1, i] = 1.0

    # print('old distance is ', distances)
    # print('new is ', new_array)
            
    # scale
    new_array[0, :] = new_array[0, :] / 160.0

    return new_array



def sample_frames_auto(data, num_frames, max_frame_gap=10):
    """根据帧间最大间隔均匀采集指定数量的帧"""
    indices = np.linspace(0, len(data) - 1, num_frames, dtype=int)
    # 检查相邻下标之间的间隔
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > max_frame_gap:
            # 调整下标以保持间隔不超过max_frame_gap
            offset = (indices[i] - indices[i - 1] - max_frame_gap) // 2
            indices[i - 1] += offset
            indices[i] -= offset
    return data[indices]


def sample_frames_fix(data, num_frames, offset, frame_interval=5):
    """按固定间隔采集指定数量的帧，尽量靠中间取"""
    total_frames = len(data)
    start_index = (total_frames - (num_frames - 1) * frame_interval) // 2
    end_index = start_index + (num_frames - 1) * frame_interval + 1

    start_index += offset
    end_index += offset

    if start_index < 0 or end_index > total_frames:
        raise ValueError("无法按指定间隔采集足够数量的帧")

    indices = [start_index + i * frame_interval for i in range(num_frames)]
    return data[indices]

def Data_Augmentation():
    pass

def prepare_datasets(datasets, num_distance_frames, num_IR_frames):
    """准备训练集"""
    # sampled_distance_dataset = [sample_frames_auto(fix_sonic(data), num_distance_frames) for data in distance_dataset]
    # sampled_IR_dataset = [sample_frames_fix(fix_IR(data), num_IR_frames) for data in IR_dataset]
    
    # train
    sampled_distance_dataset = []
    sampled_IR_dataset = []
    sampled_gt = []
    sampled_filename = []
    
    # test
    sampled_distance_dataset_test = []
    sampled_IR_dataset_test = []
    sampled_gt_test = []
    sampled_filename_test = []
    index = 0
    for distance_data, IR_data, gt_data, filename_data in zip(*datasets):
        index += 1
        # 每 5 个取一个数据作为测试集
        if index % 5 == 0:
            # 没有偏移
            offset = 0
            try:
                sampled_distance_data = sample_frames_fix(fix_sonic(distance_data), num_distance_frames, offset, 2)
                sampled_IR_data = sample_frames_fix(fix_IR(IR_data), num_IR_frames, offset, 3)
            except Exception as e:
                continue
            sampled_distance_dataset_test.append(sampled_distance_data)
            sampled_IR_dataset_test.append(sampled_IR_data)
            sampled_gt_test.append(gt_data)
            sampled_filename_test.append(f'offset_{offset}|{filename_data}')
        else: 
            # 数据增强
            # 随机偏移: offset 45 to 83
            # 翻转: 没有效果
            # 倒序: 
            # 噪声: 没效果
            
            # offset
            for offset in [-2, -1, 0, 1, 2]:# [-2, -1, 0, 1, 2]:
                try:
                    sampled_distance_data = sample_frames_fix(fix_sonic(distance_data), num_distance_frames, offset, 2)
                    sampled_IR_data = sample_frames_fix(fix_IR(IR_data), num_IR_frames, offset, 3)
                    # skewed_data = skew_data_limited(sampled_IR_data.reshape(-1,8,8), 50)
                    sampled_distance_dataset.append(sampled_distance_data)
                    
                    sampled_IR_dataset.append(sampled_IR_data)
                    sampled_gt.append(gt_data)
                    sampled_filename.append(f'offset_{offset}|{filename_data}')
                except Exception as e:
                    continue
            
            # TODO
            # noise
            # noise_IR = np.random.randn(*sampled_IR_data.shape) * (sampled_IR_data.max() - sampled_IR_data.min()) * 2
            # noise_DIST = np.random.randn(*sampled_distance_data.shape) * (sampled_distance_data.max() - sampled_distance_data.min())
            
            # TODO
            # flip
            # if random.random() < 0.3:
            #     try:
            #         flipped_IR = sampled_IR_data.reshape(-1,8,8)[:,::-1,:]
            #         sampled_distance_dataset.append(sampled_distance_data)
            #         sampled_IR_dataset.append(flipped_IR.reshape(-1,64))
            #         sampled_gt.append(gt_data)
            #         sampled_filename.append(f'offset_{offset}_flip|{filename_data}')
            #     except Exception as e:
            #             continue
                
            # IR flip
            # # data[:, :, ::-1]__ 
            # sampled_distance_dataset.append(sampled_distance_data)
            # sampled_IR_dataset.append(sampled_IR_data[:, ::-1, :])
            # sampled_gt.append(gt_data)
            # sampled_filename.append(f'offset_{offset}|{filename_data}')
    return (sampled_distance_dataset, sampled_IR_dataset, sampled_gt, sampled_filename), \
        (sampled_distance_dataset_test, sampled_IR_dataset_test, sampled_gt_test, sampled_filename_test)


def fix_IR(IR: np.ndarray):
    return IR.reshape(-1, 64)

def fix_sonic(dis: np.ndarray):
    return dis.T


def make_dataset():
    datasets = load_preprocess(data_dir='./data')
    # 准备训练集
    (sampled_distance_dataset, sampled_IR_dataset, groudtruth, sampled_filename_dataset)\
    ,(sampled_distance_dataset_test, sampled_IR_dataset_test, groudtruth_test, sampled_filename_dataset_test)\
        = prepare_datasets(datasets, 14, 9)

    # 打印结果以验证
    for i in range(2):
        print(f"Distance dataset {i+1}: {len(sampled_distance_dataset)}, {sampled_distance_dataset[i].shape}")
        print(f"IR dataset {i+1}: {len(sampled_IR_dataset)}, {sampled_IR_dataset[i].shape}")
        print(f"gt dataset {i+1}: {len(groudtruth)}")
        print(f"filename dataset {i+1}: {len(sampled_filename_dataset)}")

    return (sampled_distance_dataset, sampled_IR_dataset, groudtruth, sampled_filename_dataset),\
    (sampled_distance_dataset_test, sampled_IR_dataset_test, groudtruth_test, sampled_filename_dataset_test)


if __name__ == '__main__':
    make_dataset()
