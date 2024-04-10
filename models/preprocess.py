import os
import numpy as np
import pandas as pd
import scipy.io
import glob
from scipy.ndimage import minimum_filter
import torch


DATA_TYPE = 'all'
data_version = 2

# 转换为PyTorch张量并缩放数据
def scale_IR(dataset):
    tensor = torch.tensor(dataset, dtype=torch.float32)
    max_val = 35. # tensor.max()
    min_val = 15. # tensor.min()
    scaled_tensor = (tensor - min_val) / (max_val - min_val)
    return scaled_tensor, max_val, min_val



def distance_preprocess(distances):
    # 创建一个新的 2*n 的 ndarray
    # new_array = np.empty((2, distances.shape[1]))

    # # 复制原始数组到新数组的第一行
    # new_array[0, :] = distances

    # # 对于第一行中的每个元素
    # for i in range(distances.shape[1]):
    #     if distances[0, i] == 38000.0:
    #         # 如果元素是 38000.0，则替换为 50~120 的随机数
    #         new_array[0, i] = np.random.randint(50, 121)
    #         # 在新行对应的位置写入 0.001
    #         new_array[1, i] = 0.001
    #     else:
    #         # 否则在新行对应的位置写入 1.0
    #         new_array[1, i] = 1.0

    # # print('old distance is ', distances)
    # # print('new is ', new_array)
            
    # # scale
    # new_array[0, :] = new_array[0, :] / 160.0

    filtered_array = minimum_filter(distances, size=3) / 160.0

    return filtered_array



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


def load_preprocess(data_dir='data/', pre_keywords='low-posi*'):

    if DATA_TYPE == 'all':
        fs = glob.glob(os.path.join(data_dir, pre_keywords))
        folders = [os.path.basename(folder) for folder in fs]
    else:
        _pre = pre_keywords.split('-')[0]
        folders = [
            f'{_pre}-position-nobody',
            # f'{_pre}-position-passenger',
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
        sample_ids = set(file.split('_')[1][:-4] if data_version == 1 else file[:-4] for file in files if file.endswith('.mat'))
        
        # Process each sample
        for sample_id in sample_ids:
            
            # Load CSV and MAT files for the current sample
            t_filename = f'{folder}_{sample_id}' if data_version == 1 else sample_id
            csv_file = os.path.join(folder_path, f"{t_filename}.csv")
            mat_file = os.path.join(folder_path, f"{t_filename}.mat")
            
            if os.path.exists(csv_file):
                distance_dataset.append(distance_preprocess(pd.read_csv(csv_file, header=None, dtype=float).values))
            if os.path.exists(mat_file):
                IR_dataset.append(scipy.io.loadmat(mat_file)['IR_video'])
                filename_dataset.append(f'{folder}_{sample_id}')
            else:
                pass

            groudtruth.append(labels.index(label))


    print('distance length is', len(distance_dataset))
    print('IR length is', len(IR_dataset))
    print('gt length is', len(groudtruth))

    return distance_dataset, IR_dataset, groudtruth, filename_dataset

   
def _mirror_IR(IR_data):
    IR_data_mirrored = IR_data.reshape(-1, 8, 8)[:,:,::-1]
    return IR_data_mirrored.reshape(-1, 64)


def _reverse_timeline(IR, distance, gt):
    IR_re = IR[::-1, :]
    distance_re = distance[::-1, :]

    if DATA_TYPE == 'all':
        '''swap the label such that sit2stand is stand2sit, vise versa'''
        if gt == 2:
            gt == 4
        elif gt == 4:
            gt == 2
    
    return IR_re, distance_re, gt


def prepare_datasets(datasets, ratio, num_distance_frames, num_IR_frames):
    # set parameters
    SONIC_INTERVAL = 2
    IR_INTERVAL = 3

    # split the dataset
    assert len(datasets[0]) == len(datasets[1]) == len(datasets[2]) == len(datasets[3]), 'datasets length not match!'

    num = len(datasets[0])
    ids = np.arange(num)
    np.random.shuffle(ids)

    ids_train, ids_test = ids[:int(num * ratio)], ids[int(num * ratio):]

    """准备训练集"""
    # sampled_distance_dataset = [sample_frames_auto(fix_sonic(data), num_distance_frames) for data in distance_dataset]
    # sampled_IR_dataset = [sample_frames_fix(fix_IR(data), num_IR_frames) for data in IR_dataset]
    sampled_distance_train_dataset = []
    sampled_IR_train_dataset = []
    sampled_train_gt = []

    for i in ids_train:
        distance_data, IR_data, gt_data = datasets[0][i], datasets[1][i], datasets[2][i]
        for offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            try:
                sampled_distance_data = sample_frames_fix(fix_sonic(distance_data), num_distance_frames, offset, SONIC_INTERVAL)
                sampled_IR_data = sample_frames_fix(fix_IR(IR_data), num_IR_frames, offset, IR_INTERVAL)
            except Exception as e:
                continue
            sampled_distance_train_dataset.append(sampled_distance_data)
            sampled_IR_train_dataset.append(sampled_IR_data)
            sampled_train_gt.append(gt_data)

            '''data augment'''
            # mirror the IR
            mirrored_IR = _mirror_IR(sampled_IR_data)
            sampled_distance_train_dataset.append(sampled_distance_data)
            sampled_IR_train_dataset.append(mirrored_IR)
            sampled_train_gt.append(gt_data)
            # reverse timeline
            IR_re, distance_re, gt_re = _reverse_timeline(sampled_IR_data, sampled_distance_data, gt_data)
            sampled_distance_train_dataset.append(distance_re)
            sampled_IR_train_dataset.append(IR_re)
            sampled_train_gt.append(gt_re)


    # data scale
    train_distance_tensor = torch.tensor(np.stack(sampled_distance_train_dataset, axis=0), dtype=torch.float32)
    train_IR_tensor, max_IR, min_IR = scale_IR(np.stack(sampled_IR_train_dataset, axis=0))

    train_dataset = (train_distance_tensor, train_IR_tensor, sampled_train_gt, [0] * len(sampled_train_gt))

    '''prepare the test set'''
    sampled_distance_test_dataset = []
    sampled_IR_test_dataset = []
    sampled_test_gt = []
    sampled_test_filename = []

    for i in ids_test:
        distance_data, IR_data, gt_data, filename = datasets[0][i], datasets[1][i], datasets[2][i], datasets[3][i]
        try:
            sampled_distance_data = sample_frames_fix(fix_sonic(distance_data), num_distance_frames, 0, SONIC_INTERVAL)
            sampled_IR_data = sample_frames_fix(fix_IR(IR_data), num_IR_frames, 0, IR_INTERVAL)
        except Exception as e:
            continue
        sampled_distance_test_dataset.append(sampled_distance_data)
        sampled_IR_test_dataset.append(sampled_IR_data)
        sampled_test_gt.append(gt_data)
        sampled_test_filename.append(filename)

    # data scale
    test_distance_tensor = torch.tensor(np.stack(sampled_distance_test_dataset, axis=0), dtype=torch.float32)
    test_IR_tensor, max_IR, min_IR = scale_IR(np.stack(sampled_IR_test_dataset, axis=0))

    test_dataset = (test_distance_tensor, test_IR_tensor, sampled_test_gt, sampled_test_filename)

    return train_dataset, test_dataset


def fix_IR(IR: np.ndarray):
    return IR.reshape(-1, 64)

def fix_sonic(dis: np.ndarray):
    return dis.T


def make_dataset():
    datasets = load_preprocess(data_dir='../data_v2', pre_keywords='low-posi*')
    # 准备训练集
    train_dataset, test_dataset = prepare_datasets(datasets, 0.7, 14, 9)

    # 打印结果以验证
    print(f"Distance train dataset: {train_dataset[0].shape}")
    print(f"IR train dataset: {train_dataset[1].shape}")
    print(f"gt train dataset: {len(train_dataset[2])}")

    print("test dataset has file amount: ", test_dataset[0].shape)

    return train_dataset, test_dataset


if __name__ == '__main__':
    make_dataset()
