import os
import numpy as np
import pandas as pd
import scipy.io
import glob
from scipy.ndimage import minimum_filter


DATA_TYPE = 'all'

def load_preprocess(data_dir='data/', pre_keywords='high-posi*'):

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


def prepare_datasets(datasets, num_distance_frames, num_IR_frames):
    """准备训练集"""
    # sampled_distance_dataset = [sample_frames_auto(fix_sonic(data), num_distance_frames) for data in distance_dataset]
    # sampled_IR_dataset = [sample_frames_fix(fix_IR(data), num_IR_frames) for data in IR_dataset]
    sampled_distance_dataset = []
    sampled_IR_dataset = []
    sampled_gt = []
    sampled_filename = []

    for distance_data, IR_data, gt_data, filename_data in zip(*datasets):
        for offset in [-1, 0, 1]:
            try:
                sampled_distance_data = sample_frames_fix(fix_sonic(distance_data), num_distance_frames, offset, 2)
                sampled_IR_data = sample_frames_fix(fix_IR(IR_data), num_IR_frames, offset, 3)
            except Exception as e:
                continue
            sampled_distance_dataset.append(sampled_distance_data)
            sampled_IR_dataset.append(sampled_IR_data)
            sampled_gt.append(gt_data)
            sampled_filename.append(f'offset_{offset}|{filename_data}')

    return sampled_distance_dataset, sampled_IR_dataset, sampled_gt, sampled_filename


def fix_IR(IR: np.ndarray):
    return IR.reshape(-1, 64)

def fix_sonic(dis: np.ndarray):
    return dis.T


def make_dataset():
    datasets = load_preprocess(data_dir='../data')
    # 准备训练集
    sampled_distance_dataset, sampled_IR_dataset, groudtruth, sampled_filename_dataset = prepare_datasets(datasets, 14, 9)

    # 打印结果以验证
    for i in range(2):
        print(f"Distance dataset {i+1}: {len(sampled_distance_dataset)}, {sampled_distance_dataset[i].shape}")
        print(f"IR dataset {i+1}: {len(sampled_IR_dataset)}, {sampled_IR_dataset[i].shape}")
        print(f"gt dataset {i+1}: {len(groudtruth)}")
        print(f"filename dataset {i+1}: {len(sampled_filename_dataset)}")

    return sampled_distance_dataset, sampled_IR_dataset, groudtruth, sampled_filename_dataset


if __name__ == '__main__':
    make_dataset()
