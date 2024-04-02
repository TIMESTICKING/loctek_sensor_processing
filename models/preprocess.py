import os
import numpy as np
import pandas as pd
import scipy.io
import glob

def load_preprocess(data_dir='data/', pre_keywords='low-posi*'):

    fs = glob.glob(os.path.join(data_dir, pre_keywords))
    folders = [os.path.basename(folder) for folder in fs]

    # Prepare to collect data and labels
    distance_dataset = []
    IR_dataset = []
    groudtruth = []
    labels = ['idle', 'sit', 'sit2stand', 'stand', 'stand2sit']

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
            else:
                pass

            groudtruth.append(labels.index(label))


    print('distance length is', len(distance_dataset))
    print('IR length is', len(IR_dataset))
    print('gt length is', len(groudtruth))

    return distance_dataset, IR_dataset, groudtruth


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


def sample_frames_fix(data, num_frames, frame_interval=5):
    """按固定间隔采集指定数量的帧，尽量靠中间取"""
    total_frames = len(data)
    start_index = (total_frames - (num_frames - 1) * frame_interval) // 2
    end_index = start_index + (num_frames - 1) * frame_interval + 1

    if start_index < 0 or end_index > total_frames:
        raise ValueError("无法按指定间隔采集足够数量的帧")

    indices = [start_index + i * frame_interval for i in range(num_frames)]
    return data[indices]


def prepare_datasets(distance_dataset, IR_dataset, num_distance_frames, num_IR_frames):
    """准备训练集"""
    # sampled_distance_dataset = [sample_frames_auto(fix_sonic(data), num_distance_frames) for data in distance_dataset]
    # sampled_IR_dataset = [sample_frames_fix(fix_IR(data), num_IR_frames) for data in IR_dataset]
    sampled_distance_dataset = []
    sampled_IR_dataset = []

    for distance_data, IR_data in zip(distance_dataset, IR_dataset):
        try:
            sampled_distance_data = sample_frames_auto(fix_sonic(distance_data), num_distance_frames)
            sampled_IR_data = sample_frames_fix(fix_IR(IR_data), num_IR_frames)
        except Exception as e:
            continue
        sampled_distance_dataset.append(sampled_distance_data)
        sampled_IR_dataset.append(sampled_IR_data)

    return sampled_distance_dataset, sampled_IR_dataset


def fix_IR(IR: np.ndarray):
    return IR.reshape(-1, 64)

def fix_sonic(dis: np.ndarray):
    return dis.T


def make_dataset():
    distance_dataset, IR_dataset, groudtruth = load_preprocess(data_dir='../data')
    # 准备训练集
    sampled_distance_dataset, sampled_IR_dataset = prepare_datasets(distance_dataset, IR_dataset, 20, 5)

    # 打印结果以验证
    for i in range(2):
        print(f"Distance dataset {i+1}: {sampled_distance_dataset[i].shape}")
        print(f"IR dataset {i+1}: {sampled_IR_dataset[i].shape}")

    return sampled_distance_dataset, sampled_IR_dataset, groudtruth


if __name__ == '__main__':
    make_dataset()
