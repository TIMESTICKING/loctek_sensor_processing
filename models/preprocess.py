import os
import numpy as np
import pandas as pd
import scipy.io
import glob

def load_preprocess(data_dir='data', pre_keywords='high-posi*'):

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

    return new_array


if __name__ == '__main__':
    load_preprocess()
