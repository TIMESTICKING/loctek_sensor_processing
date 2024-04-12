import numpy as np
from scipy.interpolate import interp1d

def nearest_neighbor_interpolate_and_analyze(arr):
    # 找出不需要插值的索引和对应的值
    valid_indices = np.where(arr != 38000)[0]
    valid_values = arr[valid_indices]

    # 创建最近邻插值函数
    interp_func = interp1d(valid_indices, valid_values, kind='nearest', fill_value="extrapolate")
    
    # 找出需要插值的索引
    indices_to_interpolate = np.where(arr == 38000)[0]

    # 进行插值
    if len(indices_to_interpolate) > 0:
        arr[indices_to_interpolate] = interp_func(indices_to_interpolate)
    
    # print(arr)

    # 检查总体数值是否大于70
    if np.mean(arr) > 70:
        return (True, 0)

    # 检查总体方差是否过大（这里方差阈值设为1000，可调整）
    if np.var(arr) > 45:
        return (True, -1)

    # 如果以上条件都不满足，返回(False, None)
    return (False, None)

# 示例数组
array_example = np.array([90, 55, 75, 104, 50, 60, 103])

# 使用函数
result = nearest_neighbor_interpolate_and_analyze(array_example)
print(result)
