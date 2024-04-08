import numpy as np
from scipy.ndimage import minimum_filter

# 创建一个示例数组
array = np.random.rand(10)

# 应用最小值滤波，滤波器大小为 2x2
filtered_array = minimum_filter(array, size=3)

print("原始数组:")
print(array)
print("经过最小值滤波后的数组:")
print(filtered_array)
