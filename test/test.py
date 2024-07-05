# import numpy as np
# import matplotlib.pyplot as plt

# def custom_sigmoid(x, a=0.2, b=32.5):
#     return 1 / (1 + np.exp(-a * (x - b)))

# # 生成温度数据
# temperatures = np.linspace(-10, 90, 100)
# mapped_values = custom_sigmoid(temperatures)

# # 绘制映射函数图像
# plt.figure(figsize=(8, 6))
# plt.plot(temperatures, mapped_values, label='Mapped Values')
# plt.xlabel('Temperature')
# plt.ylabel('Mapped Value')
# plt.title('Custom Sigmoid-like Mapping Function')
# plt.grid(True)
# plt.legend()
# plt.show()
import scipy.io
import numpy 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter
import scipy.signal as signal
from scipy.ndimage import zoom
from scipy.interpolate import interp2d
from matplotlib.colors import ListedColormap
# mat_file = r"E:\loctek\2024\loctek_sensor_processing\data\low-position-sit2stand\new岑航斌0627_低位_坐姿到站姿_1719467168.mat"
#mat_file = r"E:\loctek\2024\loctek_sensor_processing\data\low-position-stand\new黄镓辉_低位_站姿_1719371024.mat"
mat_file = r"E:\loctek\2024\loctek_sensor_processing\persondata\0704\低位\站姿到坐姿\new0704_低位_站姿到坐姿_1720080730.mat"
#mat_file = r"E:\loctek\2024\loctek_sensor_processing\data\low-position-stand\newxu-0703_低位_站姿_1719967337.mat"
# mat_file = r"E:\loctek\2024\loctek_sensor_processing\data\high-position-nobody\newwuren-s_高位_无人_1719378675.mat"


raw_data = scipy.io.loadmat(mat_file)['IR_video']

# 调用均值滤波函数
filtered_data = np.zeros_like(raw_data)
window_size = 3

# 定义一个滤波器函数，这里使用平均滤波器作为示例
def apply_filter(matrix):
    filtered_matrix = uniform_filter(matrix, size=window_size,mode='reflect')
    # filtered_matrix = signal.medfilt(matrix,kernel_size=window_size)
    return filtered_matrix

for i in range(8):
    for j in range(8):
        # 提取当前格子的数据
        grid_data = raw_data[:, i, j]
        # 应用滤波器
        filtered_grid = apply_filter(grid_data)
        # 存储结果
        filtered_data[:, i, j] = filtered_grid

def interpolate_data_bilinear_highres(data, factor):
    num_frames, size_x, size_y = data.shape
    new_size_x = size_x * factor
    new_size_y = size_y * factor
    interpolated_data = np.zeros((num_frames, new_size_x, new_size_y))
    
    for frame_idx in range(num_frames):
        frame_data = data[frame_idx]
        x = np.arange(size_x)
        y = np.arange(size_y)
        interp_func = interp2d(x, y, frame_data, kind='linear')
        x_new = np.linspace(0, size_x - 1, new_size_x)
        y_new = np.linspace(0, size_y - 1, new_size_y)
        interpolated_frame = interp_func(x_new, y_new)
        interpolated_data[frame_idx] = interpolated_frame
    
    return interpolated_data

# 使用更高分辨率插值处理数据，例如增加到 4 倍分辨率
factor = 4
before_data = interpolate_data_bilinear_highres(raw_data,factor)
after_data = interpolate_data_bilinear_highres(filtered_data,factor)

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# 初始时显示的图像
#im1 = ax1.imshow(before_data[0], cmap='jet',vmin=18,vmax=25, animated=True)
im1 = ax1.imshow(before_data[0], cmap='jet', animated=True)
#im2 = ax2.imshow(after_data[0], cmap='jet',vmin=18,vmax=25, animated=True)
im2 = ax2.imshow(after_data[0], cmap='jet', animated=True)

# 添加标题和标签
ax1.set_title('Before Filtering')
ax2.set_title(f'After Filtering (w={window_size})')

# 更新函数，用于每一帧的更新
def update(frame):
    im1.set_array(before_data[frame])
    im2.set_array(after_data[frame])
    return im1, im2



# 设置动画
ani = FuncAnimation(fig, update, frames=len(before_data), interval=(1000/14))
plt.colorbar(im1, ax=ax1, orientation='vertical', label='Temperature (°C)')
plt.colorbar(im2, ax=ax2, orientation='vertical', label='Temperature (°C)')
# 显示动态图像
plt.show()