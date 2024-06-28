import numpy as np
import matplotlib.pyplot as plt

def custom_sigmoid(x, a=0.2, b=32.5):
    return 1 / (1 + np.exp(-a * (x - b)))

# 生成温度数据
temperatures = np.linspace(-10, 90, 100)
mapped_values = custom_sigmoid(temperatures)

# 绘制映射函数图像
plt.figure(figsize=(8, 6))
plt.plot(temperatures, mapped_values, label='Mapped Values')
plt.xlabel('Temperature')
plt.ylabel('Mapped Value')
plt.title('Custom Sigmoid-like Mapping Function')
plt.grid(True)
plt.legend()
plt.show()
