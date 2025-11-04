import numpy as np

# 加载.npy文件
data = np.load('data_process/MQL4GRec/Sports/Sports.emb-ViT-L-14.npy')

# 查看基本信息
print("数组形状:", data.shape)
print("数据类型:", data.dtype)
print("数组维度:", data.ndim)
print("前几个元素:", data[:5])  # 显示前5个元素
# 显示统计信息
print("最大值:", data.max())
print("最小值:", data.min())
print("平均值:", data.mean())
print("标准差:", data.std())
# 在Jupyter Notebook中运行
import matplotlib.pyplot as plt

# 如果是2D数据，可以可视化
plt.figure(figsize=(10, 6))
plt.imshow(data[:100, :100])  # 显示前100x100的数据
plt.colorbar()
plt.show()