import pandas as pd
import numpy as np
from PIL import Image

# 指定 CSV 文件的路径
filename = r'D:\CSI数据集\分割标注数据\gesture1_1_DB.csv'

# 读取 CSV 文件
data = pd.read_csv(filename)

# 将数据转换为 NumPy 数组
data_array = data.values

# 创建一个空的图像
image = Image.new('L', (270, data_array.shape[0]))

# 将数据填充到图像中
for i in range(data_array.shape[0]):
    for j in range(270):
        pixel_value = int(data_array[i, j] * 255)  # 将数据值映射到 0-255 的范围
        image.putpixel((j, i), pixel_value)

# 保存图像
image.save(r'D:\CSI数据集\1')

# 将图像数据保存为 CSV 文件
# image_data = np.array(image)
# pd.DataFrame(image_data).to_csv(r'C:\Users\YourName\YourFolder\data1.csv', index=False)



