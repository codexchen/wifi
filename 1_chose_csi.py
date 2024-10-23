import pandas as pd

csi_matrix_path = 'D:\\CSI数据集\\预处理数据\\pai_shou_DB.csv'  # CSI矩阵文件路径

# 定义导出文件路径
output_matrix_path = r'D:\CSI数据集\分割标注数据\pai_shou_DB.csv'

# 导入CSI矩阵和标注文件
csi_matrix = pd.read_csv(csi_matrix_path, header=None,encoding='gbk')

# 保留第1000行到第1500行的内容
filtered_csi_matrix = csi_matrix.iloc[1000:1500]

# 导出到指定路径
filtered_csi_matrix.to_csv(output_matrix_path, index=False, header=False)

print(f'{output_matrix_path}')
