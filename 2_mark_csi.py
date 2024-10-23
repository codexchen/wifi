import pandas as pd
import os


source_file_path = 'D:\\CSI数据集\\分割标注数据\\gesture4_5_DB.csv'

new_directory = 'D:\\CSI数据集\\分割标注数据\\标注数据'

os.makedirs(new_directory, exist_ok=True)

new_file_path = os.path.join(new_directory, 'gesture4_5_DB.csv')

# 自定义内容
custom_content = 'pai_shou'

data = pd.read_csv(source_file_path)

# 在每一行的最后添加自定义内容
data['Custom Content'] = custom_content


if 'your_label_here' in data.columns:
    data.drop('your_label_here', axis=1, inplace=True)


data.to_csv(new_file_path, index=False, header=False)