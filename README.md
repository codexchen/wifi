# 基于WiFi传感的手势识别系统

## 目录
1. [项目简介](#项目简介)
2. [项目结构](#项目结构)
3. [主要功能](#主要功能)
4. [使用说明](#使用说明)
5. [环境要求](#环境要求)
6. [实验结果](#实验结果)
7. [注意事项](#注意事项)
8. [贡献](#贡献)
9. [许可](#许可)

## 项目简介

本项目是一个基于WiFi传感的手势识别系统。它使用信道状态信息（CSI）来识别不同的手势。系统包括数据处理、图像转换和深度学习模型训练等多个组件。

## 项目结构

```
datasets/
├── train
│   ├── conference/
│   │   ├── hua_quan/
│   │   ├── pai_shou/
│   │   ├── shang_hua/
│   │   ├── wo_quan/
│   │   ├── xia_hua/
│   │   ├── you_hua/
│   │   └── zuo_hua/
│   └── lab/
│       ├── hua_quan/
│       ├── pai_shou/
│       ├── shang_hua/
│       ├── wo_quan/
│       ├── xia_hua/
│       ├── you_hua/
│       └── zuo_hua/
└── test
    ├── conference/
    │   ├── hua_quan/
    │   ├── pai_shou/
    │   ├── shang_hua/
    │   ├── wo_quan/
    │   ├── xia_hua/
    │   ├── you_hua/
    │   └── zuo_hua/
    └── lab/
        ├── hua_quan/
        ├── pai_shou/
        ├── shang_hua/
        ├── wo_quan/
        ├── xia_hua/
        ├── you_hua/
        └── zuo_hua/
```

## 主要功能

1. **chose_csi.py**: 选择CSI数据中的特定部分进行处理。
2. **mark_csi.py**: 为CSI数据添加标签。
3. **csi_image.py**: 将CSI数据转换为图像格式。
4. **train_with_lstm_cnn.py**: 使用LSTM-CNN模型进行手势识别训练。
5. **train_with_lstm.py**: 使用LSTM模型进行手势识别训练。
6. **train_with_cnn.py**: 使用cnn模型进行手势识别训练。

## 使用说明
请将脚本内的路径改为实际存放位置
1. 数据准备：
   - 使用 `chose_csi.py` 选择手势波动的CSI数据段。
   - 使用 `mark_csi.py` 为数据添加正确的手势标签。

2. 数据转换：
   - 运行 `csi_image.py` 将CSI数据转换为图像格式。

3. 模型训练：
   - 确保数据集按照上述结构组织。
   - 运行 `train_with_lstm_cnn.py` 开始训练模型。

4. 结果分析：
   - 训练完成后，查看生成的 `training_history_gesture.png` 了解模型的训练过程。
   - 训练好的模型将保存为 `lstm_cnn_gesture_model.pth`。
   - 运行 `train_with_lstm.py` 开始训练模型。
   - 运行 `train_with_cnn.py` 开始训练模型。

## 环境要求

请查看 `requirements.txt` 文件了解所需的Python包。

## 实验结果

下表展示了三种神经网络模型（LSTM-CNN、LSTM和CNN）在实验室和会议室两种环境下，对七种不同手势的识别准确率：

<table>
  <tr>
    <th colspan="8">表1: 三种神经网络实验结果</th>
  </tr>
  <tr>
    <th></th>
    <th>左挥</th>
    <th>右挥</th>
    <th>上挥</th>
    <th>下挥</th>
    <th>握拳</th>
    <th>敲击</th>
    <th>画圈</th>
  </tr>
  <tr>
    <td colspan="8"><strong>实验室</strong></td>
  </tr>
  <tr>
    <td>LSTM-CNN</td>
    <td>93.6%</td>
    <td>92.7%</td>
    <td>95%</td>
    <td>93.3%</td>
    <td>97.2%</td>
    <td>98%</td>
    <td>93.3%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>87.9%</td>
    <td>82.4%</td>
    <td>90.3%</td>
    <td>77.5%</td>
    <td>87.1%</td>
    <td>85.3%</td>
    <td>79.1%</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>85.5%</td>
    <td>80.4%</td>
    <td>79.2%</td>
    <td>75.4%</td>
    <td>79.5%</td>
    <td>82.1%</td>
    <td>77.4%</td>
  </tr>
  <tr>
    <td colspan="8"><strong>会议室</strong></td>
  </tr>
  <tr>
    <td>LSTM-CNN</td>
    <td>94.7%</td>
    <td>94.1%</td>
    <td>96%</td>
    <td>94.6%</td>
    <td>98.1%</td>
    <td>98.6%</td>
    <td>95.3%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>87.5%</td>
    <td>83.9%</td>
    <td>90.7%</td>
    <td>79.2%</td>
    <td>89.1%</td>
    <td>84.3%</td>
    <td>76.4%</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>81%</td>
    <td>78.6%</td>
    <td>77.5%</td>
    <td>75.4%</td>
    <td>80.3%</td>
    <td>80.6%</td>
    <td>77.2%</td>
  </tr>
</table>

从实验结果可以看出，LSTM-CNN模型在两种环境下对所有手势的识别效果都优于LSTM和CNN模型。特别是在会议室环境中，LSTM-CNN模型的表现更为出色，对大多数手势的识别准确率都在95%以上。

## 注意事项

- 请确保在运行脚本之前正确设置数据路径。
- 根据实际情况调整 `train_with_lstm_cnn.py` 中的超参数，如 `batch_size`、`num_epochs` 等。

## 贡献

欢迎提出问题和贡献代码。请遵循标准的GitHub工作流程：Fork、修改、提交Pull Request。

