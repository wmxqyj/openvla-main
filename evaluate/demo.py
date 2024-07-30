import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = []
with open('./tmp/actions_data.txt', 'r') as file:
    for line in file:
        # 分割行中的预测值和真实值
        pred_str, gt_str = line.strip().split('], [')
        # 将字符串转换为浮点数列表
        pred = list(map(float, pred_str.strip('[]').split()))
        gt = list(map(float, gt_str.strip('[]').split()))
        data.append((np.array(pred), np.array(gt)))

# 解析数据
action_pred_list = np.array([item[0] for item in data])
action_gt_list = np.array([item[1] for item in data])

# 确保数据的形状正确
assert action_pred_list.shape == action_gt_list.shape

# 绘制数据
param_names = ['x', 'y', 'z', 'pitch', 'roll', 'yaw', 'openOrClose']

fig, axs = plt.subplots(1, len(param_names), figsize=(20, 5))
for i, param in enumerate(param_names):
    axs[i].plot(action_gt_list[:, i], label='True Value', marker='o')
    axs[i].plot(action_pred_list[:, i], label='Predicted Value', marker='x')
    axs[i].set_title(param)
    axs[i].legend()

plt.savefig('./tmp/test_1.png')
plt.show()

