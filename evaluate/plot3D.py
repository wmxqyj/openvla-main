#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:qyj 
# datetime:2024/7/30 16:45 
# software: PyCharm
import numpy as np
from matplotlib import pyplot as plt

continuous_actions_pred_np = np.loadtxt('continuous_actions_pred.txt')
continuous_actions_gt_np = np.loadtxt('continuous_actions_gt.txt')


x_pred, y_pred, z_pred = continuous_actions_pred_np[:, 0], continuous_actions_pred_np[:,1], continuous_actions_pred_np[:, 2]
x_gt, y_gt, z_gt = continuous_actions_gt_np[:, 0], continuous_actions_gt_np[:, 1], continuous_actions_gt_np[:, 2]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制预测点
ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o', label='Predicted')
# 绘制真实点
ax.scatter(x_gt, y_gt, z_gt, c='b', marker='x', label='Ground Truth')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
plt.show()
