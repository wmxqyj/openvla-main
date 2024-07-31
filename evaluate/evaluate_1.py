
import tensorflow_datasets as tfds
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import numpy as np
import matplotlib.pyplot as plt


# 将图片保存为gif
def as_gif(images):
    # Render the images as the gif:
    images[0].save('./tmp/temp.gif', save_all=True, append_images=images[1:], duration=1000, loop=0)
    gif_bytes = open('./tmp/temp.gif', 'rb').read()
    return gif_bytes


# 将输出文件保存为
def save_action(action_pred_list, action_gt_list):
    with open('./tmp/actions_data.txt', 'w') as file:
        for pred, gt in zip(action_pred_list, action_gt_list):
            pred_str, gt_str = str(pred), str(gt.numpy())
            pred_str, gt_str = pred_str.replace('\n', ''), gt_str.replace('\n', '')
            file.write(f"{pred_str}, {gt_str}\n")


# 将图像保存为2维对比视图
def save_2dVisual(action_pred_list, action_gt_list, image_list):
    # 创建图表
    param_names = ['x', 'y', 'z', 'pitch', 'roll', 'yaw', 'openOrClose']
    fig, axs = plt.subplots(2, len(param_names), figsize=(20, 5))
    for i, frame in enumerate(image_list[:7]):
        axs[0, i].imshow(frame)
        axs[0, i].axis('off')

    # 绘制数据
    action_pred_list = np.array([item for item in action_pred_list])
    action_gt_list = np.array([item for item in action_gt_list])
    for i, param in enumerate(param_names):
        axs[1, i].plot(action_gt_list[:, i], label='True Value', marker='o')
        axs[1, i].plot(action_pred_list[:, i], label='Predicted Value', marker='x')
        axs[1, i].set_title(param)
        axs[1, i].legend()

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('./tmp/combined_plot.png')
    plt.show()


class DataConfig:
    device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")


cfg = DataConfig()

# 1.导入模型
processor = AutoProcessor.from_pretrained("/data1/gyh/models/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "/data1/gyh/models/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(cfg.device)

builder = tfds.builder_from_directory(builder_dir='/data1/gyh/datasets/bridge_orig/1.0.0')
ds = builder.as_dataset(split='val[:1]')              # 只取train数据集的第一个episode

action_pred_list = []
action_gt_list = []
image_list = []

for episode in ds:
    steps = episode['steps']
    for step in steps:
        image = Image.fromarray(np.array(step['observation']['image_2']))
        image_list.append(image)
        prompt = str(step['language_instruction'])
        inputs = processor(prompt, image).to(cfg.device, dtype=torch.bfloat16)
        action_pred = vla.predict_action(**inputs, unnorm_key='bridge_orig', do_sample=False)
        action_gt = step['action']
        action_pred_list.append(action_pred)
        action_gt_list.append(action_gt)


save_action(action_pred_list, action_gt_list)
as_gif(image_list)
save_2dVisual(action_pred_list, action_gt_list, image_list)




