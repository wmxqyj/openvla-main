import tensorflow_datasets as tfds
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np
import matplotlib.pyplot as plt

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

class DataConfig:
    data_root_dir: Path = Path("/data1/gyh/datasets")  # Path to Open-X dataset directory
    dataset_name: str = "bridge"  # Name of fine-tuning dataset (e.g., `droid_wipe`)

    batch_size: int = 1                        # Fine-tuning batch size
    grad_accumulation_steps: int = 1           # Gradient accumulation steps
    image_aug: bool = True                     # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000
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

# 2.加载数据集

# action_tokenizer主要是将一系列的文本转化为tokens，主要使用processor.tokenizer进行初始化
action_tokenizer = ActionTokenizer(processor.tokenizer)

# 创建RLDSBatchTransform的对象，用于对数据集进行批处理转换
batch_transform = RLDSBatchTransform(
    action_tokenizer,  # 对动作进行标记化的函数
    processor.tokenizer,  # 对文本数据进行标记化的函数
    image_transform=processor.image_processor.apply_transform,  # 用于对图像数据进行变换
    # 构建提示文本，根据模型选择不同的处理函数。主要是作者两版代码的区别   TODO 这个提示工程自己改写一下 or no
    prompt_builder_fn=PurePromptBuilder,
)

# 创建一个RLDSDataset对象，对数据集进行处理
vla_dataset = RLDSDataset(
    cfg.data_root_dir,  # 数据集的根目录
    cfg.dataset_name,   # 数据集的名称
    batch_transform,    # 定义数据批处理的函数
    resize_resolution = (224, 224),  # 定义图像大小为一个元组形式
    shuffle_buffer_size=cfg.shuffle_buffer_size,             # 设置缓冲区大小
    image_aug=cfg.image_aug,                                 # 是否进行图像增强
)

# 对数据集进行padding操作，在右侧对数据进行填充
collator = PaddedCollatorForActionPrediction(
    processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
)

# 创建数据加载器，从数据集中加载数据
dataloader = DataLoader(
    vla_dataset,
    batch_size=cfg.batch_size,       # 每个批次的大小
    sampler=None,                    # 默认采样方式
    collate_fn=collator,             # 指定数据批处理的函数，即上述对数据进行的预处理padding操作
    num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
)


# 3.预测数据
# 遍历数据加载器，获取每个批次的数据
vla.eval()
for batch_idx, batch in enumerate(dataloader):
    with torch.autocast("cuda", dtype=torch.bfloat16):  # 混合精度训练，将张量转化为float16类型吗，加速gpu运算并节省显存
        # 前向传播，将批次数据输vla得到output
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(cfg.device),  # 从数据加载器中获取的输入ID批次
            attention_mask=batch["attention_mask"].to(cfg.device),  # 从数据加载器中获取的注意力掩码批次
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(cfg.device),  # 获取图像批次
            labels=batch["labels"],  # 从数据加载器中获取的标签批次
        )

    """
    计算准确率和L1损失。
    """
    # 提取模型的直接输出，与动作相关的部分从num_patches : -1
    action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
    # 在第二个维度上（logits 对应每个 token 的预测分数）取最大值的索引，这些索引即为预测的动作 token。
    action_preds = action_logits.argmax(dim=2)
    # 提取动作标签，即从数据加载器中获取的标签批次，从第二个维度开始，并将标签移动到与预测相同的设备上
    action_gt = batch["labels"][:, 1:].to(action_preds.device)
    # 创建掩码，用于过滤掉真实动作小于动作开始索引的元素，即没有动作的标签。
    mask = action_gt > action_tokenizer.action_token_begin_idx
    # print(action_preds)

    # 4.对比预测数据和真实数据，计算准确率
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = correct_preds.sum().float() / mask.sum().float()
    print("当前准确率为：", action_accuracy)

    # 将预测动作和真实动作转换为连续动作，并进行L1损失计算
    continuous_actions_pred = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
    )
    continuous_actions_gt = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
    )

    # 5.数据可视化-三维坐标
    # 将Tensors转换为NumPy数组
    pred_np = continuous_actions_pred.detach().cpu().numpy()
    gt_np = continuous_actions_gt.detach().cpu().numpy()
    # 保存为文本文件
    np.savetxt('continuous_actions_pred.txt', pred_np)
    np.savetxt('continuous_actions_gt.txt', gt_np)

    break


# 5.数据可视化-散点对比
builder = tfds.builder_from_directory(builder_dir='/data1/gyh/datasets/bridge_orig/1.0.0')
ds = builder.as_dataset(split='val[:1]')              # 只取train数据集的第一个episode

def as_gif(images):
  # Render the images as the gif:
  images[0].save('./tmp/temp.gif', save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open('./tmp/temp.gif', 'rb').read()
  return gif_bytes


action_pred_list = []
action_gt_list = []
image_list = []

for episode in ds:
    steps = episode['steps']
    for step in steps:
        image = Image.fromarray(np.array(step['observation']['image_1']))
        image_list.append(image)
        prompt = str(step['language_instruction'])
        inputs = processor(prompt, image).to(cfg.device, dtype=torch.bfloat16)
        action_pred = vla.predict_action(**inputs, unnorm_key='bridge_orig', do_sample=False)
        action_gt = step['action']
        action_pred_list.append(action_pred)
        action_gt_list.append(action_gt)


# 保存文件
as_gif(image_list)
with open('./tmp/actions_data.txt', 'w') as file:
    for pred, gt in zip(action_pred_list, action_gt_list):
        pred_str, gt_str = str(pred), str(gt.numpy())
        pred_str, gt_str = pred_str.replace('\n', ''), gt_str.replace('\n', '')
        file.write(f"{pred_str}, {gt_str}\n")


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