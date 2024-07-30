from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")


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

    # 将预测动作和真实动作转换为连续动作，并进行L1损失计算
    continuous_actions_pred = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
    )
    continuous_actions_gt = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
    )



    # 4.对比预测数据和真实数据，计算准确率
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = correct_preds.sum().float() / mask.sum().float()
    print("当前准确率为：", action_accuracy)

    # 5.数据可视化

    continuous_actions_pred_np = continuous_actions_pred.detach().cpu().numpy()
    continuous_actions_gt_np = continuous_actions_gt.detach().cpu().numpy()

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

    break




