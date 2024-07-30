"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
import sys
sys.path.append('/data1/qyj/vla')
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    # => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

    # fmt: on


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")   # 当前模型的路径和数据集的名称

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()     # 创建一个partial state的分布式对象
    # torch.cuda.set_device(device_id := distributed_state.local_process_index) # 设置当前设备所要使用的gpu
    device_id = 6
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()               # 情清空cuda缓存
    print("loading gpu success,")
    # Configure Unique Experiment ID & Log Directory  ，生成实验唯一ID，包含模型路径，数据集名称，batch size，learning rate
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    # 根据是否使用lora和量化来完善实验id
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"

    # Start =>> Build Directories
    # 创建运行目录和适配器目录，如果运行目录已经存在就不在创建
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    # 创建量化配置,在配置了lora的前提下。创建一个bitsandbytes对象指定量化细节。
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,      # 模型数据集
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    # 判断模型是否要进行量化训练，否则直接将其布置到设备上
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    # 对模型要使用的lora进行配置
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,         # 防止过拟合
            target_modules="all-linear",           # 所有线性层
            init_lora_weights="gaussian",          # 权重的初始化方法
        )
        vla = get_peft_model(vla, lora_config)   # 调用get_peft_model将vla与lora进行结合
        vla.print_trainable_parameters()         # 打印可训练参数

    # 将vla包装在pytorch的分布式数据并行处理中，实现多GPU训练   TODO 在这一步GPU出现了问题，有待解决
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # 创建一个adamw优化器用于训练模型，首先筛选出需要梯度更新的参数保存在trainable_params中，然后将其作为筛选器的输入
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # action_tokenizer主要是将一系列的文本转化为tokens，主要使用processor.tokenizer进行初始化
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---

    # 创建RLDSBatchTransform的对象，用于对数据集进行批处理转换
    batch_transform = RLDSBatchTransform(
        action_tokenizer,          # 对动作进行标记化的函数
        processor.tokenizer,       # 对文本数据进行标记化的函数
        image_transform=processor.image_processor.apply_transform,     # 用于对图像数据进行变换
        # 构建提示文本，根据模型选择不同的处理函数。主要是作者两版代码的区别
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    # 创建一个RLDSDataset对象，对数据集进行处理
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,       # 数据集的根目录
        cfg.dataset_name,        # 数据集的名称
        batch_transform,         # 定义数据批处理的函数
        resize_resolution = (224, 224),   # 定义图像大小为一个元组形式
        shuffle_buffer_size=cfg.shuffle_buffer_size,        # 设置缓冲区大小
        image_aug=cfg.image_aug,                            # 是否进行图像增强
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # 在main_process为true时执行，保存数据统计信息到指定目录，用于反归一化操作
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

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

    # 在wandb平台上初始化日志记录，只在主进程中执行，init函数为配置参数主要有实体，项目，名称
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # 创建三个双端队列，用于存储近期的训练指标，长度为配置文件中的梯度累计步数，主要包括损失，动作准确率，L1损失
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:    # 使用tqdm创建一个进度条
        vla.train()                 # 转化为训练模式
        optimizer.zero_grad()       # 清零优化器optimizer的梯度
        # 遍历数据加载器，获取每个批次的数据
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):   # 混合精度训练，将张量转化为float16类型吗，加速gpu运算并节省显存
                # 前向传播，将批次数据输vla得到output
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),    # 从数据加载器中获取的输入ID批次
                    attention_mask=batch["attention_mask"].to(device_id),    # 从数据加载器中获取的注意力掩码批次
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),   # 获取图像批次
                    labels=batch["labels"],     # 从数据加载器中获取的标签批次
                )
                loss = output.loss        # 从output中提取损失值

            # 将损失值进行标准化处理。具体为将损失值除以梯度累计步数，以减少训练过程中的梯度爆炸和消失
            normalized_loss = loss / cfg.grad_accumulation_steps

            # 进行反向传播。
            normalized_loss.backward()

            """
            计算准确率和L1损失。
            """
            # 提取模型的直接输出，与动作相关的部分从num_patches : -1
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            # 在第二个维度上（logits 对应每个 token 的预测分数）取最大值的索引，这些索引即为预测的动作 token。
            action_preds = action_logits.argmax(dim=2)
            # 提取动作标签，即从数据加载器中获取的标签批次，从第二个维度开始，并将标签移动到与预测相同的设备上
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            # 创建掩码，用于过滤掉真实动作小于动作开始索引的元素，即没有动作的标签。
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask        # 计算预测正确的动作数量，通过掩码进行筛选
            action_accuracy = correct_preds.sum().float() / mask.sum().float()   # 计算准确率

            # 将预测动作和真实动作转换为连续动作，并进行L1损失计算
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # 每积累grad_accumulation_steps个索引后进行一次梯度更新，得到当前积累步数的索引
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            # 通过求和近期损失列表、近期动作准确率列表和近期损失列表并除以各自列表的长度，来计算平滑指标
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # 如果当前是主进程且挡墙步索引是0的倍数，则进行wandb记录
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {"train_loss": smoothened_loss, "action_accuracy": smoothened_action_accuracy, "l1_loss": smoothened_l1_loss}, step=gradient_step_idx
                )

            # 检查当前是否已经积累足够的梯度，是否需要进行优化步长的操作
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()            # 更新模型参数
                optimizer.zero_grad()       # 清空当前累计梯度
                progress.update()           # 更新训练进度

            # 在每个save_steps的倍数时，进行模型保存操作
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:    # 如果是在主进程中
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir    # 获取保存目录

                    # 保存预训练的处理器和权重
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # 在多线程并行训练中同步各个进程执行，等待所有模型都到达这一步一起执行下一步
                dist.barrier()

                # 将lora权重合并到模型的主干中，加快推理速度
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(  # 加载预训练的视觉模型
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)  # 从vla中创建一个peftModel并指定adap_dir参数为true
                    merged_vla = merged_vla.merge_and_unload()  # 合并权重并卸载一些组件释放内存
                    # 如果当前进程是主进程，则保存合并后的模型
                    if distributed_state.is_main_process:
                        merged_vla.save_pretrained(run_dir)

                # Block on Main Process Checkpointing
                dist.barrier()


if __name__ == "__main__":
    finetune()
