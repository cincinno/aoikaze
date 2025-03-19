from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch

# ================== 配置区 ==================
model_path = "/users/um202270118/AI/deepseek/deepseek/DeepSeek-R1-Distill-Qwen-14B"
dataset_path = './mixed_dataset.jsonl'
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 模型加载 ==================
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    max_memory={i: "40GB" for i in range(8)}, # 8卡配置示例
    offload_folder="offload",
)
'''
# ================== 适配器配置 ==================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
'''
# ================== 参数统计 ==================
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"可训练参数: {trainable_params} | 总参数: {all_param} | 占比: {100*trainable_params/all_param:.2f}%")

print_trainable_parameters(model)

# ================== 数据处理 ==================
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 数据格式化函数
# 数据格式化函数
flag=True
def format_instruction(sample):
    dialog = []
    has_system = False
    system_msg = ""
    
    # 遍历对话记录
    for msg in sample['conversations']:  # 注意字段名要与数据格式一致
        if msg["role"] == "system":
            system_msg = msg["content"]
            continue
            
        if msg["role"] == "user":
            # 处理第一个用户消息时嵌入系统提示
            if not has_system and system_msg:
                content = f"\n{system_msg}\n<｜User｜>\n{msg['content']}"
                has_system = True
            else:
                content = '<｜User｜>'+msg['content']
            
            # 保留中文冒号（无需替换）
            dialog.append(f"{content}")
            
        elif msg["role"] == "assistant":
            # 添加助手回复和结束标记
            dialog.append(f"<｜Assistant｜>{msg['content']} {tokenizer.eos_token}")

    formatted_text = "\n".join(dialog)
    global flag
    if flag:print(formatted_text)
    flag=False
    return {"text": formatted_text}

# 加载并处理数据集
#dataset = load_dataset('json', data_files=dataset_path, split='train')
dataset = load_dataset('json', data_files=dataset_path)['train']
dataset = dataset.map(format_instruction, remove_columns=['conversations'])


# ================== 训练参数 ==================
training_args = SFTConfig(
    output_dir="./14B_kobayakawa",
    per_device_train_batch_size=1,        # 降低批次大小
    gradient_accumulation_steps=8,        # 调整梯度累积步数
    num_train_epochs=3,
    learning_rate=2e-5,                   # 提升学习率到常规范围
    logging_steps=20,
    fp16=False,                           # 禁用FP16
    bf16=torch.cuda.is_bf16_supported(),  # 优先使用BF16
    save_total_limit=2,
    report_to="tensorboard",
    max_grad_norm=1.0,                    # 适当放宽梯度裁剪
    optim="adamw_torch",
    max_seq_length=2048,                   # 缩短序列长度
    packing=True,                         # 启用序列打包
    gradient_checkpointing=True           # 启用梯度检查点
)
# ================== 训练初始化 ==================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
try:
    print("启动监督式微调...")
    trainer.train()
except KeyboardInterrupt:
    print("训练被中断")
finally:
    # 保存最终模型
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"模型和分词器已保存至 {training_args.output_dir}")