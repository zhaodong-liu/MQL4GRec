# 文本处理流程

本文档描述了MQL4GRec系统的文本嵌入生成流程，遵循与图像处理流程相同的结构。

## 概述

文本处理流程将商品的文本信息（标题+描述）转换为稠密嵌入向量，使用预训练语言模型（默认使用LLaMA-7B）。这些嵌入随后被量化为离散代码，用于基于T5的生成式推荐模型。

## 流程架构

```
商品元数据 (JSON)
    ↓
[加载与清洗文本]
    ↓
标题 + 描述
    ↓
[分词处理]
    ↓
LLaMA/BERT 编码器
    ↓
[平均池化]
    ↓
文本嵌入 (.npy)
    ↓
[RQVAE 量化]
    ↓
离散文本代码 (.index_lemb.json)
```

## 阶段1：文本数据加载

**脚本：** `amazon_text_emb.py` - `load_data()` 函数

**输入：**
- `{dataset}.item.json`：商品元数据文件

**处理过程：**
1. 加载商品元数据JSON文件
2. 提取商品特征（标题、描述、品牌、类别）

**输出：**
- 字典映射：item_id → 元数据字段

**元数据示例：**
```json
{
  "0": {
    "title": "Ernie Ball Regular Slinky 电吉他弦",
    "description": "电吉他专用高质量镍缠绕琴弦...",
    "brand": "Ernie Ball",
    "categories": "乐器,吉他,琴弦"
  }
}
```

## 阶段2：文本生成与清洗

**脚本：** `amazon_text_emb.py` - `generate_text()` 和 `preprocess_text()` 函数

**输入：**
- 商品元数据字典
- 使用的特征字段（默认：`['title', 'description']`）

**处理过程：**
1. 对每个商品，提取指定的文本字段
2. 使用 `clean_text()` 函数清洗文本：
   - 移除HTML实体
   - 规范化空白字符
   - 去除特殊字符
3. 拼接字段（标题 + 描述）
4. 按商品ID排序

**输出：**
- 元组列表：`[(item_id, [cleaned_texts]), ...]`

**示例：**
```python
[
  (0, ["Ernie Ball Regular Slinky 电吉他弦", "电吉他专用高质量镍缠绕琴弦..."]),
  (1, ["Fender 吉他拨片", "优质赛璐珞拨片，多种颜色..."]),
  ...
]
```

## 阶段3：文本嵌入生成

**脚本：** `amazon_text_emb.py` - `generate_item_embedding()` 函数

**输入：**
- 清洗后的文本列表
- 预训练语言模型（LLaMA-7B / BERT）
- 分词器

**处理过程：**

### 3.1 文本分词
```python
tokenizer(
    sentences,
    max_length=2048,        # 最大序列长度
    truncation=True,        # 超长则截断
    return_tensors='pt',    # 返回PyTorch张量
    padding="longest"       # 填充到批次中最长的序列
)
```

### 3.2 模型前向传播
```python
outputs = model(
    input_ids=encoded_sentences.input_ids,
    attention_mask=encoded_sentences.attention_mask
)
```

### 3.3 平均池化
在序列维度上应用注意力掩码的平均池化：
```python
masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
mean_output = masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
```

### 3.4 字段聚合
对多个文本字段（标题、描述）的嵌入取平均：
```python
field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
```

**输出：**
- `{dataset}.emb-{model_name}-td.npy`
- 形状：`[商品数量, 嵌入维度]`
- 示例：`Instruments.emb-llama-td.npy` → `[30000, 4096]`

**支持的模型：**

| 模型 | 嵌入维度 | 速度 | 质量 |
|-------|---------------|-------|---------|
| huggyllama/llama-7b | 4096 | 中等 | 最佳 |
| bert-base-uncased | 768 | 快速 | 良好 |
| bert-large-uncased | 1024 | 中等 | 较好 |

## 阶段4：RQVAE量化

**脚本：**
- 训练：`index/main.py` 或 `index/main_mul.py`
- 代码生成：`index/generate_indices_distance.py`

**输入：**
- `{dataset}.emb-llama-td.npy`：文本嵌入

**处理过程：**

### 4.1 训练RQVAE模型
```bash
cd index
python main.py \
    --data_root ../data_process/MQL4GRec/{dataset} \
    --embedding_file .emb-llama-td.npy \
    --num_emb_list 256 256 256 256 \
    --sk_epsilons 0.05 0.1 0.1 0.1 \
    --epochs 300
```

**关键参数：**
- `num_emb_list`：4个量化级别的码本大小（默认：每级256）
- `sk_epsilons`：Sinkhorn-Knopp算法的epsilon值，用于软分配
- `embedding_file`：文本嵌入文件后缀

### 4.2 生成离散代码
```bash
python generate_indices_distance.py \
    --ckpt_path ./ckpt/{dataset}/text_rqvae_model.pth \
    --data_root ../data_process/MQL4GRec/{dataset} \
    --embedding_file .emb-llama-td.npy \
    --output_file .index_lemb.json
```

**输出：**
- `{dataset}.index_lemb.json`：商品ID → 离散token序列
- 格式：`{"0": ["<a_123>", "<b_45>", "<c_67>", "<d_89>"], ...}`

**Token命名规则：**
- 第1层：`<a_{code}>` (0-255)
- 第2层：`<b_{code}>` (0-255)
- 第3层：`<c_{code}>` (0-255)
- 第4层：`<d_{code}>` (0-255)

## 使用示例

### 示例1：基础文本嵌入生成

```bash
export CUDA_VISIBLE_DEVICES=0

python amazon_text_emb.py \
    --dataset Instruments \
    --root ./MQL4GRec_data \
    --gpu_id 0 \
    --plm_name llama \
    --model_name_or_path huggyllama/llama-7b \
    --max_sent_len 2048
```

### 示例2：使用BERT替代LLaMA

```bash
python amazon_text_emb.py \
    --dataset Arts \
    --root ./MQL4GRec_data \
    --plm_name bert \
    --model_name_or_path bert-base-uncased \
    --max_sent_len 512
```

### 示例3：多数据集批量处理

```bash
#!/bin/bash
DATASETS=("Instruments" "Arts" "Games")

for dataset in "${DATASETS[@]}"; do
    echo "正在处理 $dataset..."
    python amazon_text_emb.py \
        --dataset $dataset \
        --root ./MQL4GRec_data \
        --gpu_id 0
done
```

### 示例4：仅使用CPU处理

```bash
python amazon_text_emb.py \
    --dataset Beauty \
    --root ./MQL4GRec_data \
    --gpu_id -1 \
    --max_sent_len 512
```

### 示例5：使用词丢弃数据增强

```bash
python amazon_text_emb.py \
    --dataset Instruments \
    --root ./MQL4GRec_data \
    --word_drop_ratio 0.1  # 随机丢弃10%的词
```

## 完整流程脚本

**文件：** `data_process/3_get_text_emb.sh`

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python data_process/amazon_text_emb.py \
    --dataset Instruments \
    --root data_process/MQL4GRec \
    --gpu_id 0 \
    --plm_name llama \
    --model_name_or_path huggyllama/llama-7b \
    --model_cache_dir ~/.cache/huggingface \
    --max_sent_len 2048

echo "文本嵌入生成成功！"
```

## 参数说明

### 文本嵌入生成 (`amazon_text_emb.py`)

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `--dataset` | str | Instruments | 数据集名称 |
| `--root` | str | data_process/MQL4GRec | 数据根目录 |
| `--gpu_id` | int | 0 | GPU编号（-1表示CPU） |
| `--plm_name` | str | llama | 模型类型标识符 |
| `--model_name_or_path` | str | huggyllama/llama-7b | HuggingFace模型路径 |
| `--model_cache_dir` | str | .cachemodels | 模型缓存目录 |
| `--max_sent_len` | int | 2048 | 最大分词长度 |
| `--word_drop_ratio` | float | -1 | 词丢弃比例（< 0则禁用） |

### RQVAE训练 (`index/main.py`)

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `--data_root` | str | - | 数据集目录 |
| `--embedding_file` | str | .emb-llama-td.npy | 嵌入文件后缀 |
| `--num_emb_list` | int[] | 256 256 256 256 | 每层码本大小 |
| `--sk_epsilons` | float[] | 0.05 0.1 0.1 0.1 | Sinkhorn epsilon值 |
| `--epochs` | int | 300 | 训练轮数 |
| `--batch_size` | int | 4096 | 训练批次大小 |
| `--lr` | float | 1e-3 | 学习率 |

## 性能优化

### 内存优化

1. **降低最大序列长度：**
   ```bash
   --max_sent_len 512  # 而不是2048
   ```

2. **使用更小的模型：**
   ```bash
   --model_name_or_path bert-base-uncased  # 768维而不是4096维
   ```

3. **在代码中减小批次大小：**
   在 `generate_item_embedding()` 函数中修改 `batch_size = 1`

### 速度优化

1. **使用缓存模型：**
   ```bash
   export TRANSFORMERS_CACHE=/fast/cache/huggingface
   --model_cache_dir /fast/cache/huggingface
   ```

2. **启用混合精度（修改代码）：**
   ```python
   with torch.cuda.amp.autocast():
       outputs = model(...)
   ```

3. **并行处理不同数据集：**
   ```bash
   # 在不同GPU上处理不同数据集
   CUDA_VISIBLE_DEVICES=0 python amazon_text_emb.py --dataset Arts &
   CUDA_VISIBLE_DEVICES=1 python amazon_text_emb.py --dataset Games &
   ```

## 数据流图

```
amazon18_data/
└── Metadata/
    └── meta_Instruments.json.gz
            ↓
    [amazon18_data_process.py]
            ↓
MQL4GRec_data/Instruments/
├── Instruments.item.json  ← 商品元数据
└── Instruments.item2id    ← ID映射
            ↓
    [amazon_text_emb.py]
    - 加载元数据
    - 清洗文本（标题 + 描述）
    - 使用LLaMA分词
    - 提取隐藏状态
    - 平均池化
            ↓
MQL4GRec_data/Instruments/
└── Instruments.emb-llama-td.npy  ← 稠密嵌入 [N, 4096]
            ↓
    [index/main.py]
    - 训练4层码本的RQVAE
    - 残差量化
    - Sinkhorn-Knopp优化
            ↓
index/ckpt/Instruments/
└── text_rqvae_model.pth  ← 训练好的量化器
            ↓
    [index/generate_indices_distance.py]
    - 加载RQVAE模型
    - 量化嵌入
    - 生成离散代码
            ↓
MQL4GRec_data/Instruments/
└── Instruments.index_lemb.json  ← 离散代码
            {
              "0": ["<a_123>", "<b_45>", "<c_67>", "<d_89>"],
              "1": ["<a_67>", "<b_234>", "<c_12>", "<d_156>"],
              ...
            }
            ↓
    [pretrain.py / finetune.py]
    - 将tokens添加到T5分词器
    - 使用代码训练生成模型
```

## 与图像处理流程对比

| 方面 | **文本流程** | **图像流程** |
|--------|-------------------|-------------------|
| **输入数据** | 标题 + 描述 | 商品JPG图像 |
| **预处理** | 文本清洗、分词 | 调整大小、归一化、裁剪 |
| **编码器** | LLaMA-7B (4096维) | CLIP ViT-L/14 (768维) |
| **池化方式** | 注意力掩码平均池化 | CLS token / 平均池化 |
| **输出文件** | `.emb-llama-td.npy` | `.emb-ViT-L-14.npy` |
| **量化代码** | `.index_lemb.json` | `.index_vitemb.json` |
| **Token前缀** | `<a_*>`, `<b_*>`, `<c_*>`, `<d_*>` | `<A_*>`, `<B_*>`, `<C_*>`, `<D_*>` |
| **典型维度** | 4096 (LLaMA) 或 768 (BERT) | 768 (ViT-L) 或 512 (ViT-B) |
| **处理时间** | ~30-60分钟 (3万商品, GPU) | ~15-30分钟 (3万商品, GPU) |

## 故障排除

### 问题1：内存不足

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解决方案：**
1. 降低序列长度：`--max_sent_len 512`
2. 使用更小的模型：`--model_name_or_path bert-base-uncased`
3. 使用CPU：`--gpu_id -1`
4. 在代码中减小批次大小：修改 `batch_size = 1`

### 问题2：模型下载失败

**症状：**
```
OSError: Can't load config for 'huggyllama/llama-7b'
```

**解决方案：**
```bash
# 预先下载模型
export HF_ENDPOINT=https://hf-mirror.com  # 如需使用镜像
python -c "from transformers import AutoModel; AutoModel.from_pretrained('huggyllama/llama-7b')"

# 或指定本地路径
--model_name_or_path /path/to/local/llama-7b
```

### 问题3：处理速度慢

**症状：**
中等规模数据集处理需要数小时

**解决方案：**
1. 检查GPU利用率：`nvidia-smi`
2. 确保CUDA可用：
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
3. 使用缓存模型（设置 `TRANSFORMERS_CACHE`）
4. 降低max_sent_len以加快分词速度

### 问题4：嵌入为空或无效

**症状：**
```
Embeddings shape: (30000, 0) 或出现NaN值
```

**解决方案：**
1. 检查商品元数据文件是否存在：`ls MQL4GRec_data/Instruments/*.item.json`
2. 验证文本内容不为空
3. 确保模型正确加载（检查日志）
4. 在处理过程中检查GPU内存

## 测试与验证

### 验证嵌入质量

```python
import numpy as np

# 加载嵌入
emb = np.load('MQL4GRec_data/Instruments/Instruments.emb-llama-td.npy')

print(f"形状: {emb.shape}")              # 应为 [商品数, 4096]
print(f"均值: {emb.mean():.4f}")         # 应接近0
print(f"标准差: {emb.std():.4f}")        # 应约为1
print(f"包含NaN: {np.isnan(emb).any()}")  # 应为False
print(f"包含Inf: {np.isinf(emb).any()}")  # 应为False

# 检查商品间的相似度
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(emb[:10], emb[:10])
print(f"自相似度:\n{sim}")
```

### 验证量化代码

```python
import json

# 加载代码
with open('MQL4GRec_data/Instruments/Instruments.index_lemb.json') as f:
    codes = json.load(f)

print(f"商品数量: {len(codes)}")
print(f"示例代码: {codes['0']}")  # 应为 ['<a_X>', '<b_Y>', '<c_Z>', '<d_W>']

# 验证所有商品都有4个代码（4个量化层级）
assert all(len(v) == 4 for v in codes.values()), "所有商品应有4个代码"
print("✓ 所有商品都有4个量化代码")
```

## 与T5模型集成

生成文本代码后，它们将在T5训练中使用：

### 1. 添加Tokens到分词器

```python
from data import BaseDataset

dataset = BaseDataset(args)
new_tokens = dataset.get_new_tokens()  # 获取 <a_*>, <b_*>, <c_*>, <d_*>

tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

### 2. 创建训练序列

**序列推荐任务 (seqrec)：**
```
输入:  "Recommend items based on history: <a_12> <b_34> <c_56> <d_78> <a_23> <b_45> <c_67> <d_89>"
目标: "<a_99> <b_11> <c_22> <d_33>"
```

### 3. 约束生成

在推理过程中，使用基于Trie的约束确保生成有效的商品代码：
```python
from generation_trie import Trie

# 从有效的商品代码构建Trie
trie = Trie()
for item_codes in codes.values():
    trie.add(tokenizer.convert_tokens_to_ids(item_codes))

# 约束束搜索
outputs = model.generate(
    input_ids=inputs,
    max_length=20,
    num_beams=20,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist())
)
```

## 后续步骤

完成文本嵌入生成后：

1. **训练量化器**（如果还没做）：
   ```bash
   cd index
   bash scripts/run.sh  # 同时训练文本和图像RQVAE
   ```

2. **生成代码**：
   ```bash
   bash scripts/gen_code_dis.sh
   ```

3. **预训练T5**：
   ```bash
   cd ..
   bash scripts/pretrain.sh
   ```

4. **微调**：
   ```bash
   bash scripts/finetune.sh
   ```

## 参考资料

### 相关文件
- `data_process/amazon_text_emb.py`：主要文本嵌入脚本
- `data_process/utils.py`：辅助函数（clean_text, load_plm等）
- `index/main.py`：单数据集RQVAE训练
- `index/main_mul.py`：多数据集RQVAE训练
- `index/generate_indices_distance.py`：代码生成脚本

### 论文引用
```bibtex
@inproceedings{mql4grec,
  title={Multimodal Quantitative Language for Generative Recommendation},
  booktitle={ICLR},
  year={2025}
}
```

### 模型引用

**LLaMA:**
```bibtex
@article{touvron2023llama,
  title={Llama: Open and efficient foundation language models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

**Transformers:**
```bibtex
@inproceedings{wolf2020transformers,
  title={Transformers: State-of-the-art natural language processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and others},
  booktitle={EMNLP System Demonstrations},
  year={2020}
}
```
