# MQL4GRec 图片处理完整 Pipeline

本文档详细介绍 MQL4GRec 中图片处理的完整流程，从原始图片嵌入到最终的推荐生成。

---

## 概览

MQL4GRec 使用 **Residual Quantized Variational AutoEncoder (RQVAE)** 将连续的图片嵌入向量转换为离散的量化代码（quantized codes），然后将这些代码作为 token 输入到 T5 生成模型中进行推荐。

**核心思想**：将图片表示为一串离散的 token（如 `<A_123><B_45><C_67><D_89>`），使得 T5 模型可以像处理文本一样处理图片信息。

---

## Pipeline 三大阶段

### 阶段 1: 图片嵌入提取（预处理）

**目标**：将原始图片转换为连续的嵌入向量

#### 1.1 输入数据
- 原始图片文件（来自推荐数据集中的商品图片）

#### 1.2 图片编码器
使用预训练的视觉模型提取图片特征：
- **模型**: ViT-L-14（Vision Transformer Large）
- **输出维度**: 通常为 768 或更高维度的嵌入向量

#### 1.3 数据存储格式
生成的嵌入向量存储为 `.npy` 文件：
```
/data_root/
  ├── Dataset1/
  │   └── Dataset1.emb-ViT-L-14.npy    # 图片嵌入矩阵 [num_items, embedding_dim]
  ├── Dataset2/
  │   └── Dataset2.emb-ViT-L-14.npy
  ...
```

**关键文件**: `<dataset>.emb-ViT-L-14.npy`
- 形状: `[num_items, 768]`（假设 ViT-L-14 输出 768 维）
- 每一行对应一个商品的图片嵌入

---

### 阶段 2: 量化翻译器训练（Quantitative Translator Training）

**目标**：训练 RQVAE 模型，将连续嵌入映射为离散代码

#### 2.1 运行脚本
```bash
cd index
bash scripts/run.sh
```

脚本内容示例（针对图片嵌入）：
```bash
Model=ViT-L-14
Code_num=256
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'

python -u main_mul.py \
  --num_emb_list 256 256 256 256 \           # 4层码本，每层256个代码
  --sk_epsilons 0.0 0.0 0.0 0.003 \          # Sinkhorn-Knopp 正则化参数
  --device cuda:0 \
  --data_root /path/to/embeddings/ \
  --embedding_file .emb-ViT-L-14.npy \       # 图片嵌入文件后缀
  --datasets $Datasets \
  --ckpt_dir log/$Datasets/${Model}_${Code_num} \
  --batch_size 2048 \
  --epochs 500
```

#### 2.2 RQVAE 模型架构

**核心组件**（定义在 `index/models/rqvae.py:9`）：

1. **Encoder**（编码器）
   - 多层感知机（MLP）
   - 输入维度: 768（ViT-L-14 嵌入维度）
   - 隐藏层: `[2048, 1024, 512, 256, 128, 64]`
   - 输出维度: 32（`e_dim`，量化嵌入维度）

2. **Residual Vector Quantizer**（残差向量量化器）
   - 4 层量化（对应 4 个 codebook）
   - 每层码本大小: 256（`num_emb_list = [256, 256, 256, 256]`）
   - 量化过程使用 **Sinkhorn-Knopp 算法**优化代码分配
   - 输出: 4 个离散索引 `[idx_A, idx_B, idx_C, idx_D]`，每个范围 0-255

3. **Decoder**（解码器）
   - 与 Encoder 对称的 MLP
   - 重建原始 768 维嵌入向量

#### 2.3 训练过程

**损失函数**（`index/models/rqvae.py:73`）：
```python
loss_total = loss_recon + quant_loss_weight * quant_loss
```
- **loss_recon**: 重建损失（MSE），确保解码后能恢复原始嵌入
- **quant_loss**: 量化损失，使编码器输出接近码本嵌入

**关键参数**：
- `sk_epsilons`: Sinkhorn-Knopp 的 epsilon 参数，控制每层量化的平滑度
  - 前几层设为 0.0（硬量化）
  - 最后一层设为 0.003（软量化，增加多样性）

**输出模型**：
- 保存在 `log/<Datasets>/ViT-L-14_256/best_collision_model.pth`
- 包含训练好的 RQVAE 权重和配置

#### 2.4 评估指标

训练过程监控两个指标：
1. **重建损失 (Reconstruction Loss)**: 越低越好
2. **碰撞率 (Collision Rate)**: 不同商品被映射到相同代码的比例，越低越好

---

### 阶段 3: 生成量化代码（Code Generation）

**目标**：使用训练好的 RQVAE 为每个商品生成唯一的离散代码序列

#### 3.1 运行脚本
```bash
cd index
bash scripts/gen_code_dis.sh
```

脚本内容示例：
```bash
Dataset=Instruments

# 生成图片代码
python -u generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path log/$Dataset/ViT-L-14_256/best_collision_model.pth \
  --output_dir ./data/$Dataset \
  --output_file ${Dataset}.index_vitemb.json \
  --content image                              # 指定为图片模态
```

#### 3.2 代码生成流程

**关键文件**: `index/generate_indices_distance.py`

##### 3.2.1 基本生成过程

```python
# 加载 RQVAE 模型
model = RQVAE(...)
model.load_state_dict(checkpoint['state_dict'])

# 对每个商品的图片嵌入进行量化
for embeddings in data_loader:
    indices, distances = model.get_indices(embeddings, use_sk=False)
    # indices: [batch_size, 4] - 4个离散索引
    # distances: [batch_size, 4, 256] - 到每个码本中每个代码的距离
```

##### 3.2.2 碰撞检测与解决

RQVAE 可能将不同商品映射到相同代码（碰撞）。系统使用距离信息解决碰撞：

**碰撞解决算法**（`index/generate_indices_distance.py:158`）：

1. **检测碰撞**：找到所有映射到相同代码的商品组
2. **对每个碰撞组**：
   - 计算每个商品到其代码的最小距离
   - 距离最小的商品保留原代码
   - 其他商品尝试替代代码
3. **替代代码选择**（优先级从高到低）：
   - 修改第 4 层代码（最后一层）：选择距离第 2 近的代码
   - 如果仍碰撞，继续尝试距离第 3、4、5...近的代码
   - 如果第 4 层所有代码都碰撞，修改第 3 层代码，重新尝试第 4 层
4. **迭代**：重复 2-3 直到无碰撞或达到最大迭代次数

#### 3.3 Token 前缀映射

根据 `--content` 参数确定 token 前缀（`index/generate_indices_distance.py:66`）：

```python
if args.content == 'image':
    prefix = ["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>"]  # 图片代码
else:
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]  # 文本代码
```

**示例转换**：
- 原始索引: `[123, 45, 67, 89]`
- 图片 token 序列: `["<A_123>", "<B_45>", "<C_67>", "<D_89>"]`

#### 3.4 输出文件格式

**文件**: `<dataset>.index_vitemb.json`（图片代码）

结构：
```json
{
  "0": ["<A_123>", "<B_45>", "<C_67>", "<D_89>"],
  "1": ["<A_200>", "<B_12>", "<C_150>", "<D_3>"],
  "2": ["<A_5>", "<B_78>", "<C_230>", "<D_99>"],
  ...
}
```
- 键: 商品 ID（字符串格式）
- 值: 4 个图片代码 token 的列表

**对应的文本代码文件**: `<dataset>.index_lemb.json`（使用相同流程生成，但用 LLaMA 嵌入和小写前缀）

---

## 阶段 4: 在 T5 模型中使用图片代码

### 4.1 Tokenizer 扩展

在预训练/微调阶段，将量化代码 token 添加到 T5 tokenizer 词汇表：

**代码位置**: `data.py:74`

```python
def get_all_tokens(self):
    prefix_list = [
        "<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>",  # 文本代码
        "<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>"   # 图片代码
    ]

    new_tokens = set()
    for prefix in prefix_list:
        for i in range(256):  # code_num = 256
            token = prefix.format(i)
            new_tokens.add(token)

    # 生成 256*8 = 2048 个新 token
    return sorted(list(new_tokens))
```

然后调整模型词汇表：
```python
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

### 4.2 数据加载

**多模态数据集示例**（`data.py:439`）：

```python
class FusionSeqRecDataset:
    def _load_data(self):
        # 加载交互序列
        self.inters = json.load(open(f"{dataset}.inter.json"))
        # 加载文本代码
        self.indices = json.load(open(f"{dataset}.index_lemb.json"))
        # 加载图片代码
        self.image_indices = json.load(open(f"{dataset}.index_vitemb.json"))

    def __getitem__(self, index):
        # 历史序列使用文本代码
        history_text = ["".join(self.indices[str(i)]) for i in user_history]
        # 目标使用图片代码
        target_image = "".join(self.image_indices[str(target_item)])

        input = prompt + "".join(history_text)
        output = target_image
        return {"input_ids": input, "labels": output}
```

### 4.3 训练任务类型

与图片相关的任务（`pretrain.py` / `finetune.py`）：

1. **seqimage**: 图片序列推荐
   - 输入: 历史商品的图片代码序列
   - 输出: 下一个商品的图片代码

2. **item2image**: 文本到图片翻译
   - 输入: 商品的文本代码
   - 输出: 商品的图片代码

3. **image2item**: 图片到文本翻译
   - 输入: 商品的图片代码
   - 输出: 商品的文本代码

4. **fusionseqrec**: 融合序列推荐
   - 输入: 混合文本和图片代码的历史序列
   - 输出: 目标商品代码（文本或图片）

5. **fgfusionseqrec**: 细粒度融合推荐
   - 为文本和图片模态使用不同的 soft prompts

### 4.4 推理与约束生成

在测试阶段，使用 Trie 结构约束生成，确保只生成有效的图片代码序列：

**代码位置**: `data.py:138`

```python
def get_prefix_allowed_tokens_fn_new(self, tokenizer):
    # 构建允许的 token 转移图
    self.allowed_tokens = defaultdict(set)
    for image_code in self.image_indices.values():
        # image_code: ["<A_123>", "<B_45>", "<C_67>", "<D_89>"]
        token_ids = [tokenizer(tok)["input_ids"][1] for tok in image_code]

        for i in range(len(token_ids) - 1):
            self.allowed_tokens[token_ids[i]].add(token_ids[i+1])

    def prefix_allowed_tokens_fn(batch_id, sentence):
        # 只允许生成有效的下一个 token
        last_token = sentence[-1]
        return list(self.allowed_tokens[last_token])

    return prefix_allowed_tokens_fn
```

在测试时调用：
```python
outputs = model.generate(
    input_ids=inputs,
    num_beams=20,
    prefix_allowed_tokens_fn=dataset.get_prefix_allowed_tokens_fn_new(tokenizer)
)
```

这确保模型只生成数据集中存在的图片代码序列。

---

## 完整数据流示例

### 输入：商品图片
```
Item 0: /images/product_0.jpg
Item 1: /images/product_1.jpg
...
```

### 步骤 1: 提取图片嵌入
```python
# 使用 ViT-L-14
embeddings = vit_model.encode(images)
# embeddings.shape: [num_items, 768]
np.save("Dataset.emb-ViT-L-14.npy", embeddings)
```

### 步骤 2: 训练 RQVAE
```bash
python main_mul.py --embedding_file .emb-ViT-L-14.npy ...
# 输出: log/Dataset/ViT-L-14_256/best_collision_model.pth
```

### 步骤 3: 生成离散代码
```bash
python generate_indices_distance.py --content image ...
# 输出: Dataset.index_vitemb.json
```

生成的代码示例：
```json
{
  "0": ["<A_123>", "<B_45>", "<C_67>", "<D_89>"],
  "1": ["<A_200>", "<B_12>", "<C_150>", "<D_3>"]
}
```

### 步骤 4: 构建推荐训练数据

用户交互序列：
```json
{
  "user_1": [0, 1, 5, 8, 12],
  "user_2": [3, 7, 1, 9]
}
```

转换为训练样本（seqimage 任务）：
```
Input:  <A_123><B_45><C_67><D_89>                    # Item 0 的图片代码
Output: <A_200><B_12><C_150><D_3>                    # Item 1 的图片代码

Input:  <A_123><B_45><C_67><D_89><A_200><B_12><C_150><D_3>
Output: <A_50><B_100><C_200><D_10>                   # Item 5 的图片代码
```

### 步骤 5: T5 训练
```python
# T5 encoder 处理输入图片代码序列
encoder_outputs = t5.encoder(tokenizer(input_sequence))

# T5 decoder 生成下一个图片代码
predictions = t5.decoder(encoder_outputs)
```

### 步骤 6: 推理与解码

给定历史：`[Item 0, Item 1]`

```python
input = "<A_123><B_45><C_67><D_89><A_200><B_12><C_150><D_3>"
predicted_code = model.generate(input, num_beams=20)
# predicted_code: "<A_50><B_100><C_200><D_10>"

# 解码为商品 ID
predicted_item = image_code_to_item_id[predicted_code]
# predicted_item: 5
```

---

## 关键文件总结

| 文件路径 | 功能 |
|---------|------|
| `index/main_mul.py` | RQVAE 训练主程序（多数据集） |
| `index/models/rqvae.py` | RQVAE 模型定义 |
| `index/models/rq.py` | 残差向量量化器实现 |
| `index/generate_indices_distance.py` | 生成离散代码并解决碰撞 |
| `index/scripts/run.sh` | RQVAE 训练脚本（文本和图片） |
| `index/scripts/gen_code_dis.sh` | 代码生成脚本 |
| `data.py` | 数据集类，处理多模态代码加载 |
| `pretrain.py` | T5 预训练（多任务） |
| `finetune.py` | T5 微调 |
| `test.py` / `test_ddp.py` | 推理和评估 |

---

## 配置参数详解

### RQVAE 训练参数

```bash
--num_emb_list 256 256 256 256     # 4层码本，每层256个代码（总代码空间: 256^4）
--sk_epsilons 0.0 0.0 0.0 0.003    # Sinkhorn-Knopp 参数，最后一层使用软量化
--e_dim 32                         # 量化嵌入维度
--layers 2048 1024 512 256 128 64  # Encoder/Decoder 隐藏层尺寸
--embedding_file .emb-ViT-L-14.npy # 图片嵌入文件后缀
--batch_size 2048                  # 训练批大小
--epochs 500                       # 训练轮数
--eval_step 2                      # 每2轮评估一次
```

### T5 训练参数（图片任务）

```bash
--tasks seqimage,item2image,image2item     # 图片相关任务
--image_index_file .index_vitemb.json      # 图片代码文件
--code_num 256                             # 每层码本大小
--max_his_len 20                           # 最大历史序列长度
```

---

## 高级话题

### 1. 为什么使用 4 层残差量化？

- **单层量化问题**: 256 个代码不足以表示所有商品（通常有数千个）
- **4 层残差量化**: 理论代码空间 = 256^4 = 4,294,967,296，远超商品数量
- **残差设计**: 每层修正前一层的量化误差，提高重建质量

### 2. Sinkhorn-Knopp 算法的作用

- **问题**: 标准 VQ 可能导致代码利用不均（某些代码过度使用，某些从不使用）
- **解决**: SK 算法强制代码分配接近均匀分布
- **参数**: `epsilon` 控制约束强度
  - `epsilon = 0.0`: 硬约束（严格均匀）
  - `epsilon > 0.0`: 软约束（允许一定偏差）

### 3. 碰撞率优化策略

生成代码时的碰撞解决确保：
1. **唯一性**: 每个商品有唯一代码
2. **最小扰动**: 优先修改最后一层（影响最小）
3. **语义保持**: 使用距离第二近的代码，保持嵌入空间邻近性

### 4. 多模态融合机制

系统支持三种融合策略：

1. **Early Fusion**: 在输入层混合文本和图片代码
   ```
   Input: <a_1><b_2><c_3><d_4><A_5><B_6><C_7><D_8>...
   ```

2. **Late Fusion**: 分别训练文本和图片任务，推理时集成
   ```python
   score = alpha * text_score + (1-alpha) * image_score
   ```

3. **Fine-Grained Fusion** (fgfusionseqrec): 为每种模态使用专门的 soft prompts

---

## 常见问题

### Q1: 图片代码和文本代码有什么区别？
**A**:
- **训练数据不同**: 图片代码来自 ViT 嵌入，文本代码来自 LLaMA 嵌入
- **Token 前缀不同**: 图片用大写 `<A_>`, `<B_>` 等，文本用小写 `<a_>`, `<b_>` 等
- **语义空间不同**: 图片代码捕获视觉特征，文本代码捕获语义特征

### Q2: 为什么需要碰撞检测？
**A**: RQVAE 是有损压缩，将高维连续空间（768维）映射到离散空间（4个索引），信息损失不可避免。不同商品可能被映射到相同代码。碰撞解决算法通过选择次优但唯一的代码确保商品可区分性。

### Q3: 如何调整代码空间大小？
**A**: 修改 `--num_emb_list` 参数。例如：
- `256 256 256 256`: 4 层，每层 256 个代码（当前配置）
- `512 512 512 512`: 4 层，每层 512 个代码（更大空间，需要更多内存）
- `256 256 256`: 3 层（更小空间，可能碰撞率更高）

### Q4: 能否可视化量化代码的语义？
**A**: 可以！通过 t-SNE 或 UMAP 降维码本嵌入：
```python
from sklearn.manifold import TSNE
embeddings = model.rq.layers[0].embeddings.weight.detach().cpu().numpy()
tsne = TSNE(n_components=2).fit_transform(embeddings)
# 绘制 256 个代码的 2D 分布
```

---

## 总结

MQL4GRec 的图片处理 pipeline 实现了一个完整的 **图片离散化 → 序列建模 → 推荐生成** 流程：

1. **预处理**: ViT-L-14 提取图片嵌入 (768维)
2. **量化**: RQVAE 将嵌入转换为 4 个离散索引（每个 0-255）
3. **Token化**: 索引映射为 T5 token（如 `<A_123>`）
4. **训练**: T5 学习图片代码序列的生成模式
5. **推理**: 约束生成确保输出有效的图片代码序列

这种设计使得 T5 模型能够统一处理文本和图片两种模态，实现真正的多模态推荐。

---

**相关论文**: "Multimodal Quantitative Language for Generative Recommendation" (ICLR 2025)
