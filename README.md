# 基于联邦学习的隐私保护新闻推荐算法研究

本项目是一个面向毕业设计的实验型推荐系统仓库。项目基于 `MIND` 新闻推荐数据集，使用 `SBERT` 预计算新闻文本向量，结合 `Multi-Head Attention` 建模用户历史点击兴趣，并分别实现了集中式训练与基于 `Flower` 的联邦学习训练流程。

## 项目目标

- 构建一个新闻推荐基础模型
- 对比集中式训练与联邦训练的效果
- 在联邦训练流程中尝试加入初步的隐私保护机制
- 为毕业论文提供可复现实验代码

## 项目结构

```text
graduation_project/
|- baseline_centralized.py      # 集中式训练基线
|- dataset.py                   # 数据集封装与负采样
|- evaluation.py                # 离线评估脚本
|- federated_main.py            # Flower 联邦训练主程序
|- models.py                    # 用户编码器与推荐模型
|- preprocess/
|  |- preprocess_news.py        # 提取新闻文本向量
|  |- preprocess_behavior.py    # 构造联邦训练数据
|  |- preprocess_dev.py         # 构造验证集数据
|- tests/
|  |- test_dataset.py           # 数据管道自检
|  |- check_data.py             # 样本数据检查
|- dataset/                     # MINDsmall_train / MINDsmall_dev
|- processed/                   # 预处理产物
|- checkpoints/                 # 模型权重
```

## 环境要求

- Python 3.11+
- 建议使用独立虚拟环境
- GPU 非必需，但预处理 `SBERT` 向量时有 GPU 会明显更快

安装依赖：

```bash
pip install -r requirements.txt
```

## 最小启动说明

如果你只是想先把项目跑起来，而不是立刻重做全部实验，按下面顺序执行即可。

### 场景 A：仓库里已经有 `processed/` 和 `checkpoints/`

这种情况下可以直接做检查和评估：

```bash
pip install -r requirements.txt
python tests/check_data.py
python tests/test_dataset.py
python evaluation.py
```

适用场景：

- 想快速确认仓库是否可用
- 想直接查看现有模型的离线评估结果
- 暂时不重新做预处理和训练

### 场景 B：从原始数据开始完整跑通

确保 `dataset/` 下已经放好 `MINDsmall_train` 和 `MINDsmall_dev` 后，依次执行：

```bash
pip install -r requirements.txt
python preprocess/preprocess_news.py
python preprocess/preprocess_behavior.py
python preprocess/preprocess_dev.py
python baseline_centralized.py
python evaluation.py
```

如果还要跑联邦学习实验，再执行：

```bash
python federated_main.py
```

### 最短验证路径

如果你只想做一次“最小可执行验证”，推荐只跑下面三步：

```bash
pip install -r requirements.txt
python tests/check_data.py
python evaluation.py
```

这三步能回答三个最关键的问题：

- 依赖是否装齐
- 预处理产物是否存在且可读
- 已有模型是否能被加载并完成评估

### Windows 一键入口

如果你在 Windows 环境下操作，也可以直接运行仓库根目录下的批处理脚本：

```bat
run_quickstart.bat
```

用途：

- 检查已有预处理产物是否可读
- 检查数据集封装是否正常
- 直接评估现有模型

完整实验流程：

```bat
run_full_pipeline.bat
```

用途：

- 从预处理开始重新生成实验输入
- 训练集中式基线模型
- 评估模型
- 启动联邦学习实验

## 数据准备

仓库默认使用 `MINDsmall`，并约定目录如下：

```text
dataset/
|- MINDsmall_train/
|  |- news.tsv
|  |- behaviors.tsv
|- MINDsmall_dev/
   |- news.tsv
   |- behaviors.tsv
```

如果目录结构不一致，预处理脚本将无法找到数据文件。

## 运行顺序

### 1. 生成新闻向量

```bash
python preprocess/preprocess_news.py
```

输出文件：

- `processed/news_embeddings.npy`
- `processed/news_id_dict.pkl`

### 2. 生成联邦训练数据

```bash
python preprocess/preprocess_behavior.py
```

输出文件：

- `processed/federated_data.pkl`

如需生成 Non-IID 客户端划分数据，可额外执行：

```bash
python preprocess/preprocess_behavior_noniid.py
```

输出文件：

- `processed/federated_data_noniid.pkl`

### 3. 生成验证集

```bash
python preprocess/preprocess_dev.py
```

输出文件：

- `processed/dev_data_all.pkl`

### 4. 运行集中式基线

```bash
python baseline_centralized.py
```

输出文件：

- `checkpoints/centralized_attn_e5.pth` 或同类命名文件

命名规则：

- `centralized_attn_e5.pth`：表示集中式训练、使用注意力、训练 5 个 epoch
- `centralized_mean_e5.pth`：表示集中式训练、关闭注意力、训练 5 个 epoch

示例：

```bash
python baseline_centralized.py --epochs 5
python baseline_centralized.py --epochs 5 --disable-attention
```

### 5. 运行联邦训练

```bash
python federated_main.py
```

默认配置：

- `50` 个客户端
- 每轮采样比例 `0.2`
- `10` 轮联邦训练
- 聚合策略 `FedAvg`

默认输出命名示例：

- `checkpoints/fed_attn_nodp_r10_round_5.pth`
- `checkpoints/fed_attn_nodp_r10_round_10.pth`

命名规则：

- `attn` / `mean`：是否启用注意力用户编码器
- `nodp` / `dp0.001`：是否启用噪声注入及其规模
- `r10`：总联邦轮数
- `round_10`：当前保存的是第几轮模型

示例：

```bash
python federated_main.py --num-rounds 10
python federated_main.py --num-rounds 10 --sigma 0.001
python federated_main.py --num-rounds 10 --disable-attention
python federated_main.py --num-rounds 10 --data-file processed/federated_data_noniid.pkl
```

### 6. 评估模型

```bash
python evaluation.py
```

当前评估脚本默认读取：

- `checkpoints/centralized_model.pth`

推荐显式传入待评估模型路径与模型结构配置。

示例：

```bash
python evaluation.py --model-path checkpoints/centralized_attn_e5.pth --model-name centralized_attn_e5
python evaluation.py --model-path checkpoints/fed_attn_nodp_r10_round_10.pth --model-name fed_attn_nodp_r10_round_10
python evaluation.py --model-path checkpoints/fed_mean_nodp_r10_round_10.pth --model-name fed_mean_nodp_r10_round_10 --disable-attention
```

## 当前实现内容

### 模型

- 新闻表示：使用 `paraphrase-multilingual-MiniLM-L12-v2` 提取标题与摘要联合向量
- 用户表示：对历史点击新闻向量做 `Multi-Head Attention` 聚合
- 排序方式：候选新闻与用户向量做点积打分

### 联邦训练

- 框架：`Flower`
- 聚合方法：`FedAvg`
- 客户端构造方式：按用户随机切分为 50 组
- 隐私增强尝试：梯度裁剪与参数噪声注入

## 实验配置建议

如果你的论文需要展示“集中式基线 vs 联邦训练 vs 消融实验 vs DP 实验”，建议按下面方式组织：

- 集中式基线：`python baseline_centralized.py --epochs 5`
- 联邦主实验：`python federated_main.py --num-rounds 10`
- 注意力消融：`python federated_main.py --num-rounds 10 --disable-attention`
- DP 对比：`python federated_main.py --num-rounds 10 --sigma 0.001`

这样生成的 checkpoint 文件名会直接体现实验身份，后续写论文表格和答辩 PPT 时不容易混淆。

### 评估指标

- `Loss`
- `AUC`
- `MRR`

## 已有实验产物

仓库中已经存在部分预处理结果与模型权重，例如：

- `processed/news_embeddings.npy`
- `processed/federated_data.pkl`
- `processed/dev_data_all.pkl`
- `checkpoints/centralized_model.pth`
- `checkpoints/fed_model_round_10.pth`

这些文件说明项目已经完成过至少一轮预处理和训练实验。

## 已知限制

- 当前项目更偏实验代码，不是生产级推荐系统
- 联邦数据划分采用随机用户切分，尚未系统研究 Non-IID 场景
- 差分隐私实现仍是初步尝试，不等价于严格 DP 训练
- 测试目前主要用于自检，覆盖率还不高
- 评估脚本和联邦实验脚本仍有进一步整理空间

## 建议的复现检查

第一次接手仓库时，建议按下面顺序确认：

```bash
python -m py_compile baseline_centralized.py dataset.py evaluation.py federated_main.py models.py
python tests/check_data.py
python tests/test_dataset.py
```

如果缺少 `torch`、`flwr` 或 `sentence-transformers`，先安装依赖再继续。
