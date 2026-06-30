# agents.md — UNet Airfoil Flow Field Prediction

> **项目总览文档**
>
> 基于UNet的翼型可压缩流场机器学习推理方法
>
> 论文参考：《基于UNet的翼型可压缩流场机器学习推理方法》，朱智杰等，南京航空航天大学学报
>
> 项目路径：`/Users/origami/Desktop/Unet_wingfield/`

---

## 一、项目目标

构建一个基于深度 UNet 卷积神经网络的翼型可压缩流场快速推理系统，实现：

1. 输入翼型外形网格信息 + 来流条件（马赫数、攻角、雷诺数）→ 快速预测完整流场（密度、压力、速度）
2. 推理速度比传统 CFD 数值模拟快数个数量级（论文报告 ~12ms/样本）
3. 精度优于传统 CNN（FCN）方法，能准确捕捉激波、分离流等复杂流动现象

---

## 二、数据理解

### 2.1 数据来源

用户提供两类 CFD 计算数据：

| 数据集 | 文件数 | 说明 |
|--------|--------|------|
| `train_data_2822/` | 1,153 | 单一 RAE2822 翼型，多工况组合（马赫数×攻角×雷诺数） |
| `train_data_hybridwings/` | 341 | 341 种不同翼型，单一工况（Ma=0.50, AoA=3.00°, Re=5M） |

### 2.2 文件命名规则

格式：`{翼型名}(gen{迭代})_{Ma×100}_{AoA×100}_{Re/1000}.npz`

示例：`rae2822gen0_50_300_5000.npz` → RAE2822, Ma=0.50, AoA=3.00°, Re=5,000,000

### 2.3 数据格式

每个 `.npz` 文件存储一个 `(16, 128, 128)` float64 数组（key `"a"`），通道布局：

| 通道 | 内容 | 用途 |
|------|------|------|
| 0 | Ma∞ | 输入 — 来流条件 |
| 1 | AoA∞ | 输入 — 来流条件 |
| 2 | Re∞ | 输入 — 来流条件 |
| 3-11 | 网格度量 (J⁻¹, ξ̂_x, ξ̂_y, \|∇ξ\|/J, η̂_x, η̂_y, \|∇η\|/J, x₀, y₀) | 输入 — 坐标变换信息 |
| 12 | ρ (密度) | 输出 — 流场 |
| 13 | p (压力) | 输出 — 流场 |
| 14 | u (x方向速度) | 输出 — 流场 |
| 15 | v (y方向速度) | 输出 — 流场 |

输入输出切分：12 通道输入（3 IC + 9 网格度量）→ 4 通道输出（ρ, p, u, v）

### 2.4 工况分布 (train_data_2822)

- Ma: 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80
- AoA: −15° ~ 15°, 步长 1°
- Re: 3M, 4M, 5M, 6M, 6.5M, 7M, 8M

---

## 三、项目架构

### 目录结构

```
/Users/origami/Desktop/Unet_wingfield/
├── config/
│   └── train_config.py       # 配置管理（数据、模型、训练超参数）
├── data/
│   ├── preprocess.py         # 文件名解析、逐通道归一化、输入输出切分
│   └── dataset.py            # PyTorch Dataset + DataLoader
├── models/
│   ├── blocks.py             # 基础构件（DoubleConv, Down, Up, Bottleneck）
│   └── unet.py               # 深度 UNet（6层下采样+6层上采样+跳跃连接）
├── training/
│   ├── trainer.py            # 训练循环（L1 loss + AMP + checkpoint管理）
│   └── scheduler.py          # 学习率调度（StepLR）
├── inference/
│   └── predict.py            # 推理模块（加载模型→推理→逆归一化）
├── utils/
│   ├── metrics.py            # MSE, 相对L2误差, 误差图 e_map, Cp
│   └── visualization.py      # 论文式流场对比图、Cp曲线、训练曲线
├── scripts/
│   ├── train.py              # 训练入口（argparse）
│   ├── evaluate.py           # 评估入口
│   └── analyze_flow.py       # 论文式流场分析与可视化报告
├── results/models/           # CPU训练产出的checkpoints（epoch_010~050）
├── results_gen0/             # MPS训练产出的checkpoints与分析结果
├── data_metricGen/           # 原始CFD数据处理脚本
├── Unet_wingfield.md         # 详细项目文档
├── agents.md                 # 本文档
└── README.md                 # 快速入门
```

### 模块依赖流

```
用户
  │
  ├── scripts/train.py
  │     ├── config/train_config.py        → TrainConfig (dataclass)
  │     ├── data/dataset.py               → AirfoilDataset, create_dataloaders
  │     │     └── data/preprocess.py      → parse_filename, Normalizer, split_input_target
  │     ├── models/unet.py                → UNet, build_unet
  │     │     └── models/blocks.py        → DoubleConv, DownBlock, UpBlock, Bottleneck
  │     ├── training/trainer.py           → Trainer (train_epoch, validate, run)
  │     │     └── training/scheduler.py   → build_scheduler (StepLR)
  │     └── utils/visualization.py        → plot_training_history
  │
  ├── scripts/evaluate.py
  │     ├── config/train_config.py        → InferenceConfig
  │     ├── data/dataset.py               → AirfoilDataset
  │     ├── models/unet.py                → build_unet
  │     ├── utils/metrics.py              → evaluate_model
  │     └── utils/visualization.py        → plot_flow_field_comparison
  │
  └── scripts/analyze_flow.py
        ├── config/train_config.py        → TrainConfig
        ├── data/preprocess.py            → Normalizer, parse_filename, split_input_target
        ├── models/unet.py                → build_unet
        ├── utils/metrics.py              → summarize_flow_error, compute_pressure_coefficient
        └── utils/visualization.py        → plot_paper_flow_comparison, plot_pressure_error_map, plot_surface_cp_comparison
```

---

## 四、项目迭代历程

### Phase 1：论文理解与数据探索

- 阅读用户提供的 PDF 论文，理解 UNet 架构设计（6层深度）、坐标变换方法（9通道网格度量）、数据格式
- 分析 `.npz` 文件格式，确认 16 通道布局
- 发现 train_data_2822（单一翼型多工况）和 train_data_hybridwings（多翼型单工况）两种数据集

### Phase 2：项目骨架搭建

- 设计模块化项目结构，区分 config / data / models / training / inference / utils / scripts
- 使用 `dataclass` 管理配置（TrainConfig），支持属性自动计算（device_str, use_amp）
- 实现 `Normalizer` 类，逐通道 min-max 归一化到 [0,1]
- 实现 `AirfoilDataset`，按需加载 `.npz` 文件，自动切分输入输出

### Phase 3：UNet 模型实现

- 按照论文设计 6 层深度 UNet：
  - Entry: DoubleConv(12→32)
  - Encoder: DownBlock × 6（32→64→128→256→512→1024→2048）
  - Bottleneck: DoubleConv(2048→2048)
  - Decoder: UpBlock × 6（2048→1024→512→256→128→64→32）
  - Final: Conv2d(32→4, kernel=1)
- 参数总量：~200M（base_filters=32）
- 损失函数：L1 Loss（论文指定，对激波区域异常值更鲁棒）

**架构修正：** 初始实现中 Bottleneck 错误地将通道数从 2048 翻倍到 4096，导致 UpBlock 通道不匹配。修正为保持通道数不变。

### Phase 4：CPU 训练（v1）

- 首次训练在 CPU 上运行（代码只在 CUDA / CPU 之间选择，未检测 MPS）
- 运行了 ~6 小时，完成 50 个 epoch
- 最佳 Val Loss：0.00446（epoch 45）
- 速度：~260 秒/epoch（batch_size=16）→ 优化到 ~158 秒/epoch（batch_size=32）
- 总共约 14 小时才能完成 200 轮，效率极低

### Phase 5：MPS + AMP 加速（v2）

- 发现 MacBook M5 Pro 有 MPS（Metal Performance Shaders）GPU 后端
- 优化 `device_str` 检测逻辑：CUDA > MPS > CPU
- 添加 AMP（Automatic Mixed Precision）支持，float16 加速
- 配置 DataLoader 优化（num_workers, pin_memory 根据设备自动设置）
- 基准测试：CPU 158.7s/epoch → MPS+AMP **20.5s/epoch**（**7.7倍加速**）
- 完整 200 轮训练从 14 小时缩短到约 1 小时

### Phase 6：MPS 训练执行

- 第一次 MPS 训练 session 中断于 epoch 110（约 55%），保存了 110 个 checkpoint
- 最佳 Val Loss：0.00283（epoch 104），比 CPU 的 0.00446（epoch 45）好 37%
- 训练速度进一步优化到 12-18 秒/epoch

### Phase 7：论文式流场分析与可视化

- 阅读并对照论文第 2 节“结果和分析”：核心证据链为 CFD 真值、UNet 预测、归一化误差图 `e_map`、压力场 MSE、翼型表面/近壁压力系数 `Cp` 曲线
- 扩展 `utils/metrics.py`：新增逐通道 MSE、MAE、relative L2、max absolute error、误差汇总表，并修正 `e_map` 公式说明
- 扩展 `utils/visualization.py`：新增论文式三列面板（CFD / UNet / e_map）、压力误差图、近壁 Cp 曲线，保留原有基础对比图接口
- 新增 `scripts/analyze_flow.py`：支持指定样本或批量样本，自动加载 checkpoint 与 normalizer，完成推理、反归一化、指标计算、PNG 出图和 Markdown 报告
- 单样本验证：`rae2822gen0_50_300_5000.npz` 可成功生成 `flow_panel.png`、`pressure_error.png`、`cp_comparison.png`、`flow_metrics.csv`、`analysis_report.md`；其中压力通道 MSE≈6.98×10⁻⁶

---

## 五、关键设计决策

### 5.1 为什么用 L1 Loss 而不是 MSE？

L1 Loss 对异常值（激波附近的高梯度区域）惩罚更轻，有助于在激波和分离流区域保持更好的物理一致性。论文报告的收敛值为 (4.15±0.21)×10⁻⁴。

### 5.2 为什么选 6 层 UNet？

| 层数 | 最深层分辨率 | 参数量 | 感受野 |
|------|------------|--------|--------|
| 4 | 8×8 | ~30M | 有限 |
| 6 | 2×2 | ~200M | 全局 |
| 7 | 1×1 | >800M | 过大 |

6 层在参数量和感受野间取得最佳平衡。128×128 经过 6 次 2× 下采样到 2×2，编码器捕获全局流场上下文，解码器通过跳跃连接保留局部细节（附面层、激波等）。

### 5.3 为什么需要坐标变换？

CFD 网格是绕翼型的 C 型贴体网格，而 UNet 期望笛卡尔坐标输入。论文提出坐标变换将 (x,y) → (ξ,η) ∈ [0,1]²，变换矩阵的 9 个分量（J⁻¹, 法向量, 边长, 翼型表面坐标）作为额外输入通道，让网络隐式学习变换关系。

### 5.4 逐通道归一化

16 个通道的物理量纲差异巨大（ρ≈1, p 可达 13.7, 网格度量 0-6）。逐通道 min-max 归一化到 [0,1] 确保各通道对 loss 贡献均衡，防止压力通道主导训练。

### 5.5 训练策略

- **优化器：** AdamW（learning_rate=1e-3, weight_decay=1e-5）
- **学习率调度：** StepLR，每 50 epoch LR 减半（0.5）
- **批大小：** CPU 16→32，MPS 32（充分利用 GPU 内存）
- **数据划分：** 80% 训练 / 15% 验证 / 5% 测试
- **Checkpoint：** 每 epoch 保存最佳模型，每 10 epoch 保存完整状态

---

## 六、训练结果对比

### Loss 收敛

| 指标 | CPU (50 epoch) | MPS (110 epoch) |
|------|---------------|-----------------|
| 最佳 Val Loss | 0.00446 @ ep45 | **0.00283 @ ep104** |
| 最终 Train Loss | 0.00569 | **0.00379** |
| 训练时间 | ~6 小时 | **~33 分钟** |
| 每 epoch | ~260s | **~18s** |
| 加速比 | 1× | **~14×** |

### MPS 训练完整 Loss 轨迹

```
Epoch    Train Loss     Val Loss
─────────────────────────────────
    1     0.332          0.176
   10     0.0142         0.0150
   20     0.00688        0.00831
   30     0.00659        0.00913
   40     0.00606        0.00709
   50     0.00549        0.00596    ← LR 减半
   60     0.00405        0.00437
   70     0.00436        0.00630
   80     0.00410        0.00440
   90     0.00372        0.00422
  100     0.00357        0.00318
  104       —            0.00283    ← 最佳
  110     0.00379        0.00324
─────────────────────────────────
```

### 评估结果

在测试集上评估（epoch 45 最佳模型）：平均 L1 Loss = **0.00422**

---

## 七、代码模块详解

详见 `Unet_wingfield.md` 第4节（模块详解），包含每个模块的代码节选与设计思路说明。

核心模块概览：

- **config/train_config.py**（72行）：使用 dataclass 管理所有超参数，提供 device_str、use_amp、pin_memory 等计算属性
- **data/preprocess.py**（140行）：正则文件名解析 + Normalizer 类（逐通道 min-max 归一化 + 持久化）+ 输入输出切分
- **data/dataset.py**（110行）：AirfoilDataset 继承 torch Dataset，按需加载 .npz + create_dataloaders 统一入口
- **models/blocks.py**（102行）：DoubleConv（2×Conv-BN-LeakyReLU）、DownBlock（Pool+Conv）、UpBlock（TransposedConv+Skip+Conv）
- **models/unet.py**（110行）：UNet 类 + build_unet 工厂函数，6 层编解码
- **training/trainer.py**（130行）：Trainer 类（训练/验证循环 + AMP + checkpoint管理）
- **training/scheduler.py**（15行）：StepLR 构建函数
- **inference/predict.py**（85行）：Predictor 类（从 checkpoint 加载 + 推理 + 逆归一化）
- **utils/metrics.py**（165行）：MSE、MAE、相对 L2 误差、误差图 e_map、Cp 计算、逐通道误差汇总
- **utils/visualization.py**（268行）：流场对比图、论文式 CFD/UNet/e_map 三列图、压力误差图、Cp 曲线、训练曲线
- **scripts/train.py**（80行）：argparse 入口，支持 --data-dir / --epochs / --batch-size / --save-dir / --device
- **scripts/evaluate.py**（98行）：argparse 入口，加载模型 → 评估 → 生成基础可视化
- **scripts/analyze_flow.py**（329行）：论文式流场分析入口，加载模型 → 推理 → 反归一化 → 指标表 → PNG 图组 → Markdown 报告

---

## 八、训练流程

```
train_data_2822/*.npz (1153 files)
    │
    ▼
Normalizer.fit() — 逐通道 min/max，保存 normalizer.npz
    │
    ▼
split_input_target() — (N,16,128,128) → input(N,12) + target(N,4)
    │
    ▼
random_split() — 80% Train(924) / 15% Val(172) / 5% Test(57)
    │
    ▼
UNet 前向传播 — 12ch → Encoder×6 → Bottleneck → Decoder×6 → 4ch
    │
    ▼
L1 Loss → backward() → AdamW.step() → StepLR.step()
    │
    ▼ (每 epoch)
best_model.pth (val_loss 降低时保存)
    │
    ▼ (每 10 epoch)
epoch_XXX.pth (完整状态：模型+优化器+调度器+历史)
```

---

## 九、运行指南

### 训练（MPS/GPU 加速）

```bash
cd /Users/origami/Desktop/Unet_wingfield
python3 scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 32 --save-dir results_gen0
```

### 评估

```bash
python3 scripts/evaluate.py --data-dir train_data_2822
```

### 论文式流场分析与可视化

单样本分析：

```bash
python3 scripts/analyze_flow.py \
  --sample rae2822gen0_50_300_5000.npz \
  --output-dir results_gen0/flow_analysis \
  --extent -0.25 1.25 -0.4 1.1
```

批量分析前 3 个样本：

```bash
python3 scripts/analyze_flow.py \
  --max-samples 3 \
  --output-dir results_gen0/flow_analysis
```

输出内容：

- `flow_panel.png`：4 个流场通道的 CFD 真值 / UNet 预测 / `e_map` 三列对比
- `pressure_error.png`：压力通道归一化误差图，标注压力 MSE
- `cp_comparison.png`：近壁上/下采样线压力系数曲线
- `flow_metrics.csv`：逐通道 MSE、MAE、relative L2、最大绝对误差和数值范围
- `analysis_report.md`：样本工况、指标表和图像汇总

### Python API 推理

```python
from inference.predict import Predictor
predictor = Predictor.from_checkpoint("results_gen0/best_model.pth", "results_gen0/normalizer.npz")
flow_pred = predictor.predict_and_denormalize(raw_data)  # (1, 4, 128, 128)
```

---

## 十、已知问题与改进方向

### 已知问题

1. **Session 超时：** exec_command session 有输出超时限制，长时间无输出（log_interval=10）的 session 可能被自动终止
2. **数据集内存：** Normalizer 的 `_load_all()` 一次性加载 1153 个文件（~4.8GB），可能有内存压力；已提供 `RunningMinMax` 增量式方案
3. **Python 3.9 兼容：** 部分代码使用 `X | None` 语法（Python 3.10+），在 Mac 系统默认 Python 3.9 上不兼容
4. **Cp 曲线采样：** 当前数据未提供独立翼型壁面 mask 或表面索引，`analyze_flow.py` 默认用 C 型贴体网格中线附近两条近壁采样线估算 Cp，可通过 `--surface-axis` / `--surface-index` / `--surface-offset` 调整

### 改进方向

1. **多翼型训练：** 当前仅使用 RAE2822 单一翼型数据，可合并 train_data_hybridwings（341 翼型）进行泛化性训练
2. **模型压缩：** base_filters=32 产生 200M 参数，降至 16 可得 ~50M 模型，速度提升 3-4×
3. **Resume 训练：** 当前不支持从 checkpoint 恢复，可扩展从 epoch_XXX.pth 加载继续训练
4. **num_workers：** 默认 4 可能导致 multiprocessing 问题，数字小或为 0 更稳定
5. **结果可视化：** 已新增论文式分析脚本；后续可进一步加入翼型壁面真实索引、局部激波/分离区域自动框选、批量样本排序和 HTML 报告

---

*文档生成日期：2026-06-30*
*项目路径：`/Users/origami/Desktop/Unet_wingfield/`*
*生成工具：Codex (GPT-5)*
