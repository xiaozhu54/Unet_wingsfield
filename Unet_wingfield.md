# UNet Airfoil Compressible Flow Field Prediction — 项目文档

> **论文参考：** *基于UNet的翼型可压缩流场机器学习推理方法*，朱智杰、赵国庆、高远、招启军，南京航空航天大学学报
>
> **项目路径：** `/Users/origami/Desktop/Unet_wingfield/`

---

## 1. 项目概述

本项目基于 UNet 卷积神经网络，实现二维翼型可压缩流场的快速高精度推理。核心任务是将 CFD 数值模拟计算得到的翼型周围流场（压力、密度、速度分布）作为监督信号，训练一个深度 UNet 模型，使其能够在给定翼型外形、来流马赫数、攻角和雷诺数的情况下，快速预测完整的流场分布。

### 论文核心贡献

| 贡献 | 说明 |
|------|------|
| **坐标变换方法** | 将 CFD 计算空间 (x, y) 映射到神经网络空间 (ξ, η)，使流场数据分布更均匀，提升 UNet 训练效果 |
| **深度 UNet 架构** | 6 层下采样 + 6 层上采样，跳跃连接保留多尺度特征，专门用于捕捉激波、分离流等复杂流动现象 |
| **快速推理模型** | 训练完成后，单次流场推理仅需 ~12ms（FCN 对比为 88ms），且精度优于传统 CNN 方法 |

---

## 2. 数据理解

### 2.1 数据来源

- **train_data_2822/** — 1,153 个 `.npz` 文件，单一 RAE2822 翼型，多工况组合
- **train_data_hybridwings/** — 341 个 `.npz` 文件，341 种不同翼型，单一工况

### 2.2 文件名编码

```
{翼型名}(gen{迭代})_{Ma×100}_{AoA×100}_{Re/1000}.npz
```

| 示例 | Ma | AoA | Re |
|------|----|-----|----|
| `rae2822gen0_50_300_5000` | 0.50 | 3.00° | 5,000,000 |
| `rae2822gen0_55_-1000_3000` | 0.55 | -10.00° | 3,000,000 |
| `naca0012_50_300_5000` | 0.50 | 3.00° | 5,000,000 |

解析代码（`data/preprocess.py`）：

```python
_FILENAME_PATTERN = re.compile(r"(.+?)(?:gen\d+)?_(\d+)_(-?\d+)_(\d+)\.npz")

def parse_filename(filename: str):
    m = _FILENAME_PATTERN.match(filename)
    name = m.group(1)
    ma = int(m.group(2)) / 100.0
    aoa = int(m.group(3)) / 100.0
    re_val = int(m.group(4)) * 1000
    return name, ma, aoa, re_val
```

### 2.3 数据格式

每个 `.npz` 文件包含一个 `(16, 128, 128)` 的 float64 数组（key `"a"`），16 个通道的布局对应论文中的输入/输出定义：

| 通道 | 内容 | 域 | 说明 |
|------|------|----|------|
| 0 | Ma∞ | **输入 — IC** | 来流马赫数（全场常数） |
| 1 | AoA∞ | **输入 — IC** | 攻角（全场常数） |
| 2 | Re∞ | **输入 — IC** | 雷诺数（全场常数） |
| 3 | J⁻¹ | **输入 — 网格** | 坐标变换雅可比逆 |
| 4 | ξ̂_x | **输入 — 网格** | i 边单位法向量 x 分量 |
| 5 | ξ̂_y | **输入 — 网格** | i 边单位法向量 y 分量 |
| 6 | \|∇ξ\|/J | **输入 — 网格** | i 边长度 |
| 7 | η̂_x | **输入 — 网格** | j 边单位法向量 x 分量 |
| 8 | η̂_y | **输入 — 网格** | j 边单位法向量 y 分量 |
| 9 | \|∇η\|/J | **输入 — 网格** | j 边长度 |
| 10 | x₀ | **输入 — 网格** | 翼型表面 x 坐标 |
| 11 | y₀ | **输入 — 网格** | 翼型表面 y 坐标 |
| 12 | ρ | **输出 — 流场** | 密度场 |
| 13 | p | **输出 — 流场** | 压力场 |
| 14 | u | **输出 — 流场** | x 方向速度场 |
| 15 | v | **输出 — 流场** | y 方向速度场 |

**输入/输出切分逻辑**（`data/preprocess.py`）：

```python
def split_input_target(raw_data, config):
    ic = raw_data[:, config.ic_channels]      # ch[0-2]
    grid = raw_data[:, config.grid_channels]  # ch[3-11]
    flow = raw_data[:, config.flow_channels]  # ch[12-15]
    inputs = np.concatenate([ic, grid], axis=1)  # → (N, 12, H, W)
    return inputs, flow                          # → (N, 4, H, W)
```

### 2.4 工况分布（train_data_2822）

| 参数 | 范围 | 步长 |
|------|------|------|
| Ma | 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80 | — |
| AoA | −15° ~ 15° | 1° |
| Re | 3M, 4M, 5M, 6M, 6.5M, 7M, 8M | — |

---

## 3. 项目架构

### 3.1 目录结构

```
Unet_wingfield/
├── config/
│   └── train_config.py         训练配置（数据、模型、训练超参数）
├── data/
│   ├── preprocess.py           文件名解析、归一化、输入输出切分
│   └── dataset.py              PyTorch Dataset / DataLoader
├── models/
│   ├── blocks.py               基础构件（DoubleConv, Down, Up）
│   └── unet.py                 深度 UNet 网络架构
├── training/
│   ├── trainer.py              训练循环、checkpoint 管理
│   └── scheduler.py            学习率调度器
├── inference/
│   └── predict.py              推理模块（加载模型 → 预测 → 逆归一化）
├── utils/
│   ├── metrics.py              评估指标（MSE, e_map, Cp）
│   └── visualization.py        可视化（流场对比图、Cp 曲线、训练曲线）
├── scripts/
│   ├── train.py                训练入口脚本
│   └── evaluate.py             评估入口脚本
├── results/models/             模型权重和归一化参数存储
├── requirements.txt
├── README.md
└── Unet_wingfield.md           本文档
```

### 3.2 模块依赖与数据流

```
用户输入 (scripts/train.py)
    │
    ├── config/train_config.py        ← 配置管理
    │
    ├── data/preprocess.py            ← 文件名解析 + 归一化参数
    │       └── parse_filename()
    │       └── Normalizer.fit() / normalize() / denormalize()
    │       └── split_input_target()
    │
    ├── data/dataset.py               ← 数据加载
    │       └── AirfoilDataset.__getitem__()
    │       └── create_dataloaders()
    │
    ├── models/blocks.py              ← 神经网络构件
    │       └── DoubleConv / DownBlock / UpBlock / Bottleneck
    │
    ├── models/unet.py                ← UNet 模型
    │       └── UNet (12ch → 4ch)
    │       └── build_unet()
    │
    ├── training/trainer.py           ← 训练引擎
    │       └── Trainer.train_epoch()
    │       └── Trainer.validate()
    │       └── Trainer.run()
    │
    └── utils/visualization.py        ← 结果可视化
            └── plot_flow_field_comparison()
            └── plot_training_history()
```

---

## 4. 模块详解

### 4.1 配置管理 `config/train_config.py`

使用 Python `dataclass` 管理所有超参数，提供运行时属性（如 `device_str`）和参数验证。

```python
@dataclass
class TrainConfig:
    input_channels: int = 12       # 3 IC + 9 网格度量
    output_channels: int = 4       # ρ, p, u, v
    base_filters: int = 32         # UNet 基础通道数
    n_down: int = 6                # 下采样/上采样层数（论文值）
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3
    lr_scheduler_step: int = 50    # StepLR 步长
    lr_scheduler_gamma: float = 0.5  # 每步 LR 减半

    @property
    def device_str(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
```

通道索引采用 `List[int]` 显式指定，支持在配置层面控制输入/输出的通道切分方式：

```python
ic_channels: List[int] = field(default_factory=lambda: [0, 1, 2])
grid_channels: List[int] = field(default_factory=lambda: list(range(3, 12)))
flow_channels: List[int] = field(default_factory=lambda: [12, 13, 14, 15])
```

---

### 4.2 数据预处理 `data/preprocess.py`

#### 文件名解析

正则表达式从文件名中提取翼型名称、马赫数、攻角和雷诺数：

```python
_FILENAME_PATTERN = re.compile(r"(.+?)(?:gen\d+)?_(\d+)_(-?\d+)_(\d+)\.npz")
# "rae2822gen0_50_-100_3000" → ("rae2822", 0.50, -1.00, 3_000_000)
```

#### 逐通道 Min-Max 归一化

```python
class Normalizer:
    def normalize(self, data: np.ndarray) -> np.ndarray:
        out = data.copy().astype(np.float32)
        for c in range(data.shape[1]):
            lo, hi = self.channel_mins[c], self.channel_maxs[c]
            span = hi - lo
            out[:, c] = (data[:, c] - lo) / span if span > 1e-12 else 0.0
        return out
```

归一化参数通过 `np.savez_compressed` 持久化到 `normalizer.npz`，评估/推理时由 `Normalizer.load()` 恢复。

**设计考虑：** 早期版本用 `_load_all()` 一次性加载全部 1,153 个文件来计算 min/max（约 4.8 GB 内存），会导致内存压力。因此提供了 `RunningMinMax` 辅助类（后续可增量式拟合）：

```python
class RunningMinMax:
    def update(self, data: np.ndarray):
        sample_min = data.min(axis=(1, 2))
        sample_max = data.max(axis=(1, 2))
        np.minimum(self.mins, sample_min, out=self.mins)
        np.maximum(self.maxs, sample_max, out=self.maxs)
```

---

### 4.3 数据集 `data/dataset.py`

`AirfoilDataset` 继承 `torch.utils.data.Dataset`，每个样本对应一个 `.npz` 文件：

```python
class AirfoilDataset(Dataset):
    def __getitem__(self, idx):
        raw = np.load(f"{self.data_dir}/{self.files[idx]}")["a"]  # (16, H, W)
        if self.normalizer:
            raw = self.normalizer.normalize(raw[np.newaxis])[0]
        inputs, targets = split_input_target(raw[np.newaxis], self.config)
        return torch.from_numpy(inputs[0]), torch.from_numpy(targets[0])
```

`create_dataloaders()` 完成三个关键步骤：

1. **拟合归一化器** — 遍历所有数据计算 min/max
2. **数据集划分** — `random_split` 按 80% / 15% / 5% 切分为 train / val / test
3. **构建 DataLoader** — 指定 batch_size、shuffle、pin_memory

```python
def create_dataloaders(data_dir, config) -> Tuple[DataLoader, DataLoader, DataLoader | None, Normalizer]:
    temp_dataset = AirfoilDataset(data_dir, config=cfg)
    normalizer = Normalizer()
    all_data = temp_dataset._load_all()
    normalizer.fit(all_data)
    full_dataset = AirfoilDataset(data_dir, normalizer=normalizer, config=cfg)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, ...)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, ...)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, ...)
    return train_loader, val_loader, test_loader, normalizer
```

---

### 4.4 模型构件 `models/blocks.py`

四个基础模块构成 UNet 的积木：

#### DoubleConv

两个连续的 `Conv2D → BatchNorm → LeakyReLU` 块：

```
输入 → Conv2D(3×3) → BN → LeakyReLU → Conv2D(3×3) → BN → LeakyReLU → 输出
```

```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_bn=True, dropout=0.0):
        layers = []
        for i in range(2):
            _in = in_ch if i == 0 else out_ch
            layers.append(nn.Conv2d(_in, out_ch, kernel_size, padding=1, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)
```

#### DownBlock

`MaxPool2d(2×2)` 下采样后接 `DoubleConv`，空间尺寸减半、通道数翻倍。

#### UpBlock

`ConvTranspose2d(2×2, stride=2)` 上采样后与对应层的跳跃连接输出做 **通道拼接**，再接 `DoubleConv`：

```
输入 → TransposedConv(×2) → [拼接跳过连接] → DoubleConv → 输出
```

```python
class UpBlock(nn.Module):
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.nn.functional.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        x = torch.cat([skip, x], dim=1)  # 跳跃连接拼接
        return self.conv(x)
```

#### Bottleneck

最深层的一个 `DoubleConv`，不对空间尺寸或通道数做变换。

---

### 4.5 UNet 模型 `models/unet.py`

按照论文的 6 层深度 UNet 架构组装：

```
输入 (12, 128, 128)
    │
    ├─ Entry: DoubleConv(12 → 32)
    │
    ├─ Encoder (×6):                    Skip Connections
    │   ├─ DownBlock(32 → 64)    ────────▶ ch[0]
    │   ├─ DownBlock(64 → 128)   ────────▶ ch[1]
    │   ├─ DownBlock(128 → 256)  ────────▶ ch[2]
    │   ├─ DownBlock(256 → 512)  ────────▶ ch[3]
    │   ├─ DownBlock(512 → 1024) ────────▶ ch[4]
    │   └─ DownBlock(1024 → 2048)───────▶ ch[5]
    │
    ├─ Bottleneck: DoubleConv(2048 → 2048)
    │
    ├─ Decoder (×6):
    │   ├─ UpBlock(2048 → 1024) + skip[5]
    │   ├─ UpBlock(1024 → 512)  + skip[4]
    │   ├─ UpBlock(512 → 256)   + skip[3]
    │   ├─ UpBlock(256 → 128)   + skip[2]
    │   ├─ UpBlock(128 → 64)    + skip[1]
    │   └─ UpBlock(64 → 32)     + skip[0]
    │
    └─ Final: Conv2d(32 → 4, kernel=1)
          │
         输出 (4, 128, 128)  —  ρ, p, u, v
```

```python
class UNet(nn.Module):
    def forward(self, x):
        x = self.entry(x)
        skips = []
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)
        x = self.bottleneck(x)
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i + 1)])
        return self.final(x)
```

```python
def build_unet(config) -> UNet:
    return UNet(
        in_channels=config.input_channels,    # 12
        out_channels=config.output_channels,  # 4
        base_filters=config.base_filters,     # 32
        n_down=config.n_down,                 # 6
    )
```

**模型参数量：** 约 200M（base_filters=32, n_down=6）。减少 `base_filters` 到 16 可降至约 50M。

**架构修正说明：** 初始版本中 Bottleneck 错误地将输入通道从 2048 翻倍到了 4096，导致第一个 UpBlock 在 `ConvTranspose2d` 时通道不匹配。修正为保持通道数不变即可解决问题。

---

### 4.6 训练循环 `training/trainer.py`

`Trainer` 类封装了完整的训练流程：

| 方法 | 功能 |
|------|------|
| `train_epoch()` | 遍历 DataLoader，执行前向 → loss → 反向 → 优化 |
| `validate()` | 无梯度模式下评估验证集 |
| `run()` | 控制训练循环、打印日志、保存 checkpoint |

**损失函数：** 论文使用 L1 Loss：

```python
self.criterion = nn.L1Loss()
```

**Checkpoint 策略：**

```python
# 最佳模型保存（基于验证 loss）
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pth"))

# 周期性完整 checkpoint
if epoch % self.config.save_interval == 0:
    torch.save({
        "epoch": epoch,
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict(),
        "history": self.history,
    }, f"epoch_{epoch:03d}.pth")
```

---

### 4.7 学习率调度 `training/scheduler.py`

使用 `StepLR`，每 50 个 epoch LR 减半（gamma=0.5）：

```python
def build_scheduler(optimizer, config):
    return lr_sched.StepLR(
        optimizer, step_size=config.lr_scheduler_step, gamma=config.lr_scheduler_gamma)
```

初始学习率 1e-3，200 epoch 结束时 LR ≈ 6.25e-5。

---

### 4.8 推理模块 `inference/predict.py`

`Predictor` 类支持从 checkpoint 文件加载模型并执行推理：

```python
class Predictor:
    @classmethod
    def from_checkpoint(cls, checkpoint_path, norm_path, config, device):
        model = build_unet(config)
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        normalizer = Normalizer.load(norm_path)
        return cls(model, normalizer, config, device)

    def predict(self, raw_data):
        normed = self.normalizer.normalize(raw_data)
        inputs, _ = split_input_target(normed, self.config)
        x = torch.from_numpy(inputs).to(self.device)
        y_pred = self.model(x).cpu().numpy()
        return y_pred
```

---

### 4.9 评估指标 `utils/metrics.py`

| 函数 | 公式 | 用途 |
|------|------|------|
| `mean_squared_error` | MSE = mean((F − F̂)²) | 整体误差度量 |
| `relative_l2_error` | ‖F − F̂‖₂ / ‖F‖₂ | 每样本相对误差 |
| `compute_error_map` | e_map = (|F−F̂|−min)/(max−min) | 论文定义的归一化误差图 |
| `compute_pressure_coefficient` | Cp = (p − p∞) / (½ρ∞U∞²) | 压力系数 |

**误差图 e_map** 是论文的核心可视化指标，将误差归一化到 [0, 1] 区间：

```python
def compute_error_map(y_true, y_pred):
    abs_diff = np.abs(y_true - y_pred)
    flat = abs_diff.reshape(abs_diff.shape[0], -1)
    e_map = (flat - flat.min(axis=1, keepdims=True)) / \
            (flat.max(axis=1, keepdims=True) - flat.min(axis=1, keepdims=True) + 1e-12)
    return e_map.reshape(abs_diff.shape)
```

---

### 4.10 可视化 `utils/visualization.py`

三种核心可视化：

#### 流场对比图 `plot_flow_field_comparison()`

每个物理量三列：True / Pred / Error（hot colormap），方便直观对比。

#### 压力系数曲线 `plot_pressure_coefficient()`

同时绘制 CFD 和 UNet 的 Cp 沿翼型表面分布曲线，y 轴反向（空气动力学惯例）。

#### 训练曲线 `plot_training_history()`

双 y 轴半对数曲线，展示训练和验证 loss 变化趋势。

---

### 4.11 入口脚本 `scripts/train.py`

训练入口，支持命令行参数覆盖配置：

```bash
python scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 16
```

### 4.12 评估脚本 `scripts/evaluate.py`

加载已训练模型，计算指标并生成可视化结果：

```bash
python scripts/evaluate.py --data-dir train_data_2822
```

---

## 5. 训练流程全景

```
                    ┌─────────────────────────────────────┐
                    │         train_data_2822/             │
                    │     1,153 个 .npz 文件               │
                    │     格式: (16, 128, 128)              │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │       Normalizer.fit()               │
                    │   逐通道 min/max 归一化               │
                    │   输出: normalizer.npz               │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │     split_input_target()             │
                    │   (N,16,128,128)                     │
                    │       → input (N,12,128,128)        │
                    │       → target (N,4,128,128)        │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │   random_split 80/15/5               │
                    │   → train_loader (924 samples)      │
                    │   → val_loader   (172 samples)      │
                    │   → test_loader  (57 samples)       │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │         UNet 网络                     │
                    │   输入 12ch → 编码器×6 → Bottleneck  │
                    │   → 解码器×6 → 输出 4ch              │
                    │   参数: ~200M (float32≈800MB)        │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │     训练循环 (200 epochs)            │
                    │   L1 Loss + AdamW + StepLR          │
                    │   每 epoch: ~4.3 min (CPU)          │
                    │   每 10 epoch: 打印日志 + checkpoint │
                    │   每 epoch: 保存最佳模型             │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │    输出文件                          │
                    │   results/models/best_model.pth     │
                    │   results/models/epoch_*.pth        │
                    │   results/models/normalizer.npz     │
                    │   results/training_history.png      │
                    └─────────────────────────────────────┘
```

---

## 6. 关键设计决策

### 6.1 为什么用 L1 Loss 而不是 MSE？

论文明确使用 L1 损失。L1 Loss 对异常值（尤其是激波附近的高梯度区域）的惩罚比 MSE 更轻，有助于模型在激波和分离流区域保持更好的物理一致性。

### 6.2 为什么选 6 层 UNet？

| 层数 | 最深层分辨率 | 参数量 | 感受野 |
|------|------------|--------|--------|
| 4 | 8×8 | ~30M | 有限 |
| **6** | **2×2** | **200M** | **全局** |
| 7 | 1×1 | >800M | 过大 |

6 层在参数量和感受野之间取得平衡：128×128 经过 6 次 2× 下采样到 2×2，编码器能捕获全局上下文，解码器通过跳跃连接保留局部细节。

### 6.3 数据归一化策略

对全部 16 个通道分别做 min-max 归一化到 [0, 1]。由于各物理量纲差异大（ρ ≈ 1，p 可达 13.7），逐通道归一化确保各通道对 loss 贡献均衡。

### 6.4 为什么需要坐标变换？

CFD 网格是贴体 C 型网格，而 UNet 期望笛卡尔网格输入。坐标变换将 (x,y) 空间映射到 (ξ,η) ∈ [0,1]²，变换矩阵的 9 个分量作为额外输入通道提供给网络。

### 6.5 CPU 训练策略

单 epoch 约 4.3 分钟，200 epoch 约 14.4 小时。加速选项：
- 减小 `base_filters` 到 16 → 参数量 ~50M，每 epoch 约 1 分钟
- 使用 GPU/MPS 加速

---

## 7. 训练当前进度（截至 2026-06-21 22:26）

| 指标 | 值 |
|------|-----|
| 完成轮次 | ~25-26 / 200 (13%) |
| 最佳模型保存时间 | 22:26 |
| 最新 checkpoint | epoch_020 (22:02) |
| 每 epoch 耗时 | ~260 秒 (CPU) |

**"当前工作目录缺失"说明：** 原 Codex 工作目录 `/Users/origami/Documents/Unet_wngfield/` 已迁移到桌面。所有项目文件均在 `/Users/origami/Desktop/Unet_wingfield/`。

---

## 8. 使用方法

### 安装依赖

```bash
cd /Users/origami/Desktop/Unet_wingfield
pip3 install torch numpy matplotlib
```

### 训练（单一翼型）

```bash
python3 scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 16
```

### 训练（混合翼型）

```bash
python3 scripts/train.py --data-dir train_data_hybridwings --epochs 200 --batch-size 16
```

### 自定义参数

```bash
python3 scripts/train.py \
    --data-dir train_data_2822 \
    --epochs 300 --batch-size 16 \
    --lr 1e-3 --base-filters 16 --device cpu
```

### 评估已训练的模型

```bash
python3 scripts/evaluate.py --data-dir train_data_2822
```

### Python API 推理

```python
from config.train_config import TrainConfig
from inference.predict import Predictor

predictor = Predictor.from_checkpoint(
    checkpoint_path="results/models/best_model.pth",
    norm_path="results/models/normalizer.npz",
    config=TrainConfig(),
)
raw_data = np.load("test_sample.npz")["a"]  # (16, 128, 128)
flow_pred = predictor.predict_and_denormalize(raw_data)  # (1, 4, 128, 128)
```

---

## 9. 项目文件清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `config/train_config.py` | 72 | 配置 dataclass |
| `data/preprocess.py` | 140 | 文件名解析、归一化 |
| `data/dataset.py` | 110 | PyTorch Dataset |
| `models/blocks.py` | 102 | UNet 基础构件 |
| `models/unet.py` | 110 | 深度 UNet 架构 |
| `training/trainer.py` | 115 | 训练循环 |
| `training/scheduler.py` | 15 | LR 调度 |
| `inference/predict.py` | 85 | 推理模块 |
| `utils/metrics.py` | 80 | 评估指标 |
| `utils/visualization.py` | 140 | 可视化函数 |
| `scripts/train.py` | 80 | 训练入口 |
| `scripts/evaluate.py` | 85 | 评估入口 |
| `README.md` | — | 快速入门文档 |

---

## 10. 引用

```bibtex
@article{zhu2024unet,
  title={基于UNet的翼型可压缩流场机器学习推理方法},
  author={朱智杰 and 赵国庆 and 高远 and 招启军},
  journal={南京航空航天大学学报},
  year={2024}
}
```

---

*文档生成日期：2026-06-21*
*项目路径：`/Users/origami/Desktop/Unet_wingfield/`*
