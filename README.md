# UNet Airfoil Compressible Flow Field Prediction

基于 UNet 的翼型可压缩流场机器学习推理方法。

参考论文：*基于UNet的翼型可压缩流场机器学习推理方法*，朱智杰等，南京航空航天大学。

---

## 项目结构

```
.
├── config/
│   └── train_config.py        # 训练/模型配置
├── data/
│   ├── dataset.py             # PyTorch Dataset / DataLoader
│   └── preprocess.py          # 文件名解析、归一化
├── models/
│   ├── unet.py                # 深度UNet架构
│   └── blocks.py              # DoubleConv, Down, Up 构建块
├── training/
│   ├── trainer.py             # 训练循环
│   └── scheduler.py           # LR调度
├── inference/
│   └── predict.py             # 推理模块
├── utils/
│   ├── metrics.py             # MSE, 误差图, Cp
│   └── visualization.py       # 流场对比图、Cp曲线
├── scripts/
│   ├── train.py               # 训练入口
│   └── evaluate.py            # 评估入口
├── results/                   # 模型权重 + 图表
├── requirements.txt
└── README.md
```

## 数据格式

`.npz` 文件包含 shape `(16, 128, 128)` 的数组（key `"a"`）：

| 通道 | 内容 | 说明 |
|------|------|------|
| 0    | Ma   | 马赫数 |
| 1    | AoA  | 攻角（度） |
| 2    | Re   | 雷诺数（×10⁶） |
| 3–11 | 网格度量 | J⁻¹, ξ̂_x, ξ̂_y, \|∇ξ\|/J, η̂_x, η̂_y, \|∇η\|/J, x₀, y₀ |
| 12–15 | 流场 | ρ (密度), p (压力), u (x-速度), v (y-速度) |

### 文件名命名规则

`{翼型名}(gen{迭代})_{Ma×100}_{AoA×100}_{Re/1000}.npz`

示例：`rae2822gen0_50_300_5000.npz` → RAE2822, Ma=0.50, AoA=3.00°, Re=5×10⁶

## 训练

### 单一翼型模式（默认）
```bash
python scripts/train.py --data-dir train_data_2822 --epochs 200 --batch-size 32
```

### 自定义参数
```bash
python scripts/train.py \
    --data-dir train_data_2822 \
    --epochs 300 \
    --batch-size 16 \
    --lr 1e-3 \
    --base-filters 32
```

训练好的模型权重保存在 `results/models/`，训练曲线保存在 `results/training_history.png`。

## 评估

```bash
python scripts/evaluate.py --data-dir train_data_2822
```

结果图保存在 `results/` 目录。

## 模型架构

深度 UNet：
- 6 个下采样层（Conv + UNet Block）
- 6 个上采样层（UNet Block + Transposed Conv）
- 跳跃连接（Skip Connections）
- 输入 12 通道 → 输出 4 通道
- 基础滤波器数：32（可配置）

## 引用

```bibtex
@article{zhu2024unet,
  title={基于UNet的翼型可压缩流场机器学习推理方法},
  author={朱智杰 and 赵国庆 and 高远 and 招启军},
  journal={南京航空航天大学学报},
  year={2024}
}
```
