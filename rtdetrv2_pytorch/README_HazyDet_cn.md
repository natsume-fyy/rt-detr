# RT-DETRv2-S 在 HazyDet 上训练

## 直接运行

在 `rtdetrv2_pytorch` 中安装 `requirements.txt` 的依赖，编辑 `train_hazydet.py` 顶部：

- `DATASET_DIR`：数据集路径。
- `OUTPUT_DIR`：本次实验输出目录。
- `PRETRAINED`：RT-DETRv2-S COCO 权重路径，设为 `None` 则从头训练。
- `RESUME`：续训的 `last.pth` 路径，默认 `None`，设置后优先于预训练权重。

使用本目录 README 中 **RT-DETRv2-S** 对应的权重：
[rtdetrv2_r18vd_120e_coco_rerun_48.1.pth](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth)。
不要使用 RF-DETR、RT-DETRv2-M/L 或原版 RT-DETR 权重代替。

```bash
cd rtdetrv2_pytorch
python train_hazydet.py
```

模型为 ResNet-18、3 层 v2 解码器、640 输入。默认训练 36 轮、batch size=4、lr=1e-4、8 个 worker、AMP 开启、EMA 关闭，每 5 轮额外保存 checkpoint。这些参数直接在 `train(...)` 内修改。Windows 上可将 `num_workers` 改成 0。
原版 S 配置训练 120 轮；36 轮是本脚本设置，不承诺复现官方收敛精度。
本入口使用官方训练器，没有 RF-DETR 的 HBS/ECA、梯度累积或训练后可视化功能；实际单进程有效 batch size 为 4，不是示例梯度累积后的 16。

## 数据布局

自动识别以下两种 COCO 布局：

```text
# 官方 HazyDet
HazyDet/train/hazy_images/...
HazyDet/train/train_coco.json
HazyDet/val/hazy_images/...
HazyDet/val/val_coco.json

# RF-DETR 风格
HazyDet_RFDETR/train/...jpg
HazyDet_RFDETR/train/_annotations.coco.json
HazyDet_RFDETR/valid/...jpg
HazyDet_RFDETR/valid/_annotations.coco.json
```

支持 `hazy images` 目录名，以及 `val`/`valid` 验证目录。JSON 必须含 car、truck、bus 三类；按名称映射为 0、1、2。图片名称相对于对应图片目录。转换结果保存在 `OUTPUT_DIR/prepared`，原始标注不变。检查图片存在、框和类别引用，不解码核对图片尺寸。每次启动会重新检查并生成标注。

## 小、中、大目标结果

每轮验证以及单独评估均在终端输出：

```text
COCO size metrics (0-1; N/A means no evaluable ground truth):
  small : AP=..., AR=...
  medium: AP=..., AR=...
  large : AP=..., AR=...
```

这是按 COCO 原图标注面积范围计算的指标：small 对应 0–32²，medium 对应 32²–96²，large 对应 96² 以上（边界遵循 COCO evaluator）。不是按缩放后的 640 图像重新分组。
AP 使用 IoU 0.50:0.95；AR 使用 COCO 默认 maxDets=100。0.25 表示 25%；没有可评估标注时显示 N/A，JSON 中为 null。

输出目录包含：

- `size_metrics.jsonl`：每轮 AP、AP50、AP75、AP_small/medium/large、AR_small/medium/large，epoch 从 0 开始。
- `size_metrics_latest.json`：最近一次评估；不是最佳轮次结果。
- `log.txt`：官方完整训练日志。
- `last.pth`、`best.pth`：最近一轮、按验证总体 AP 选择的最佳权重。

单独评估最佳模型（将路径替换为你的实际 OUTPUT_DIR）：

```bash
python tools/train.py -c /path/to/output/prepared/train.yml -r /path/to/output/best.pth --test-only --output-dir /path/to/output/best_eval
```

这会在 `best_eval` 内保存最佳模型的尺寸指标。默认使用验证集，不自动使用测试集。

## 高级入口与测试

原来的命令行入口保留为 `tools/train_hazydet_cli.py`，`--help` 可查看参数；该旧入口仍默认生成 R50 配置。
数据转换脚本 `tools/prepare_hazydet.py` 也继续保留。

```bash
python -m unittest discover -s tests -p test_prepare_hazydet.py -v
```

测试覆盖类别转换、配置继承、S 架构、两类目录布局、训练参数及尺寸指标映射；不等同于真实训练和精度验证。
