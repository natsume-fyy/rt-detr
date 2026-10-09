# RT-DETR 在 HazyDet 上训练与评估

本次适配的是 `rtdetr_pytorch`（原版 RT-DETR），默认 R50、640 输入、72 epochs。
使用 RGB 雾图和检测框，不需要清晰图、深度图或 MMDetection。

## 1. 环境

在仓库根目录执行 `cd rtdetr_pytorch`，以下命令均在该目录运行。
建议创建独立 Python 3.10 环境，然后安装仓库原定版本：

```powershell
conda create -n rtdetr-hazydet python=3.10 -y
conda activate rtdetr-hazydet
python -m pip install "numpy<2"
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本实现使用 `torch==2.0.1`、`torchvision==0.15.2` 的 `torchvision.datapoints` API，
不要直接用新版 torchvision 替代。训练需要可用的 CUDA 版 PyTorch 和 NVIDIA GPU。

## 2. 准备数据

官方数据集与目录参考：
[HazyDet](https://github.com/GrokCV/HazyDet)、
[官方 COCO 配置](https://github.com/GrokCV/HazyDet/blob/master/configs/_base_/datasets/hazydet_det.py)。
本脚本接受 COCO JSON（`images`、`annotations`、`categories`），不接受 YOLO TXT 或 VOC XML。

```text
D:/datasets/HazyDet/
  train/hazy_images/...jpg
  train/train_coco.json
  val/hazy_images/...jpg
  val/val_coco.json
  test/hazy_images/...jpg
  test/test_coco.json
```

`file_name` 必须相对于对应 `hazy_images` 目录。实际下载包目录和 JSON 名称可能不同，
用下面两个 pattern 参数指定；不要把清晰图目录当成雾图目录。

```powershell
python tools/prepare_hazydet.py --data-root "D:/datasets/HazyDet" --splits train val test
```

只有训练、验证数据时省略 `--splits train val test`。例如图片目录有空格：

```powershell
python tools/prepare_hazydet.py --data-root "D:/datasets/HazyDet" --image-pattern "{split}/hazy images" --ann-pattern "{split}/labels/{split}_coco.json"
```

脚本按类别名称映射为 **0=car，1=truck，2=bus**，支持原编号从 1 开始或不连续，
并检查图片存在、类别、重复 ID、框的有效性。所有 split 使用同一映射。
输出到 `dataset/hazydet_rtdetr/`，原图与原 JSON 不变。
已有输出默认拒绝覆盖；确认要重新生成时加 `--force`。
脚本只检查文件存在，不解码图片，不检查 JSON 宽高是否与图像实际宽高一致。

训练必须使用生成的 JSON。直接使用原标注并只改 `num_classes: 3` 可能导致标签越界或评估类别错位。
评估也使用转换后的标注，预测标签与评估标签保持一致；向原数据集提交预测时需要按名称恢复原 category_id。

## 3. 训练

从本目录 [README 的 Model Zoo](README.md#model-zoo) 下载 **rtdetr_r50vd** COCO 权重
`rtdetr_r50vd_6x_coco_from_paddle.pth`，放到 `weights/`。

```powershell
python tools/train.py -c dataset/hazydet_rtdetr/train.yml -t weights/rtdetr_r50vd_6x_coco_from_paddle.pth --amp --seed 42
```

`-t` 只加载形状匹配的权重，80 类分类层不匹配会跳过并重新学习，这是正常现象。
不加 `-t` 时不加载预训练权重（本配置关闭了骨干网络额外下载）。
默认 batch size=4、num_workers=0，便于 Windows 单卡运行；显存不足时在
`configs/rtdetr/rtdetr_r50vd_6x_hazydet.yml` 中同时降低两个 dataloader 的 batch_size。
训练保留官方多尺度策略，因此训练尺寸不始终为 640；评估为 640。
生成的 `train.yml` 记录绝对数据路径，移动数据或代码目录后重新生成。

断点续训使用 `-r`，不要与 `-t` 同时使用：

```powershell
python tools/train.py -c dataset/hazydet_rtdetr/train.yml -r output/rtdetr_r50vd_6x_hazydet/checkpoint.pth --amp --seed 42
```

## 4. 评估

```powershell
python tools/train.py -c dataset/hazydet_rtdetr/eval_val.yml -r output/rtdetr_r50vd_6x_hazydet/checkpoint.pth --test-only
python tools/train.py -c dataset/hazydet_rtdetr/eval_test.yml -r output/rtdetr_r50vd_6x_hazydet/checkpoint.pth --test-only
```

`eval_test.yml` 仅在准备 test split 后生成。用 val 选模型，test 用于最终报告。
输出标准 COCO bbox AP/AP50/AP75 和不同尺寸目标指标。
`checkpoint.pth` 是最近一轮，不是自动选择的最优权重；依据 `log.txt` 的验证 AP
选择相应 `checkpointXXXX.pth`（文件编号从 0000 开始）。
RDDTS 可通过 `--splits train val test RDDTS` 加入，需要按实际文件名调整 pattern；
脚本会生成独立的 `eval_RDDTS.yml`，不会将其用于训练或选模型。

## 5. 验证适配脚本

```powershell
python -m unittest discover -s tests -p test_prepare_hazydet.py -v
```

这些测试验证类别映射、错误输入和生成配置的继承，不代表真实数据训练或精度验证。
