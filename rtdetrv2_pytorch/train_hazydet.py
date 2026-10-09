"""Edit the settings below, then run: python train_hazydet.py"""
from tools.hazydet_training import train

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/output/hazydet_rtdetrv2_s"
# RT-DETRv2-S COCO weights; None means training from scratch.
PRETRAINED = "weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
RESUME = None  # Set to last.pth to resume; takes priority over PRETRAINED.


def main():
    train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        pretrained=PRETRAINED,
        resume=RESUME,
        epochs=36,
        batch_size=4,
        lr=1e-4,
        device="cuda",
        num_workers=8,
        use_amp=True,
        use_ema=False,
        checkpoint_interval=5,
        seed=42,
    )


if __name__ == "__main__":
    main()
