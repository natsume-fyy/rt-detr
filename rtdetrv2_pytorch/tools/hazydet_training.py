"""Data preparation and official trainer integration for the simple entry point."""
import json
import subprocess
import sys
from pathlib import Path

from .prepare_hazydet import normalize, write_yaml

PROJECT = Path(__file__).resolve().parents[1]


def locate_split(root, split):
    # Official HazyDet and RF-DETR/Roboflow COCO layouts.
    names = ('val', 'valid') if split == 'val' else (split,)
    for name in names:
        folder = root / name
        for images in ('hazy_images', 'hazy images'):
            annotation = folder / f'{name}_coco.json'
            if annotation.is_file() and (folder / images).is_dir():
                return folder / images, annotation
        annotation = folder / '_annotations.coco.json'
        if annotation.is_file():
            return folder, annotation
    raise FileNotFoundError(f'Cannot locate {split} COCO images/annotations under {root}')


def prepare_config(dataset_dir, output_dir, *, epochs, batch_size, lr, num_workers,
                   use_ema, checkpoint_interval):
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    prepared = output / 'prepared'
    config = {
        '__include__': [(PROJECT / 'configs/rtdetrv2/rtdetrv2_r18vd_hazydet.yml').as_posix()],
        'output_dir': output.as_posix(), 'epoches': epochs,
        'optimizer': {'lr': lr}, 'use_ema': use_ema,
        'checkpoint_freq': checkpoint_interval,
    }
    datasets = {}
    for split, loader in [('train', 'train_dataloader'), ('val', 'val_dataloader')]:
        images, annotations = locate_split(root, split)
        destination = prepared / f'{split}.json'
        if destination.resolve() == annotations.resolve():
            raise ValueError('Prepared annotations must differ from source annotations')
        with annotations.open(encoding='utf-8-sig') as handle:
            datasets[split] = normalize(json.load(handle), images)
        config[loader] = {
            'total_batch_size': batch_size, 'num_workers': num_workers,
            'dataset': {'img_folder': images.as_posix(), 'ann_file': destination.as_posix()},
        }
    if len(datasets['train']['images']) < batch_size:
        raise ValueError('Training split is smaller than batch_size (drop_last=True)')
    config['train_dataloader']['dataset']['transforms'] = {'policy': {'epoch': max(0, epochs - 3)}}
    prepared.mkdir(parents=True, exist_ok=True)
    for split, data in datasets.items():
        (prepared / f'{split}.json').write_text(json.dumps(data, ensure_ascii=True), encoding='utf-8')
    path = prepared / 'train.yml'
    write_yaml(path, config)
    return path


def train(*, dataset_dir, output_dir, pretrained=None, resume=None, epochs=36,
          batch_size=4, lr=1e-4, device='cuda', num_workers=8, use_amp=True,
          use_ema=False, checkpoint_interval=5, seed=42):
    if min(epochs, batch_size, checkpoint_interval) < 1 or num_workers < 0 or lr <= 0:
        raise ValueError('Invalid training parameters')
    checkpoint = resume or pretrained
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    output = Path(output_dir).resolve()
    if not resume and (output / 'last.pth').exists():
        raise FileExistsError('Output contains a previous run; choose a new output_dir or set RESUME')
    config = prepare_config(dataset_dir, output, epochs=epochs, batch_size=batch_size,
                            lr=lr, num_workers=num_workers, use_ema=use_ema,
                            checkpoint_interval=checkpoint_interval)
    command = [sys.executable, str(PROJECT / 'tools/train.py'), '-c', str(config),
               '--device', device, '--seed', str(seed)]
    if checkpoint:
        command += ['--resume' if resume else '--tuning', str(Path(checkpoint).resolve())]
    if use_amp:
        command += ['--use-amp']
    subprocess.run(command, cwd=PROJECT, check=True)
    print(f'Training complete. Size metrics: {output / "size_metrics.jsonl"}')
