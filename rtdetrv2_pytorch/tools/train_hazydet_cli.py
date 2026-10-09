"""Single-process HazyDet training launcher for RT-DETRv2.

Prepare COCO labels once, then delegate training to the official tools/train.py.
Paths supplied on the command line are relative to the caller's directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data-root', type=Path, help='Original HazyDet COCO dataset; prepare before training')
    source.add_argument('--config', type=Path, help='Previously generated train.yml or eval_*.yml')
    parser.add_argument('--prepared-dir', type=Path, default=PROJECT / 'dataset/hazydet_rtdetrv2')
    parser.add_argument('--ann-pattern', default='{split}/{split}_coco.json')
    parser.add_argument('--image-pattern', default='{split}/hazy_images')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--force-prepare', action='store_true', help='Replace generated annotations/configs')
    parser.add_argument('--prepare-only', action='store_true', help='Check and prepare data without importing PyTorch')
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument('--tuning', type=Path, help='RT-DETRv2-L COCO weights for fine-tuning')
    checkpoint.add_argument('--resume', type=Path, help='HazyDet checkpoint for resume/evaluation')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=None, help='Default: 72; also adjusts augmentation stop epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Single-process batch size')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Prepare if requested and print training argv without launching')
    return parser


def training_command(args, config):
    command = [sys.executable, str(PROJECT / 'tools/train.py'), '-c', str(config),
               '--device', args.device, '--seed', str(args.seed)]
    for flag in ('tuning', 'resume', 'output_dir'):
        value = getattr(args, flag)
        if value is not None:
            command.extend(['--' + flag.replace('_', '-'), str(value.resolve())])
    if args.amp:
        command.append('--use-amp')
    if args.test_only:
        command.append('--test-only')
    updates = [f'{loader}.{key}={value}'
               for loader in ('train_dataloader', 'val_dataloader')
               for key, value in [('total_batch_size', args.batch_size), ('num_workers', args.workers)]]
    if args.epochs is not None:
        updates.extend([f'epoches={args.epochs}',
                        f'train_dataloader.dataset.transforms.policy.epoch={args.epochs - 1}',
                        f'train_dataloader.collate_fn.stop_epoch={args.epochs - 1}'])
    return command + ['--update'] + updates


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or (args.epochs is not None and args.epochs < 1):
        parser.error('batch-size/epochs must be positive; workers must be nonnegative')
    if args.prepare_only and args.data_root is None:
        parser.error('--prepare-only requires --data-root')
    if args.test_only and args.resume is None:
        parser.error('--test-only requires --resume with trained HazyDet weights')
    for checkpoint in (args.tuning, args.resume):
        if checkpoint is not None and not checkpoint.is_file():
            parser.error(f'Checkpoint not found: {checkpoint}')
    if args.data_root is not None:
        command = [sys.executable, str(PROJECT / 'tools/prepare_hazydet.py'),
                   '--data-root', str(args.data_root.resolve()),
                   '--output-dir', str(args.prepared_dir.resolve()),
                   '--ann-pattern', args.ann_pattern, '--image-pattern', args.image_pattern,
                   '--splits', *args.splits]
        if args.force_prepare:
            command.append('--force')
        subprocess.run(command, cwd=PROJECT, check=True)
        config = args.prepared_dir.resolve() / 'train.yml'
    else:
        config = args.config.resolve()
        if not config.is_file():
            parser.error(f'Config not found: {config}')
    if args.prepare_only:
        return 0
    command = training_command(args, config)
    print('Training argv: ' + json.dumps(command, ensure_ascii=True), flush=True)
    if args.dry_run:
        return 0
    if args.tuning is None and args.resume is None:
        print('No checkpoint supplied: training from random initialization.', flush=True)
    return subprocess.run(command, cwd=PROJECT).returncode


if __name__ == '__main__':
    sys.exit(main())
