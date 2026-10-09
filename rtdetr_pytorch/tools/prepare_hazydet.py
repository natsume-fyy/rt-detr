"""Validate HazyDet COCO annotations and create zero-based RT-DETR copies.

Only Python's standard library is required. Images and source JSON stay intact.
"""
import argparse
import copy
import json
import math
from pathlib import Path


CLASSES = ('car', 'truck', 'bus')
PROJECT = Path(__file__).resolve().parents[1]


def normalize(data, image_root):
    data = copy.deepcopy(data)
    categories = data['categories']
    names = [c['name'].strip().lower() for c in categories]
    if len(categories) != 3 or set(names) != set(CLASSES):
        raise ValueError(f'Expected categories {CLASSES}; got {names}')
    mapping = {c['id']: CLASSES.index(name) for c, name in zip(categories, names)}
    if len(mapping) != 3:
        raise ValueError('Duplicate category IDs')
    images = {}
    for item in data['images']:
        if item['id'] in images:
            raise ValueError(f"Duplicate image ID: {item['id']}")
        if item['width'] <= 0 or item['height'] <= 0:
            raise ValueError(f"Invalid image dimensions: {item['id']}")
        # file_name is relative to the image root, including any subdirectories.
        path = (image_root / item['file_name'].replace('\\', '/')).resolve()
        if not path.is_relative_to(image_root.resolve()):
            raise ValueError(f'Image path outside image root: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
        item['file_name'] = path.relative_to(image_root.resolve()).as_posix()
        images[item['id']] = item
    if not images:
        raise ValueError('Empty image split')
    annotation_ids = set()
    counts = dict.fromkeys(CLASSES, 0)
    for ann in data['annotations']:
        if ann['id'] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {ann['id']}")
        annotation_ids.add(ann['id'])
        if ann['image_id'] not in images or ann['category_id'] not in mapping:
            raise ValueError(f"Unknown image/category in annotation {ann['id']}")
        box = ann['bbox']
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
            raise ValueError(f"Invalid COCO xywh box in annotation {ann['id']}: {box}")
        x, y, w, h = box
        img = images[ann['image_id']]
        if min(x + w, img['width']) <= max(x, 0) or min(y + h, img['height']) <= max(y, 0):
            raise ValueError(f"Box outside image in annotation {ann['id']}")
        ann['category_id'] = mapping[ann['category_id']]
        ann.setdefault('area', w * h)
        if not math.isfinite(ann['area']) or ann['area'] <= 0:
            raise ValueError(f"Invalid area in annotation {ann['id']}")
        ann.setdefault('iscrowd', 0)
        if ann['iscrowd'] not in (0, 1):
            raise ValueError(f"Invalid iscrowd in annotation {ann['id']}")
        counts[CLASSES[ann['category_id']]] += 1
    data['categories'] = [{'id': i, 'name': name} for i, name in enumerate(CLASSES)]
    data.setdefault('info', {})
    print(f'{len(images)} images, {len(annotation_ids)} boxes, counts={counts}, mapping={mapping}')
    return data


def write_yaml(path, config):
    # JSON is valid YAML; ASCII escapes also work with Windows legacy encodings.
    path.write_text(json.dumps(config, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--ann-pattern', default='{split}/{split}_coco.json')
    parser.add_argument('--image-pattern', default='{split}/hazy_images')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'dataset/hazydet_rtdetr')
    parser.add_argument('--force', action='store_true', help='Replace previously generated outputs')
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    if len(set(args.splits)) != len(args.splits) or any(s not in ('train', 'val', 'test', 'RDDTS') for s in args.splits):
        parser.error('Use unique splits from: train val test RDDTS')
    if not {'train', 'val'}.issubset(args.splits):
        parser.error('Include train and val; optionally add test and/or RDDTS')

    sources, folders, targets = {}, {}, []
    for split in args.splits:
        sources[split] = (root / args.ann_pattern.format(split=split)).resolve()
        folders[split] = (root / args.image_pattern.format(split=split)).resolve()
        targets.append(output / f'{split}.json')
    targets.append(output / 'train.yml')
    targets.extend(output / f'eval_{s}.yml' for s in args.splits if s != 'train')
    for target in targets:
        if target.resolve() in sources.values():
            parser.error(f'Refusing to overwrite source annotations: {target}')
        if target.exists() and not args.force:
            parser.error(f'Output exists: {target}; use --force to regenerate')

    # Validate all splits before writing any results.
    prepared = {}
    for split in args.splits:
        print(f'Checking {split}: {sources[split]}')
        with sources[split].open(encoding='utf-8-sig') as handle:
            prepared[split] = normalize(json.load(handle), folders[split])
    output.mkdir(parents=True, exist_ok=True)
    for split, data in prepared.items():
        (output / f'{split}.json').write_text(json.dumps(data, ensure_ascii=True), encoding='utf-8')

    base = PROJECT / 'configs/rtdetr/rtdetr_r50vd_6x_hazydet.yml'
    # Absolute includes work on Windows and on Linux, including separate drives.
    config = {'__include__': [base.as_posix()]}
    for loader, split in [('train_dataloader', 'train'), ('val_dataloader', 'val')]:
        config[loader] = {'dataset': {'img_folder': folders[split].as_posix(),
                                     'ann_file': (output / f'{split}.json').as_posix()}}
    write_yaml(output / 'train.yml', config)
    for split in args.splits:
        if split != 'train':
            write_yaml(output / f'eval_{split}.yml', {
                '__include__': ['train.yml'],
                'output_dir': f'./output/rtdetr_r50vd_6x_hazydet/eval_{split}',
                'val_dataloader': {'dataset': {'img_folder': folders[split].as_posix(),
                                              'ann_file': (output / f'{split}.json').as_posix()}}})
    print(f'Ready. Train with: python tools/train.py -c "{output / "train.yml"}" -t <coco_checkpoint.pth>')


if __name__ == '__main__':
    main()
