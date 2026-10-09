"""Named COCO bbox metrics; values use the evaluator's native 0..1 scale."""
import json
from pathlib import Path


def report_size_metrics(stats, output_dir=None, epoch=None):
    values = stats.get('coco_eval_bbox')
    if values is None:
        return
    indices = {'AP': 0, 'AP50': 1, 'AP75': 2,
               'AP_small': 3, 'AP_medium': 4, 'AP_large': 5,
               'AR_small': 9, 'AR_medium': 10, 'AR_large': 11}
    metrics = {name: (float(values[i]) if values[i] >= 0 else None)
               for name, i in indices.items()}
    print('COCO size metrics (0-1; N/A means no evaluable ground truth):')
    for size in ('small', 'medium', 'large'):
        ap, ar = metrics[f'AP_{size}'], metrics[f'AR_{size}']
        ap_text = 'N/A' if ap is None else f'{ap:.4f}'
        ar_text = 'N/A' if ar is None else f'{ar:.4f}'
        print(f'  {size:6s}: AP={ap_text}, AR={ar_text}')
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        record = {'epoch': epoch, **metrics}
        with (output / 'size_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        (output / 'size_metrics_latest.json').write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    return metrics
