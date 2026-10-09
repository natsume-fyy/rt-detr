import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_hazydet', PROJECT / 'tools/prepare_hazydet.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareHazyDetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'image.jpg').touch()
        self.data = {
            'categories': [{'id': 9, 'name': 'bus'}, {'id': 3, 'name': 'car'}, {'id': 7, 'name': 'truck'}],
            'images': [{'id': 1, 'file_name': 'image.jpg', 'width': 100, 'height': 80}],
            'annotations': [{'id': i + 1, 'image_id': 1, 'category_id': c, 'bbox': [1, 2, 3, 4]}
                            for i, c in enumerate([3, 7, 9])],
        }

    def test_mapping_and_source_preservation(self):
        original = copy.deepcopy(self.data)
        result = prepare.normalize(self.data, self.root)
        self.assertEqual(self.data, original)
        self.assertEqual([a['category_id'] for a in result['annotations']], [0, 1, 2])
        self.assertEqual(result['annotations'][0]['area'], 12)
        self.assertEqual(result['annotations'][0]['iscrowd'], 0)

    def test_invalid_inputs(self):
        for mutate in [
            lambda d: d['annotations'][0].update(category_id=99),
            lambda d: d['annotations'][0].update(bbox=[1, 2, -3, 4]),
            lambda d: d['annotations'][0].update(image_id=99),
            lambda d: d['images'][0].update(file_name='missing.jpg'),
            lambda d: d['categories'][0].update(name='person'),
        ]:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises((ValueError, FileNotFoundError)):
                prepare.normalize(data, self.root)

    def test_cli_and_merged_train_eval_configs(self):
        for split in ['train', 'val', 'test']:
            folder = self.root / split / 'hazy_images'
            folder.mkdir(parents=True)
            (folder / 'image.jpg').touch()
            (folder.parent / f'{split}_coco.json').write_text(json.dumps(self.data), encoding='utf-8')
        output = self.root / 'prepared'
        command = [sys.executable, str(PROJECT / 'tools/prepare_hazydet.py'),
                   '--data-root', str(self.root), '--output-dir', str(output),
                   '--splits', 'train', 'val', 'test']
        subprocess.run(command, check=True, capture_output=True, text=True)
        spec = importlib.util.spec_from_file_location('yaml_utils', PROJECT / 'src/core/yaml_utils.py')
        config_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_utils)
        train = config_utils.load_config(str(output / 'train.yml'), {})
        self.assertEqual(train['num_classes'], 3)
        self.assertFalse(train['remap_mscoco_category'])
        self.assertEqual(train['train_dataloader']['num_workers'], 0)
        self.assertTrue(train['train_dataloader']['dataset']['transforms']['ops'])
        self.assertEqual(train['model'], 'RTDETR')
        evaluation = config_utils.load_config(str(output / 'eval_test.yml'), {})
        self.assertEqual(evaluation['val_dataloader']['dataset']['ann_file'], (output / 'test.json').as_posix())
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
