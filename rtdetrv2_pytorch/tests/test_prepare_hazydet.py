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
    def test_small_model_and_size_metrics(self):
        sys.path.insert(0, str(PROJECT))
        self.addCleanup(lambda: sys.path.remove(str(PROJECT)))
        from tools.hazydet_training import prepare_config
        for split in ('train', 'valid'):
            folder = self.root / split
            folder.mkdir()
            (folder / 'image.jpg').touch()
            (folder / '_annotations.coco.json').write_text(json.dumps(self.data), encoding='utf-8')
        path = prepare_config(self.root, self.root / 'output', epochs=36, batch_size=1,
                              lr=1e-4, num_workers=0, use_ema=False, checkpoint_interval=5)
        import types
        package = types.ModuleType('small_test_core')
        package.__path__ = [str(PROJECT / 'src/core')]
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location('small_test_core.yaml_utils', PROJECT / 'src/core/yaml_utils.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        config = loader.load_config(str(path), {})
        self.assertEqual(config['PResNet']['depth'], 18)
        self.assertEqual(config['RTDETRTransformerv2']['num_layers'], 3)
        self.assertEqual(config['num_classes'], 3)
        self.assertEqual(config['epoches'], 36)
        self.assertFalse(config['use_ema'])
        self.assertEqual(config['train_dataloader']['dataset']['transforms']['policy']['epoch'], 33)
        self.assertIsNone(config['train_dataloader']['collate_fn']['scales'])

        spec = importlib.util.spec_from_file_location('size_metrics', PROJECT / 'src/solver/size_metrics.py')
        metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_module)
        values = [0.5, 0.7, 0.6, 0.1, 0.2, -1, 0.3, 0.4, 0.5, 0.6, 0.7, -1]
        result = metrics_module.report_size_metrics({'coco_eval_bbox': values}, self.root, epoch=2)
        self.assertEqual(result['AP_small'], 0.1)
        self.assertEqual(result['AR_medium'], 0.7)
        self.assertIsNone(result['AP_large'])
        saved = json.loads((self.root / 'size_metrics_latest.json').read_text())
        self.assertEqual(saved['epoch'], 2)
        self.assertIsNone(saved['AR_large'])

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
        import types
        package = types.ModuleType('hazydet_test_core')
        package.__path__ = [str(PROJECT / 'src/core')]
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location('hazydet_test_core.yaml_utils', PROJECT / 'src/core/yaml_utils.py')
        config_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_utils)
        train = config_utils.load_config(str(output / 'train.yml'), {})
        self.assertEqual(train['num_classes'], 3)
        self.assertFalse(train['remap_mscoco_category'])
        self.assertEqual(train['train_dataloader']['num_workers'], 0)
        self.assertTrue(train['train_dataloader']['dataset']['transforms']['ops'])
        self.assertEqual(train['model'], 'RTDETR')
        self.assertEqual(train['RTDETR']['decoder'], 'RTDETRTransformerv2')
        self.assertEqual(train['criterion'], 'RTDETRCriterionv2')
        self.assertEqual(train['train_dataloader']['total_batch_size'], 4)
        self.assertNotIn('batch_size', train['train_dataloader'])
        evaluation = config_utils.load_config(str(output / 'eval_test.yml'), {})
        self.assertEqual(evaluation['val_dataloader']['dataset']['ann_file'], (output / 'test.json').as_posix())
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

        launcher = [sys.executable, str(PROJECT / 'tools/train_hazydet_cli.py'),
                    '--config', str(output / 'train.yml'), '--epochs', '10',
                    '--batch-size', '2', '--amp', '--dry-run']
        result = subprocess.run(launcher, check=True, capture_output=True, text=True)
        argv = json.loads(result.stdout.split('Training argv: ', 1)[1])
        self.assertIn('--use-amp', argv)
        self.assertIn('epoches=10', argv)
        self.assertIn('train_dataloader.dataset.transforms.policy.epoch=9', argv)
        self.assertIn('train_dataloader.collate_fn.stop_epoch=9', argv)
        self.assertIn('train_dataloader.total_batch_size=2', argv)
        self.assertNotEqual(subprocess.run(launcher + ['--test-only'], capture_output=True).returncode, 0)

        prepared_again = self.root / 'launcher_output'
        subprocess.run([sys.executable, str(PROJECT / 'tools/train_hazydet_cli.py'),
                        '--data-root', str(self.root), '--prepared-dir', str(prepared_again),
                        '--prepare-only'], cwd=self.root, check=True, capture_output=True)
        self.assertTrue((prepared_again / 'train.yml').is_file())


if __name__ == '__main__':
    unittest.main()
