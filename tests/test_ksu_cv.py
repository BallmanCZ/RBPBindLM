"""CPU-only regression tests; extract pure functions without importing TensorFlow."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'gena_lm_transformer_ca_ksu'
sys.path.insert(0, str(MODULE))
from ksu_undersampling import ksu_undersampling

TREE = ast.parse((MODULE / 'run_ksu_gena_lm_transformer_ca.py').read_text())
NS = {'np': np, 'ksu_undersampling': ksu_undersampling, 'SEED': 42}
NAMES = {'_deterministic_ksu', '_resample_training_fold', '_validate_split'}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES], type_ignores=[]), '<pure-functions>', 'exec'), NS)


class CVRegressionTests(unittest.TestCase):
    def setUp(self):
        ids = np.arange(60)
        self.data = {
            'seq_pooled': np.column_stack([ids, ids % 7]).astype(float),
            'seq_embeddings': ids[:, None, None],
            'seq_mask': ids[:, None],
            'struc_features': ids[:, None, None],
            'labels': np.array([1] * 20 + [0] * 40),
        }

    def test_training_only_sampling_and_original_index_alignment(self):
        self.assertIn('_resample_training_fold', NS)
        coverage = []
        for train, val in StratifiedKFold(10, shuffle=True, random_state=42).split(self.data['seq_pooled'], self.data['labels']):
            sampled = NS['_resample_training_fold'](self.data, train)
            original = sampled['original_indices']
            self.assertFalse(np.intersect1d(original, val).size)
            np.testing.assert_array_equal(original, train[sampled['indices']])
            for key in self.data:
                np.testing.assert_array_equal(sampled[key], self.data[key][original])
            coverage.extend(val)
        np.testing.assert_array_equal(sorted(coverage), np.arange(60))

    def test_fold_validation_labels_do_not_change_sampling(self):
        self.assertIn('_resample_training_fold', NS)
        train = np.arange(30)
        first = NS['_resample_training_fold'](self.data, train)
        self.data['labels'][30:] = 1 - self.data['labels'][30:]
        second = NS['_resample_training_fold'](self.data, train)
        np.testing.assert_array_equal(first['original_indices'], second['original_indices'])

    def test_misaligned_branches_fail_fast(self):
        self.assertIn('_validate_split', NS)
        self.data['seq_mask'] = self.data['seq_mask'][:-1]
        with self.assertRaises(ValueError):
            NS['_validate_split'](self.data)

    def test_ksu_reordered_indices_keep_all_modalities_aligned(self):
        original_ksu = NS['ksu_undersampling']
        def reversed_selection(x, y, random_state):
            indices = np.array([7, 2, 5])
            return x[indices], y[indices], indices
        NS['ksu_undersampling'] = reversed_selection
        try:
            result = NS['_resample_training_fold'](self.data, np.arange(60)[::-1])
            for key in self.data:
                np.testing.assert_array_equal(result[key], self.data[key][result['original_indices']])
        finally:
            NS['ksu_undersampling'] = original_ksu

    def test_ksu_mismatched_returned_labels_rejected(self):
        original_ksu = NS['ksu_undersampling']
        NS['ksu_undersampling'] = lambda x, y, random_state: (x[:2], 1-y[:2], np.arange(2))
        try:
            with self.assertRaises(ValueError):
                NS['_resample_training_fold'](self.data, np.arange(60))
        finally:
            NS['ksu_undersampling'] = original_ksu

    def test_protocol_has_no_validation_threshold_search(self):
        main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'PreRBP_10fold_and_test')
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Name) and n.func.id.startswith('_choose_threshold') for n in calls))
        splits = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'split']
        self.assertEqual(ast.unparse(splits[0]), "skf.split(seq_pooled, labels)")
        # Only the final independent-test training can sample the full training set.
        full_calls = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == '_deterministic_ksu']
        self.assertEqual(len(full_calls), 1)
        loop = next(n for n in main.body if isinstance(n, ast.For))
        self.assertGreater(full_calls[0].lineno, loop.end_lineno)


if __name__ == '__main__':
    unittest.main()
