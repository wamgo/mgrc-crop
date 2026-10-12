"""Check that reusing probability files preserves training metadata without retraining."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import train_models as training


class TrainingResumeTest(unittest.TestCase):
    def test_existing_predictions_keep_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tag = 'test_source'
            for model in ['mlp', 'lgbm', 'tempcnn']:
                (output / f'{tag}_{model}.npz').touch()
            metadata = output / f'{tag}_meta.json'
            original = json.dumps({'mlp:T': 1.25, 'lgbm:history': [123]}).encode()
            metadata.write_bytes(original)
            data = {'X': np.zeros((45001, 1), np.float32),
                    'cube': np.zeros((45001, 1, 1), np.float32),
                    'y': np.zeros(45001, dtype=int)}
            with patch.object(training, 'OUT', output), patch.object(training, 'load', return_value=data):
                training.run(2017, 'L2A', 'test', [], tag)
            self.assertEqual(metadata.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
