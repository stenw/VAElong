"""
Tests for YAML application config loading and runner array construction.
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vaelong.app_config import load_app_config
from vaelong.app_runner import build_subject_arrays


class TestApplicationConfigAndRunner(unittest.TestCase):
    def test_load_app_config_with_input_only_time_varying_covariates(self):
        yaml_text = textwrap.dedent(
            """
            name: test_app

            data:
              path: ../fake.parquet
              format: parquet
              subject_col: id
              subject_label_col: label
              time_col: time
              sort_by: [id, time]
              outcome_cols: [y1, y2]
              time_varying_cols: []
              input_only_time_varying_covariate_cols: [tv1, tv2]
              baseline_cols: [age]

            variables:
              specs:
                - name: y1
                  var_type: continuous
                - name: y2
                  var_type: binary

            transforms: []
            split: {}
            model: {}
            training: {}
            tuning: {}
            plot: {}
            output:
              dir: ../out
            landmark: {}
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            config = load_app_config(config_path)

        self.assertEqual(config.data.feature_cols, ["y1", "y2"])
        self.assertEqual(config.data.input_only_time_varying_covariate_cols, ["tv1", "tv2"])

    def test_build_subject_arrays_keeps_input_only_covariates_out_of_reconstruction_targets(self):
        yaml_text = textwrap.dedent(
            """
            name: test_app

            data:
              path: ../fake.parquet
              format: parquet
              subject_col: id
              subject_label_col: label
              time_col: time
              sort_by: [id, time]
              outcome_cols: [y1, y2]
              time_varying_cols: []
              input_only_time_varying_covariate_cols: [tv1, tv2]
              baseline_cols: [age]
              strict_seq_len: true

            variables:
              specs:
                - name: y1
                  var_type: continuous
                - name: y2
                  var_type: binary

            transforms: []
            split: {}
            model: {}
            training: {}
            tuning: {}
            plot: {}
            output:
              dir: ../out
            landmark: {}
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            config = load_app_config(config_path)

        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "label": ["a", "a", "b", "b"],
                "time": [0.0, 1.0, 0.0, 1.0],
                "y1": [0.1, 0.2, 0.3, 0.4],
                "y2": [0.0, 1.0, 1.0, 0.0],
                "tv1": [10.0, 11.0, 12.0, 13.0],
                "tv2": [20.0, 21.0, 22.0, 23.0],
                "age": [40.0, 40.0, 55.0, 55.0],
            }
        )

        data, mask, baseline, times, tv_covs, patient_keys = build_subject_arrays(df, config)

        self.assertEqual(data.shape, (2, 2, 2))
        self.assertEqual(mask.shape, (2, 2, 2))
        self.assertEqual(baseline.shape, (2, 1))
        self.assertEqual(times.shape, (2, 2))
        self.assertEqual(tv_covs.shape, (2, 2, 2))
        np.testing.assert_allclose(tv_covs[0], np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32))
        self.assertEqual(list(patient_keys["subject_label"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
