"""
Unit tests for joint longitudinal-survival data utilities.
"""

import unittest

import numpy as np
import pandas as pd
import torch

from vaelong.data import (
    JointLongitudinalSurvivalDataset,
    align_time_varying_covariates_to_grid,
    build_joint_dataset_inputs,
    generate_synthetic_joint_longitudinal_survival_data,
    split_joint_tables_by_fold,
)


class TestJointLongitudinalSurvivalDataset(unittest.TestCase):
    """Test cases for the joint dataset wrapper."""

    def test_joint_dataset_getitem_and_defaults(self):
        data = np.random.randn(4, 5, 2).astype(np.float32)
        dataset = JointLongitudinalSurvivalDataset(
            data=data,
            event_times=np.array([1.2, 0.8, 2.1, 1.5], dtype=np.float32),
            event_indicators=np.array([1, 0, 1, 0], dtype=np.float32),
            normalize=False,
        )

        (
            item,
            mask,
            length,
            baseline,
            times,
            time_varying_covariates,
            event_time,
            event_indicator,
            event_covariates,
        ) = dataset[0]

        self.assertEqual(item.shape, torch.Size([5, 2]))
        self.assertEqual(mask.shape, torch.Size([5, 2]))
        self.assertEqual(length.item(), 5)
        self.assertEqual(baseline.shape, torch.Size([0]))
        self.assertEqual(times.shape, torch.Size([5]))
        self.assertEqual(time_varying_covariates.shape, torch.Size([5, 0]))
        self.assertAlmostEqual(event_time.item(), 1.2, places=6)
        self.assertEqual(event_indicator.item(), 1.0)
        self.assertEqual(event_covariates.shape, torch.Size([0]))

    def test_joint_dataset_variable_length_sequences(self):
        data = [
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[3.0]], dtype=np.float32),
        ]
        mask = [
            np.array([[1.0], [1.0]], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
        ]
        times = [
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ]
        time_varying_covariates = [
            np.array([[10.0], [11.0]], dtype=np.float32),
            np.array([[20.0]], dtype=np.float32),
        ]

        dataset = JointLongitudinalSurvivalDataset(
            data=data,
            mask=mask,
            event_times=np.array([1.5, 0.7], dtype=np.float32),
            event_indicators=np.array([1.0, 0.0], dtype=np.float32),
            times=times,
            time_varying_covariates=time_varying_covariates,
            event_covariates=np.array([[1.0], [2.0]], dtype=np.float32),
            normalize=False,
        )

        _, _, length, _, subject_times, subject_tvc, _, _, _ = dataset[1]
        self.assertEqual(length.item(), 1)
        self.assertEqual(subject_times.shape, torch.Size([2]))
        self.assertTrue(torch.allclose(subject_times[:1], torch.tensor([0.5])))
        self.assertEqual(subject_tvc.shape, torch.Size([2, 1]))
        self.assertTrue(torch.allclose(subject_tvc[0], torch.tensor([20.0])))

    def test_joint_dataset_invalid_event_indicators_raise(self):
        data = np.random.randn(2, 3, 1).astype(np.float32)
        with self.assertRaises(ValueError):
            JointLongitudinalSurvivalDataset(
                data=data,
                event_times=np.array([1.0, 2.0], dtype=np.float32),
                event_indicators=np.array([1.0, 2.0], dtype=np.float32),
                normalize=False,
            )


class TestTimeVaryingCovariateAlignment(unittest.TestCase):
    """Test LOCF alignment onto a hazard grid."""

    def test_align_time_varying_covariates_to_grid(self):
        values = np.array(
            [
                [[10.0], [20.0], [30.0]],
                [[5.0], [15.0], [25.0]],
            ],
            dtype=np.float32,
        )
        source_times = np.array(
            [
                [0.0, 1.0, 3.0],
                [0.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        )
        target_times = np.array(
            [
                [0.5, 2.0, 3.0],
                [-1.0, 1.0, 3.0],
            ],
            dtype=np.float32,
        )
        lengths = np.array([3, 2], dtype=np.int64)

        aligned = align_time_varying_covariates_to_grid(
            values,
            source_times,
            target_times,
            lengths=lengths,
            fill_value=-1.0,
        )

        expected = np.array(
            [
                [[10.0], [20.0], [30.0]],
                [[-1.0], [5.0], [15.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(aligned, expected)


class TestBuildJointDatasetInputs(unittest.TestCase):
    """Test building joint dataset inputs from tabular data."""

    def setUp(self):
        self.longitudinal_df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 2],
                "time": [0.0, 1.0, 0.0, 1.0, 2.0],
                "y": [10.0, 11.0, 20.0, np.nan, 22.0],
                "drug": ["A", "A", "B", "B", "B"],
            }
        )
        self.survival_df = pd.DataFrame(
            {
                "id": [1, 2],
                "event_time": [1.5, 2.5],
                "event": [1, 0],
                "age": [50.0, 60.0],
                "sex": ["F", "M"],
                "site": ["east", "west"],
            }
        )

    def test_build_joint_dataset_inputs(self):
        inputs = build_joint_dataset_inputs(
            longitudinal_df=self.longitudinal_df,
            survival_df=self.survival_df,
            subject_col="id",
            time_col="time",
            feature_cols=["y"],
            event_time_col="event_time",
            event_indicator_col="event",
            baseline_cols=["age", "sex"],
            time_varying_covariate_cols=["drug"],
            event_covariate_cols=["site"],
        )

        self.assertEqual(inputs["subject_ids"].tolist(), [1, 2])
        self.assertEqual(len(inputs["data"]), 2)
        self.assertEqual(inputs["data"][0].shape, (2, 1))
        self.assertEqual(inputs["data"][1].shape, (3, 1))
        self.assertEqual(inputs["mask"][1][1, 0], 0.0)
        self.assertEqual(inputs["baseline_covariates"].shape, (2, 2))
        self.assertEqual(inputs["event_covariates"].shape, (2, 1))
        self.assertEqual(inputs["time_varying_covariates"][0].shape, (2, 1))
        self.assertGreater(
            inputs["categorical_maps"]["baseline_covariates"]["sex"]["F"],
            0.0,
        )
        self.assertGreater(
            inputs["categorical_maps"]["time_varying_covariates"]["drug"]["A"],
            0.0,
        )

    def test_build_joint_dataset_inputs_reuses_category_maps(self):
        train_inputs = build_joint_dataset_inputs(
            longitudinal_df=self.longitudinal_df,
            survival_df=self.survival_df,
            subject_col="id",
            time_col="time",
            feature_cols=["y"],
            event_time_col="event_time",
            event_indicator_col="event",
            baseline_cols=["sex"],
            time_varying_covariate_cols=["drug"],
            event_covariate_cols=["site"],
        )

        test_longitudinal_df = pd.DataFrame(
            {
                "id": [3, 3],
                "time": [0.0, 1.0],
                "y": [12.0, 13.0],
                "drug": ["B", "B"],
            }
        )
        test_survival_df = pd.DataFrame(
            {
                "id": [3],
                "event_time": [1.1],
                "event": [1],
                "sex": ["M"],
                "site": ["west"],
            }
        )

        test_inputs = build_joint_dataset_inputs(
            longitudinal_df=test_longitudinal_df,
            survival_df=test_survival_df,
            subject_col="id",
            time_col="time",
            feature_cols=["y"],
            event_time_col="event_time",
            event_indicator_col="event",
            baseline_cols=["sex"],
            time_varying_covariate_cols=["drug"],
            event_covariate_cols=["site"],
            category_maps=train_inputs["categorical_maps"],
        )

        expected_drug_code = train_inputs["categorical_maps"][
            "time_varying_covariates"
        ]["drug"]["B"]
        self.assertEqual(test_inputs["time_varying_covariates"][0][0, 0], expected_drug_code)

    def test_split_joint_tables_by_fold_uses_survival_fold_map(self):
        longitudinal_df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3],
                "time": [0.0, 1.0, 0.0, 1.0, 0.0],
                "y": [10.0, 11.0, 20.0, 21.0, 30.0],
            }
        )
        survival_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "event_time": [1.5, 2.5, 0.9],
                "event": [1, 0, 1],
                "fold": [0, 1, 0],
            }
        )

        split = split_joint_tables_by_fold(
            longitudinal_df=longitudinal_df,
            survival_df=survival_df,
            subject_col="id",
            fold_id=0,
        )

        self.assertEqual(sorted(split["test_survival"]["id"].tolist()), [1, 3])
        self.assertEqual(sorted(split["train_survival"]["id"].tolist()), [2])
        self.assertEqual(sorted(split["test_longitudinal"]["id"].unique().tolist()), [1, 3])
        self.assertEqual(sorted(split["train_longitudinal"]["id"].unique().tolist()), [2])


class TestSyntheticJointDataGeneration(unittest.TestCase):
    """Test synthetic joint-data generation."""

    def test_generate_synthetic_joint_data_shapes(self):
        generated = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=12,
            seq_len=7,
            n_features=2,
            n_baseline_features=3,
            n_time_varying_covariates=2,
            n_event_covariates=2,
            seed=42,
        )

        self.assertEqual(generated["data"].shape, (12, 7, 2))
        self.assertEqual(generated["mask"].shape, (12, 7, 2))
        self.assertEqual(generated["baseline_covariates"].shape, (12, 3))
        self.assertEqual(generated["times"].shape, (12, 7))
        self.assertEqual(generated["time_varying_covariates"].shape, (12, 7, 2))
        self.assertEqual(generated["event_times"].shape, (12,))
        self.assertEqual(generated["event_indicators"].shape, (12,))
        self.assertEqual(generated["event_covariates"].shape, (12, 2))
        self.assertTrue(np.all(generated["event_times"] >= 0.0))
        self.assertTrue(np.all(np.isin(generated["event_indicators"], [0.0, 1.0])))

    def test_generate_synthetic_joint_data_is_deterministic(self):
        generated_a = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=10,
            seq_len=5,
            n_features=1,
            seed=7,
        )
        generated_b = generate_synthetic_joint_longitudinal_survival_data(
            n_samples=10,
            seq_len=5,
            n_features=1,
            seed=7,
        )

        for key in (
            "data",
            "mask",
            "baseline_covariates",
            "times",
            "time_varying_covariates",
            "event_times",
            "event_indicators",
            "event_covariates",
            "shared_effect",
        ):
            np.testing.assert_array_equal(generated_a[key], generated_b[key])


if __name__ == "__main__":
    unittest.main()
