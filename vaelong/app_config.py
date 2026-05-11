"""
YAML-backed application configuration for VAElong runners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import VariableConfig, VariableSpec


def _string_list(values: Any, field_name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError(f"{field_name} must be a list of strings.")
    return values


@dataclass
class DataConfig:
    path: str
    format: Optional[str]
    subject_col: str
    subject_label_col: Optional[str]
    time_col: str
    sort_by: list[str]
    outcome_cols: list[str]
    time_varying_cols: list[str]
    baseline_cols: list[str] = field(default_factory=list)
    observed_feature_cols: Optional[list[str]] = None
    strict_seq_len: bool = False

    @property
    def feature_cols(self) -> list[str]:
        return self.outcome_cols + self.time_varying_cols

    @property
    def resolved_observed_feature_cols(self) -> list[str]:
        return self.observed_feature_cols or self.outcome_cols


@dataclass
class TransformConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    seed: int = 42


@dataclass
class ModelConfig:
    encoder_type: str = "lstm"
    hidden_dim: int = 64
    latent_dim: int = 16


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 0.5
    use_em_imputation: bool = True
    em_iterations: int = 2
    imputation_method: str = "rwmh"
    mh_steps: int = 1
    mh_continuous_step_size: float = 0.1
    mh_bounded_step_size: float = 0.05
    mh_binary_flip_prob: float = 0.1
    device: Optional[str] = None


@dataclass
class TuningConfig:
    enabled: bool = False
    random_samples: int = 9
    learning_rates: list[float] = field(default_factory=lambda: [5e-4, 1e-3])
    weight_decays: list[float] = field(default_factory=lambda: [0.0, 1e-4])
    betas: list[float] = field(default_factory=lambda: [0.1, 0.5])
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 192])
    latent_dims: list[int] = field(default_factory=lambda: [16, 32, 48])


@dataclass
class PlotConfig:
    count: int = 4
    ids: Optional[list[str]] = None


@dataclass
class OutputConfig:
    dir: str


@dataclass
class LandmarkConfig:
    kind: str = "midpoint"


@dataclass
class ApplicationConfig:
    name: str
    data: DataConfig
    variables: VariableConfig
    transforms: list[TransformConfig]
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
    tuning: TuningConfig
    plot: PlotConfig
    output: OutputConfig
    landmark: LandmarkConfig
    config_path: Path

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    def resolve_data_path(self, override: Optional[str] = None) -> Path:
        raw = Path(override) if override is not None else Path(self.data.path)
        return raw if raw.is_absolute() else (self.config_dir / raw).resolve()

    def resolve_output_dir(self, override: Optional[str] = None) -> Path:
        raw = Path(override) if override is not None else Path(self.output.dir)
        return raw if raw.is_absolute() else (self.config_dir / raw).resolve()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": asdict(self.data),
            "variables": {
                "bounded_loss": self.variables.bounded_loss,
                "bounded_eps": self.variables.bounded_eps,
                "specs": [asdict(v) for v in self.variables.variables],
            },
            "transforms": [dict(type=t.type, **t.params) for t in self.transforms],
            "split": asdict(self.split),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "tuning": asdict(self.tuning),
            "plot": asdict(self.plot),
            "output": asdict(self.output),
            "landmark": asdict(self.landmark),
            "config_path": str(self.config_path),
        }


def _load_variable_config(raw: dict[str, Any]) -> VariableConfig:
    specs_raw = raw.get("specs")
    if not isinstance(specs_raw, list) or not specs_raw:
        raise ValueError("variables.specs must be a non-empty list.")

    specs = []
    for item in specs_raw:
        if not isinstance(item, dict):
            raise ValueError("Each variables.specs entry must be a mapping.")
        specs.append(
            VariableSpec(
                name=item["name"],
                var_type=item["var_type"],
                lower=float(item.get("lower", 0.0)),
                upper=float(item.get("upper", 1.0)),
            )
        )

    return VariableConfig(
        variables=specs,
        bounded_loss=raw.get("bounded_loss", "bce"),
        bounded_eps=float(raw.get("bounded_eps", 0.0)),
    )


def load_app_config(config_path: str | Path) -> ApplicationConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML config must be a mapping.")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Config must define a non-empty 'name'.")

    data_raw = raw.get("data")
    if not isinstance(data_raw, dict):
        raise ValueError("Config must define a 'data' section.")

    data_cfg = DataConfig(
        path=str(data_raw["path"]),
        format=data_raw.get("format"),
        subject_col=str(data_raw["subject_col"]),
        subject_label_col=data_raw.get("subject_label_col"),
        time_col=str(data_raw["time_col"]),
        sort_by=_string_list(data_raw.get("sort_by"), "data.sort_by"),
        outcome_cols=_string_list(data_raw.get("outcome_cols"), "data.outcome_cols"),
        time_varying_cols=_string_list(data_raw.get("time_varying_cols"), "data.time_varying_cols"),
        baseline_cols=_string_list(data_raw.get("baseline_cols"), "data.baseline_cols"),
        observed_feature_cols=(
            _string_list(data_raw.get("observed_feature_cols"), "data.observed_feature_cols")
            if data_raw.get("observed_feature_cols") is not None
            else None
        ),
        strict_seq_len=bool(data_raw.get("strict_seq_len", False)),
    )
    if not data_cfg.sort_by:
        data_cfg.sort_by = [data_cfg.subject_col, data_cfg.time_col]
    if not data_cfg.outcome_cols:
        raise ValueError("data.outcome_cols must be a non-empty list.")

    variables_raw = raw.get("variables")
    if not isinstance(variables_raw, dict):
        raise ValueError("Config must define a 'variables' section.")
    variable_config = _load_variable_config(variables_raw)

    transforms = []
    for item in raw.get("transforms", []):
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError("Each transform must be a mapping with a 'type'.")
        params = {k: v for k, v in item.items() if k != "type"}
        transforms.append(TransformConfig(type=str(item["type"]), params=params))

    split_raw = raw.get("split", {})
    split_cfg = SplitConfig(
        train_fraction=float(split_raw.get("train_fraction", 0.6)),
        val_fraction=float(split_raw.get("val_fraction", 0.2)),
        seed=int(split_raw.get("seed", 42)),
    )

    model_raw = raw.get("model", {})
    model_cfg = ModelConfig(
        encoder_type=str(model_raw.get("encoder_type", "lstm")),
        hidden_dim=int(model_raw.get("hidden_dim", 64)),
        latent_dim=int(model_raw.get("latent_dim", 16)),
    )

    training_raw = raw.get("training", {})
    training_cfg = TrainingConfig(
        batch_size=int(training_raw.get("batch_size", 32)),
        epochs=int(training_raw.get("epochs", 200)),
        patience=int(training_raw.get("patience", 20)),
        learning_rate=float(training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(training_raw.get("weight_decay", 1e-4)),
        beta=float(training_raw.get("beta", 0.5)),
        use_em_imputation=bool(training_raw.get("use_em_imputation", True)),
        em_iterations=int(training_raw.get("em_iterations", 2)),
        imputation_method=str(training_raw.get("imputation_method", "rwmh")),
        mh_steps=int(training_raw.get("mh_steps", 1)),
        mh_continuous_step_size=float(training_raw.get("mh_continuous_step_size", 0.1)),
        mh_bounded_step_size=float(training_raw.get("mh_bounded_step_size", 0.05)),
        mh_binary_flip_prob=float(training_raw.get("mh_binary_flip_prob", 0.1)),
        device=training_raw.get("device"),
    )

    tuning_raw = raw.get("tuning", {})
    tuning_cfg = TuningConfig(
        enabled=bool(tuning_raw.get("enabled", False)),
        random_samples=int(tuning_raw.get("random_samples", 9)),
        learning_rates=[float(v) for v in tuning_raw.get("learning_rates", [5e-4, 1e-3])],
        weight_decays=[float(v) for v in tuning_raw.get("weight_decays", [0.0, 1e-4])],
        betas=[float(v) for v in tuning_raw.get("betas", [0.1, 0.5])],
        hidden_dims=[int(v) for v in tuning_raw.get("hidden_dims", [64, 128, 192])],
        latent_dims=[int(v) for v in tuning_raw.get("latent_dims", [16, 32, 48])],
    )

    plot_raw = raw.get("plot", {})
    plot_cfg = PlotConfig(
        count=int(plot_raw.get("count", 4)),
        ids=[str(v) for v in plot_raw.get("ids")] if plot_raw.get("ids") is not None else None,
    )

    output_raw = raw.get("output", {})
    output_cfg = OutputConfig(dir=str(output_raw.get("dir", f"results/{name}")))

    landmark_raw = raw.get("landmark", {})
    landmark_cfg = LandmarkConfig(kind=str(landmark_raw.get("kind", "midpoint")))
    if landmark_cfg.kind != "midpoint":
        raise ValueError("Only landmark.kind='midpoint' is currently supported.")

    return ApplicationConfig(
        name=name,
        data=data_cfg,
        variables=variable_config,
        transforms=transforms,
        split=split_cfg,
        model=model_cfg,
        training=training_cfg,
        tuning=tuning_cfg,
        plot=plot_cfg,
        output=output_cfg,
        landmark=landmark_cfg,
        config_path=config_path,
    )
