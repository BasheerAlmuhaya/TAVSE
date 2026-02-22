"""
TAVSE configuration management.

Loads YAML configs, applies defaults, and provides a unified config object
for all experiments (audio-only, audio+rgb, audio+thermal, audio+rgb+thermal).

All cluster/user-specific paths are resolved via environment variables:
    TAVSE_DATA_ROOT  — root for all data, checkpoints, logs
    TAVSE_PROJECT_DIR — root of the TAVSE source tree

Set these in a `.env` file (see `.env.example`) or export them in your shell.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# ── Load .env file if present (lightweight, no extra dependency) ───────────
def _load_dotenv(env_path: Optional[str] = None):
    """Load key=value pairs from a .env file into os.environ (if not already set)."""
    if env_path is None:
        # Walk up from this file to find .env next to the project root
        _here = Path(__file__).resolve().parent  # src/utils/
        for candidate in [_here / "../../.env", _here / "../../../.env"]:
            candidate = candidate.resolve()
            if candidate.is_file():
                env_path = str(candidate)
                break
    if env_path is None or not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            # Do NOT override existing env vars (explicit exports take priority)
            if key not in os.environ:
                # Resolve ${VAR} references within values
                for ref_key, ref_val in os.environ.items():
                    value = value.replace(f"${{{ref_key}}}", ref_val)
                os.environ[key] = value

_load_dotenv()


# ── Resolve base paths from environment ────────────────────────────────────
def _get_data_root() -> str:
    """Return TAVSE_DATA_ROOT or a sensible default."""
    return os.environ.get("TAVSE_DATA_ROOT", os.path.expanduser("~/tavse_data"))

def _get_project_dir() -> str:
    """Return TAVSE_PROJECT_DIR or auto-detect from this file's location."""
    default = str(Path(__file__).resolve().parent.parent.parent)
    return os.environ.get("TAVSE_PROJECT_DIR", default)

DATA_ROOT = _get_data_root()
PROJECT_DIR = _get_project_dir()

# Legacy aliases (kept for backward compatibility with ingest_pipeline.py)
SCRATCH_BASE = DATA_ROOT
HOME_BASE = PROJECT_DIR


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    orig_sample_rate: int = 44100
    n_fft: int = 512
    win_length: int = 400        # 25ms at 16kHz
    hop_length: int = 160        # 10ms at 16kHz
    n_freq_bins: int = 257       # n_fft // 2 + 1
    segment_seconds: float = 2.5
    segment_samples: int = 40000  # 2.5s * 16000

    @property
    def n_stft_frames(self) -> int:
        """Number of STFT frames for segment_samples."""
        return self.segment_samples // self.hop_length + 1  # 251


@dataclass
class VisualConfig:
    fps: int = 28                 # Native SpeakingFaces fps
    roi_size: int = 96            # Mouth ROI height and width
    n_frames_per_segment: int = 70  # 2.5s * 28fps
    rgb_channels: int = 3
    thermal_channels: int = 1
    # ImageNet normalization for RGB
    rgb_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    rgb_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataConfig:
    scratch_base: str = field(default_factory=lambda: DATA_ROOT)
    rgb_lmdb_path: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/rgb_mouth.lmdb")
    thermal_lmdb_path: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/thermal_mouth.lmdb")
    audio_dir: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/audio_16k")
    noise_dir: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/noise")
    metadata_dir: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/metadata")
    train_manifest: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/metadata/train_manifest.csv")
    val_manifest: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/metadata/val_manifest.csv")
    test_manifest: str = field(default_factory=lambda: f"{DATA_ROOT}/processed/metadata/test_manifest.csv")
    # Noise mixing
    train_snr_range: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0, 10.0, 15.0])
    test_snr_levels: List[float] = field(default_factory=lambda: [-5.0, 0.0, 5.0, 10.0])


@dataclass
class ModelConfig:
    # Audio U-Net encoder
    audio_channels: List[int] = field(default_factory=lambda: [1, 32, 64, 128, 256, 512])
    audio_bottleneck_dim: int = 512
    # Visual encoder
    visual_backbone: str = "resnet18"
    visual_feat_dim: int = 512
    gru_hidden: int = 256
    gru_layers: int = 2
    # Fusion
    fusion_dim: int = 512
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    fusion_alpha_init: float = 0.1
    # Modality config
    active_modalities: List[str] = field(default_factory=lambda: ["audio"])


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    # Schedule
    warmup_steps: int = 2000
    max_epochs: int = 100
    # Loss
    loss_si_snr_weight: float = 1.0
    loss_mag_weight: float = 0.1
    # Early stopping
    patience: int = 10
    # Mixed precision
    use_amp: bool = True
    # Gradient accumulation
    grad_accum_steps: int = 1
    # Checkpointing
    checkpoint_dir: str = field(default_factory=lambda: f"{DATA_ROOT}/checkpoints")
    keep_top_k: int = 3
    # Logging
    log_dir: str = field(default_factory=lambda: f"{DATA_ROOT}/logs")
    log_interval: int = 50  # steps
    val_interval: int = 1    # epochs
    # Seed
    seed: int = 42


@dataclass
class TAVSEConfig:
    """Top-level config combining all sub-configs."""
    experiment_name: str = "audio_only"
    audio: AudioConfig = field(default_factory=AudioConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.training.checkpoint_dir, self.experiment_name)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.training.log_dir, self.experiment_name)


def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def _expand_env_vars(obj):
    """Recursively expand ${VAR} references in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def _dict_to_config(d: dict) -> TAVSEConfig:
    """Convert a flat/nested dict to TAVSEConfig."""
    cfg = TAVSEConfig()
    cfg.experiment_name = d.get("experiment_name", cfg.experiment_name)

    # Map sub-dicts to dataclass fields
    mapping = {
        "audio": (cfg.audio, AudioConfig),
        "visual": (cfg.visual, VisualConfig),
        "data": (cfg.data, DataConfig),
        "model": (cfg.model, ModelConfig),
        "training": (cfg.training, TrainingConfig),
    }
    for section, (obj, cls) in mapping.items():
        if section in d:
            for key, val in d[section].items():
                if hasattr(obj, key):
                    setattr(obj, key, val)
    return cfg


def load_config(yaml_path: str) -> TAVSEConfig:
    """Load a YAML config file and return a TAVSEConfig.

    All string values support ``${VAR}`` expansion against the process
    environment, so configs can use ``${TAVSE_DATA_ROOT}`` etc.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env_vars(raw)
    return _dict_to_config(raw)


def load_config_with_overrides(yaml_path: str, overrides: Optional[dict] = None) -> TAVSEConfig:
    """Load config from YAML, then apply dict overrides."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        _merge_dict(raw, overrides)
    raw = _expand_env_vars(raw)
    return _dict_to_config(raw)
