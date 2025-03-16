import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainerConfig:
    num_train_epochs: int | None
    warmup_ratio: float | None
    warmup_steps: int | None
    max_steps: int | None
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    save_strategy: str
    save_steps: int | float | None

    @classmethod
    def load_from_path(cls, path: Path) -> "TrainerConfig":
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls(**config["TrainerConfig"])


@dataclass
class Config:
    url: str
    metadata_path: str
    audio_encoder_name_or_path: str
    text_decoder_name_or_path: str

    @classmethod
    def load_from_path(cls, path: Path) -> "Config":
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls(**config["Config"])

    def get_tar_path(self) -> list[Path]:
        if "input" not in self.url:
            raise ValueError("URL must contain 'input' to get tar path")

        return list(Path(self.url).glob("*.tar"))


def copy_config(path: Path, output_dir: Path) -> None:
    dest_path = output_dir / path.name
    shutil.copy(path, dest_path)
    dest_path.rename(output_dir / "config.yaml")
