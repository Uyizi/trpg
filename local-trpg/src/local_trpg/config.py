"""Application settings; the existing model environment remains independent."""
from __future__ import annotations

import os
import runpy
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_project() -> Path:
    return Path(os.environ.get("LOCAL_TRPG_MODEL_PROJECT", str(PROJECT_ROOT.parent))).resolve()


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = field(default_factory=lambda: int(os.environ.get("LOCAL_TRPG_PORT", "8768")))
    backend: str = "runner"
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("LOCAL_TRPG_DATA_DIR", str(PROJECT_ROOT / "data"))))
    model_project: Path = field(default_factory=_model_project)
    model_python: Path | None = None
    model_path: Path | None = None
    context_tokens: int = 6144
    warmup_timeout_seconds: float = 360
    generation_timeout_seconds: float = 120
    narration_mode: str = field(default_factory=lambda: os.environ.get("LOCAL_TRPG_NARRATION", "brief"))

    def __post_init__(self) -> None:
        if self.narration_mode not in ("brief", "model"):
            raise ValueError("LOCAL_TRPG_NARRATION must be brief or model")
        self.model_project = Path(self.model_project).resolve()
        self.data_dir = Path(self.data_dir)
        self.model_python = Path(self.model_python or os.environ.get("LOCAL_TRPG_MODEL_PYTHON", str(self.model_project / ".venv-qwen35/bin/python")))
        if self.model_path is None:
            override = os.environ.get("LOCAL_TRPG_MODEL_PATH")
            if override:
                self.model_path = Path(override)
            else:
                # This existing file is the authoritative model-path configuration.
                paths = runpy.run_path(str(self.model_project / "models/runtime_paths.py"))
                self.model_path = Path(paths["DEFAULT_BUNDLE"])
        else:
            self.model_path = Path(self.model_path)
        if self.context_tokens != 6144:
            raise ValueError("The verified runtime profile has a 6144-token context")


def get_settings() -> Settings:
    return Settings()
