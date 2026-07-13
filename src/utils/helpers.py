"""
Utility functions for logging, configuration, and common operations.
"""
import logging
import os
import yaml
import json
from typing import Any, Dict
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level), format=fmt,
                        handlers=handlers)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_project_root() -> Path:
    return Path(__file__).parent.parent
