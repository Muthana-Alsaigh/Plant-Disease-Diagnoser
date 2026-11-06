"""Shared helpers for the pipeline."""

import os
import json
import random
import hashlib

from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1] # Path(__file__): Corrent file ath object; resolve: absolute; parents[1]: one level up
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"  # path objects concats with '/'

# Existing dataset directory
DATASET_DIR = Path(
        str(DATA / "Plant_leave_diseases_dataset_without_augmentation")
).expanduser().resolve()


MANIFESTS = DATA / "manifests"
PROCESSED_DATA = DATA / "processed"

REPORTS = ROOT / "reports"
GATES = REPORTS / "gates"
METRICS = REPORTS / "metrics"
PLOTS = REPORTS / "plots"
COMPARISON_REPORTS = REPORTS / "model_comparison"

MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
SAVED_MODELS = MODELS / "saved_models"
TFLITE_MODELS = MODELS / "tflite"

NOTEBOOKS = ROOT / "notebooks"
CONFIGS = ROOT / "configs"

# Create project directories if they don't exist.
PROJECT_DIRECTORIES = [
    MANIFESTS,
    PROCESSED_DATA,
    GATES,
    METRICS,
    PLOTS,
    COMPARISON_REPORTS,
    CHECKPOINTS,
    SAVED_MODELS,
    TFLITE_MODELS,
    NOTEBOOKS,
    CONFIGS,
]

for directory in PROJECT_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# Seed initialization
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


# ============================================================
# Images CONFIGURATION
# ============================================================
IMAGE_SIZE = (224, 224)
IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]

IMAGE_CHANNELS = 3
IMAGE_MODE = "RGB"

BATCH_SIZE = 32

# TensorFlow automatically chooses an appropriate prefetch buffer.
AUTOTUNE = tf.data.AUTOTUNE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64

# ============================================================
# DATASET CONFIGURATION
# ============================================================
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SUPPORTED_SPLIT_NAMES = {
    "train",
    "validation",
    "test",
}


# ============================================================
# QUALITY GATES
# ============================================================
class GateError(RuntimeError):
    """
    Raised when a project quality gate fails.
    """
    pass


def gate(condition: bool, message: str) -> None:
    """
    Check a quality condition.

    If the condition fails, record the failure and raise GateError.
    """

    if condition:
        return

    report = {
        "status": "FAILED",
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    GATES.mkdir(parents=True, exist_ok=True)

    report_path = GATES / "latest_gate_failure.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    raise GateError(message)


# ============================================================
# DATASET DISCOVERY
# ============================================================
def get_class_names(
    dataset_dir: Path = DATASET_DIR,
) -> list[str]:
    """
    Discover class names from the dataset directory.

    Each immediate subdirectory is considered one class.
    """

    gate(
        dataset_dir.exists(),
        f"Dataset directory does not exist: {dataset_dir}"
    )

    class_names = sorted(
        directory.name
        for directory in dataset_dir.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )

    gate(
        len(class_names) > 1,
        "The dataset must contain at least two class directories."
    )

    return class_names


def get_class_mapping(
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """
    Generate a deterministic mapping from class names to integer labels.
    """

    if class_names is None:
        class_names = get_class_names()

    return {
        class_name: index
        for index, class_name in enumerate(sorted(class_names))
    }


def get_reverse_class_mapping(
    class_mapping: dict[str, int],
) -> dict[int, str]:
    """
    Generate the reverse mapping from integer labels to class names.
    """

    return {
        index: class_name
        for class_name, index in class_mapping.items()
    }


def save_class_mapping(
    class_mapping: dict[str, int],
    path: Path | None = None,
) -> Path:
    """
    Save the class mapping used during model training.
    """

    if path is None:
        path = MODELS / "class_mapping.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(class_mapping, file, indent=4)

    return path


def load_class_mapping(
    path: Path | None = None,
) -> dict[str, int]:
    """
    Load the saved class mapping.

    The same mapping must be used during training and inference.
    """

    if path is None:
        path = MODELS / "class_mapping.json"

    gate(
        path.exists(),
        f"Class mapping not found: {path}"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# IMAGE FILE DISCOVERY
# ============================================================

def discover_images(
    dataset_dir: Path = DATASET_DIR,
) -> list[dict[str, Any]]:
    """
    Discover image files and their corresponding class labels.

    Returns:
        [
            {
                "path": "/path/to/image.jpg",
                "class_name": "Apple___Apple_scab",
                "label": 0
            },
            ...
        ]
    """

    class_names = get_class_names(dataset_dir)
    class_mapping = get_class_mapping(class_names)

    records = []

    for class_name in class_names:

        class_directory = dataset_dir / class_name

        for image_path in sorted(class_directory.rglob("*")):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):

                records.append({
                    "path": str(image_path.resolve()),
                    "class_name": class_name,
                    "label": class_mapping[class_name],
                })

    gate(
        len(records) > 0,
        "No supported image files were found in the dataset."
    )

    return records

# ============================================================
# FILE FINGERPRINTING
# ============================================================
def sha256_file(path: Path) -> str:
    """
    Calculate the SHA256 fingerprint of a file.

    Useful for dataset integrity checks and artifact tracking.
    """

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()

# ============================================================
# 11. JSON UTILITIES
# ============================================================

def save_json(data: Any, path: Path) -> Path:
    """
    Save a Python object as a JSON file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)

    return path


def load_json(path: Path) -> Any:
    """
    Load JSON data.
    """

    gate(
        path.exists(),
        f"JSON file not found: {path}"
    )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# PROJECT INFORMATION
# ============================================================

def get_project_info() -> dict[str, Any]:
    """
    Return basic project and dataset information.
    """

    class_names = get_class_names()
    records = discover_images()

    class_distribution = {
        class_name: 0
        for class_name in class_names
    }

    for record in records:
        class_distribution[record["class_name"]] += 1

    return {
        "project": "Plant Disease Diagnosis",
        "framework": "TensorFlow",
        "tensorflow_version": tf.__version__,
        "dataset_path": str(DATASET_DIR),
        "num_classes": len(class_names),
        "num_images": len(records),
        "class_names": class_names,
        "class_distribution": class_distribution,
        "image_size": IMAGE_SIZE,
        "image_channels": IMAGE_CHANNELS,
        "seed": SEED,
    }