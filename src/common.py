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
DATASET_DIR = Path(str(DATA / "Plant_leave_diseases_dataset_without_augmentation")).expanduser().resolve()


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
    MANIFESTS, PROCESSED_DATA, GATES,
    METRICS, PLOTS, COMPARISON_REPORTS, CHECKPOINTS, 
    SAVED_MODELS, TFLITE_MODELS, NOTEBOOKS, CONFIGS,
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

    try: # if its posible makes all operations determinant so outcomes do not varies between outcomes
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

SUPPORTED_SPLIT_NAMES = {"train", "validation", "test"}


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

def parse_class_name(class_name: str) -> tuple[str, str]:
    """
    Parse an original PlantVillage folder name.

    Example:
        Potato___Early_blight
        -> ("Potato", "Early_blight")
    """

    if "___" not in class_name:
        raise ValueError(
            f"Invalid class folder name: {class_name}"
        )

    plant_type, disease = class_name.split("___", 1)

    if not plant_type or not disease:
        raise ValueError(
            f"Invalid class folder name: {class_name}"
        )

    return plant_type, disease


def get_plant_types(dataset_dir: Path = DATASET_DIR) -> list[str]:

    class_names = get_class_names(dataset_dir)

    plant_types = {
        parse_class_name(class_name)[0]
        for class_name in class_names
    }

    return sorted(plant_types)


def get_diseases_by_plant(dataset_dir: Path = DATASET_DIR) -> dict[str, list[str]]:

    class_names = get_class_names(dataset_dir)

    diseases = {}

    for class_name in class_names:
        plant_type, disease = parse_class_name(class_name)

        diseases.setdefault(plant_type, []).append(disease)

    return {
        plant: sorted(set(plant_diseases))
        for plant, plant_diseases in sorted(diseases.items())
    }


def get_class_names(dataset_dir: Path = DATASET_DIR) -> list[str]:
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


def get_class_mapping(class_names: list[str] | None = None) -> dict[str, int]:
    """
    Generate a deterministic mapping from class names to integer labels.
    """

    if class_names is None:
        class_names = get_class_names()

    return {
        class_name: index
        for index, class_name in enumerate(sorted(class_names))
    }


def get_reverse_class_mapping(class_mapping: dict[str, int]) -> dict[int, str]:
    """
    Generate the reverse mapping from integer labels to class names.
    """

    return {
        index: class_name
        for class_name, index in class_mapping.items()
    }


def save_class_mapping(class_mapping: dict[str, int], path: Path | None = None) -> Path:
    """
    Save the class mapping used during model training.
    """

    if path is None:
        path = MODELS / "class_mapping.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(class_mapping, file, indent=4)

    return path


def load_class_mapping(path: Path | None = None) -> dict[str, int]:
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


def get_plant_mapping(dataset_dir: Path = DATASET_DIR) -> dict[str, int]:

    return {
        plant: idx
        for idx, plant in enumerate(
            get_plant_types(dataset_dir)
        )
    }


def get_reverse_plant_mapping(dataset_dir: Path = DATASET_DIR) -> dict[int, str]:

    return {
        idx: plant
        for plant, idx in get_plant_mapping(dataset_dir).items()
    }


def get_class_plant_mapping(dataset_dir: Path = DATASET_DIR) -> dict[int, int]:

    class_mapping = get_class_mapping(dataset_dir)
    plant_mapping = get_plant_mapping(dataset_dir)

    result = {}

    for class_name, class_id in class_mapping.items():
        plant_type, _ = parse_class_name(class_name)

        result[class_id] = plant_mapping[plant_type]

    return result


# ============================================================
# IMAGE FILE DISCOVERY
# ============================================================


def discover_images(dataset_dir: Path = DATASET_DIR) -> list[dict[str, Any]]:
    """
    Discover all images and extract their metadata.

    Each record contains:
        path        : Absolute image path
        class_name  : Original combined class name
        class_id    : Integer ID of the combined class
        plant_type  : Plant species
        plant_id    : Integer ID of the plant species
        disease     : Disease or healthy label
    """

    dataset_dir = Path(dataset_dir).expanduser().resolve()

    gate(
        dataset_dir.exists() and dataset_dir.is_dir(),
        f"Dataset directory does not exist or is not a directory: {dataset_dir}"
    )

    # Retrieve deterministic mappings
    class_names = get_class_names(dataset_dir)
    class_mapping = get_class_mapping(class_names)
    plant_mapping = get_plant_mapping(dataset_dir)

    records = []

    for class_name in class_names:

        # Extract plant and disease information
        plant_type, disease = parse_class_name(class_name)

        class_dir = dataset_dir / class_name

        # Discover supported image files
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            records.append({
                "path": str(image_path.resolve()),
                "class_name": class_name,
                "class_id": class_mapping[class_name],
                "plant_type": plant_type,
                "plant_id": plant_mapping[plant_type],
                "disease": disease,
            })

    gate(
        len(records) > 0,
        f"No supported images found in: {dataset_dir}"
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

        for chunk in iter(lambda: file.read(1024 * 1024), b""):
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