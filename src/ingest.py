"""Dataset ingestion and validation pipeline."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from tqdm import tqdm

from common import MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH, discover_images
from schemas import ImageRecord, ImageValidationResult, DuplicateRecord, ManifestRecord
from validate import validate_image, validate_dataset


def discover_dataset_records() -> list[ImageRecord]:
    """Discover dataset images and validate their metadata."""

    discovered_records = discover_images()

    records = []

    for record in discovered_records:
        try:
            records.append(ImageRecord.model_validate(record))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid image metadata for {record.get('path')}: {exc}"
            ) from exc

    return records


def run_ingestion():
    """Run the initial discovery and image-validation stages."""

    print("\n[1/2] Discovering dataset images...")

    records = discover_dataset_records()

    print(f"Discovered images: {len(records):,}")

    print("\n[2/2] Validating image integrity...")

    valid_records, validation_results = validate_dataset(records)

    invalid_count = len(records) - len(valid_records)

    print("\nIngestion validation summary")
    print("----------------------------")
    print(f"Discovered images: {len(records):,}")
    print(f"Valid images:      {len(valid_records):,}")
    print(f"Invalid images:    {invalid_count:,}")

    return valid_records, validation_results


if __name__ == "__main__":
    run_ingestion()