
"""Dataset and image validation functions."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from common import (
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    gate,
)

from schemas import (
    ImageRecord,
    ImageValidationResult,
)


# -------------------------------------------------------------------
# 1. IMAGE VALIDATION
# -------------------------------------------------------------------

def validate_image(
    record: ImageRecord,
) -> ImageValidationResult:
    """
    Validate image readability, decoding, and dimensions.

    Does not resize or modify the original image.
    """

    image_path = Path(record.path)

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.load()

            width, height = image.size
            image_format = image.format

            errors = []

            if width < MIN_IMAGE_WIDTH:
                errors.append(
                    f"Width {width} is below minimum "
                    f"{MIN_IMAGE_WIDTH}."
                )

            if height < MIN_IMAGE_HEIGHT:
                errors.append(
                    f"Height {height} is below minimum "
                    f"{MIN_IMAGE_HEIGHT}."
                )

            if errors:
                return ImageValidationResult(
                    path=image_path,
                    is_valid=False,
                    width=width,
                    height=height,
                    format=image_format,
                    error=" ".join(errors),
                )

            return ImageValidationResult(
                path=image_path,
                is_valid=True,
                width=width,
                height=height,
                format=image_format,
            )

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ) as exc:

        return ImageValidationResult(
            path=image_path,
            is_valid=False,
            error=str(exc),
        )


# -------------------------------------------------------------------
# 2. DATASET VALIDATION
# -------------------------------------------------------------------

def validate_dataset(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageValidationResult]]:
    """Validate all discovered images."""

    valid_records = []
    validation_results = []

    for record in tqdm(records, desc="Validating images"):

        result = validate_image(record)

        validation_results.append(result)

        if result.is_valid:
            valid_records.append(record)

    return valid_records, validation_results


# -------------------------------------------------------------------
# 3. VALIDATION RESULT CONSISTENCY
# -------------------------------------------------------------------

def validate_result_coverage(
    records: list[ImageRecord],
    validation_results: list[ImageValidationResult],
) -> None:
    """Ensure each discovered image has exactly one validation result."""

    discovered_paths = [
        str(Path(record.path).resolve())
        for record in records
    ]

    validated_paths = [
        str(Path(result.path).resolve())
        for result in validation_results
    ]

    gate(
        len(discovered_paths) == len(set(discovered_paths)),
        "Duplicate image paths found in discovered records.",
    )

    gate(
        len(validated_paths) == len(set(validated_paths)),
        "Duplicate paths found in validation results.",
    )

    gate(
        set(discovered_paths) == set(validated_paths),
        "Validation results do not cover all discovered images.",
    )


# -------------------------------------------------------------------
# 4. INVENTORY VALIDATION
# -------------------------------------------------------------------

def validate_inventory_completeness(
    records: list[ImageRecord],
    inventory_paths: list[str],
) -> None:
    """Ensure every discovered image appears exactly once in inventory."""

    discovered_paths = {
        str(Path(record.path).resolve())
        for record in records
    }

    normalized_inventory_paths = [
        str(Path(path).resolve())
        for path in inventory_paths
    ]

    gate(
        len(normalized_inventory_paths)
        == len(set(normalized_inventory_paths)),
        "Duplicate paths found in inventory.",
    )

    gate(
        discovered_paths == set(normalized_inventory_paths),
        "Inventory does not contain exactly all discovered images.",
    )