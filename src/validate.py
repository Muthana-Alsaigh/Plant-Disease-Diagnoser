from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from tqdm import tqdm

from common import MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH, discover_images
from schemas import ImageRecord, ImageValidationResult, DuplicateRecord, ManifestRecord

def validate_image(record: ImageRecord) -> ImageValidationResult:
    """
    Validate image integrity, dimensions, and readability.

    The original image is never modified.
    """

    image_path = Path(record.path)

    try:
        # First pass: verify the image file structure.
        with Image.open(image_path) as image:
            image.verify()

        # Second pass: reopen and fully decode the image.
        with Image.open(image_path) as image:
            image.load()

            width, height = image.size
            image_format = image.format

            if width < MIN_IMAGE_WIDTH:
                raise ValueError(
                    f"Image width ({width}) is below "
                    f"the minimum ({MIN_IMAGE_WIDTH})."
                )

            if height < MIN_IMAGE_HEIGHT:
                raise ValueError(
                    f"Image height ({height}) is below "
                    f"the minimum ({MIN_IMAGE_HEIGHT})."
                )

            return ImageValidationResult(
                path=image_path,
                is_valid=True,
                width=width,
                height=height,
                format=image_format,
            )

    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return ImageValidationResult(
            path=image_path,
            is_valid=False,
            error=str(exc),
        )


def validate_dataset(records: list[ImageRecord]) -> tuple[list[ImageRecord], list[ImageValidationResult]]:
    """
    Validate all discovered images.

    Returns:
        valid_records: Images that passed validation.
        validation_results: Results for every image, including failures.
    """

    valid_records = []
    validation_results = []

    for record in tqdm(records, desc="Validating images"):
        result = validate_image(record)
        validation_results.append(result)

        if result.is_valid:
            valid_records.append(record)

    return valid_records, validation_results