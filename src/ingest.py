
"""
Dataset ingestion pipeline.

Responsibilities:
1. Discover image records from the raw dataset.
2. Validate images using validate.py.
3. Build a complete inventory (including invalid images).
4. Export inventory and invalid-image reports.
5. Print an ingestion summary.

All validation logic is delegated to validate.py.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import REPORTS, discover_images
from schemas import ImageRecord, ImageValidationResult, InventoryRecord
from validate import validate_dataset, validate_result_coverage, validate_inventory_completeness


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INGESTION_REPORTS_DIR = REPORTS / "ingestion"
INVENTORY_PATH = INGESTION_REPORTS_DIR / "inventory.csv"
INVALID_IMAGES_PATH = INGESTION_REPORTS_DIR / "invalid_images.csv"


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

def discover_dataset_records() -> list[ImageRecord]:
    """
    Discover image records from the raw dataset and convert them
    into ImageRecord objects.
    """

    print("\nDiscovering dataset images...")

    discovered_records = discover_images()

    records = [
        ImageRecord.model_validate(record)
        for record in tqdm(discovered_records, desc="Preparing records")
    ]

    print(f"Discovered {len(records):,} images.")

    return records


# ---------------------------------------------------------------------
# Inventory creation
# ---------------------------------------------------------------------

def create_data_inventory(
    records: list[ImageRecord], validation_results: list[ImageValidationResult]
) -> list[InventoryRecord]:
    """
    Create one inventory entry for every discovered image.

    Invalid images are retained in the inventory so that the
    ingestion process remains fully traceable.
    """

    validation_by_path = {
        Path(result.path).resolve(): result
        for result in validation_results
    }

    ingestion_time = datetime.now(timezone.utc)

    inventory_records = []

    for record in tqdm(records, desc="Building inventory"):

        image_path = Path(record.path).resolve()
        validation = validation_by_path[image_path]

        try:
            file_size = image_path.stat().st_size
        except OSError:
            file_size = None

        inventory_record = InventoryRecord(
            **record.model_dump(),
            source="PlantVillage",
            width=validation.width,
            height=validation.height,
            format=validation.format,
            file_size_bytes=file_size,
            is_valid=validation.is_valid,
            validation_error=validation.error,
            ingested_at=ingestion_time,
        )

        inventory_records.append(inventory_record)

    return inventory_records


# ---------------------------------------------------------------------
# Inventory export
# ---------------------------------------------------------------------

def save_data_inventory(inventory_records: list[InventoryRecord]) -> pd.DataFrame:
    """
    Save the complete dataset inventory to CSV.
    """

    INGESTION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    inventory_df = pd.DataFrame(
        [record.model_dump(mode="json") for record in inventory_records]
    )

    inventory_df.to_csv(INVENTORY_PATH, index=False)

    print(f"\nInventory saved to: {INVENTORY_PATH}")

    return inventory_df


def save_invalid_images(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Export invalid images separately for inspection.
    """

    invalid_df = inventory_df.loc[
        ~inventory_df["is_valid"].astype(bool)
    ].copy()

    invalid_df.to_csv(INVALID_IMAGES_PATH, index=False)

    print(f"Invalid-image report saved to: {INVALID_IMAGES_PATH}")

    return invalid_df


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_ingestion_summary(
    inventory_df: pd.DataFrame, valid_records: list[ImageRecord], invalid_df: pd.DataFrame
) -> None:
    """
    Print a summary of the ingestion process.
    """

    total_images = len(inventory_df)
    valid_count = len(valid_records)
    invalid_count = len(invalid_df)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(f"Total discovered images : {total_images:,}")
    print(f"Valid images            : {valid_count:,}")
    print(f"Invalid images          : {invalid_count:,}")

    if total_images:
        valid_percentage = valid_count / total_images * 100
        invalid_percentage = invalid_count / total_images * 100

        print(f"Valid percentage        : {valid_percentage:.2f}%")
        print(f"Invalid percentage      : {invalid_percentage:.2f}%")

    print("\nImages per class:")

    class_distribution = (
        inventory_df.groupby("class_name")
        .size()
        .sort_values(ascending=False)
    )

    print(class_distribution.to_string())

    print("\nValidation status:")

    print(
        inventory_df["is_valid"]
        .value_counts()
        .rename(index={True: "Valid", False: "Invalid"})
        .to_string()
    )

    print("=" * 60)


# ---------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------

def run_ingestion() -> tuple[pd.DataFrame, list[ImageRecord]]:
    """
    Execute the complete ingestion pipeline.

    Returns:
        inventory_df: Complete inventory, including invalid images.
        valid_records: Records eligible for subsequent processing.
    """

    # 1. Discover images
    records = discover_dataset_records()

    # 2. Validate images through validate.py
    valid_records, validation_results = validate_dataset(records)

    # 3. Validate that every discovered image has a validation result
    validate_result_coverage(records, validation_results)

    # 4. Build the complete inventory
    inventory_records = create_data_inventory(
        records,
        validation_results,
    )

    # 5. Export inventory
    inventory_df = save_data_inventory(inventory_records)

    # 6. Export invalid images
    invalid_df = save_invalid_images(inventory_df)

    # 7. Validate inventory completeness
    validate_inventory_completeness(
        records,
        inventory_df["path"].tolist(),
    )

    # 8. Print summary
    print_ingestion_summary(
        inventory_df,
        valid_records,
        invalid_df,
    )

    print("\nDataset ingestion completed successfully.")

    return inventory_df, valid_records


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
