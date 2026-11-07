
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageRecord(BaseModel):
    """Metadata for a discovered dataset image."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    class_name: str
    class_id: int = Field(ge=0)
    plant_type: str
    plant_id: int = Field(ge=0)
    disease: str


class ImageValidationResult(BaseModel):
    """Result of checking an image's integrity."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    is_valid: bool
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    format: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self):
        if self.is_valid and (
            self.width is None or self.height is None
        ):
            raise ValueError(
                "Valid images must have width and height."
            )

        if not self.is_valid and not self.error:
            raise ValueError(
                "Invalid images must include an error message."
            )

        return self


class DuplicateRecord(BaseModel):
    """Duplicate detection metadata for an image."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    sha256: str
    phash: str | None = None
    duplicate_group_id: str | None = None
    is_exact_duplicate: bool = False


class ManifestRecord(ImageRecord):
    """Final validated record used by downstream pipeline stages."""

    sha256: str
    phash: str
    duplicate_group_id: str
    split: Literal["train", "validation", "test"]
