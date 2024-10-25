from pathlib import Path
from dataclasses import dataclass
from typing import Union

@dataclass
class Navigator:
    """Project paths manager"""

    root: Path = Path(__file__).parents[2]

    def __post_init__(self):
        """Initialize all project paths"""
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"

        # Create directories if they don't exist
        for path in [self.interim, self.processed]:
            path.mkdir(parents=True, exist_ok=True)

    def get_interim_path(self, presentation_name: str) -> Path:
            """Get interim path for specific presentation"""
            path = self.interim / presentation_name
            path.mkdir(parents=True, exist_ok=True)
            return path

    def get_processed_path(self, filename: str) -> Path:
        """Get path for processed results"""
        return self.processed / filename
