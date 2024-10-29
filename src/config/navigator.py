from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

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

    def find_file_by_substr(
            self,
            substr: str,
            extension: Optional[str] = None,
            base_dir: Optional[Path] = None,
        ) -> List[Path]:
        """
        Find file by substring.

        Args:
            substr: substring to search for
            extension: [".png", ".pdf"] file extension to filter by (optional)

        Example:
            -$ find_by_substr("Kolm")
            -> <path to presentation about Kolmogorov Networks>
        """
        if extension is None:
            extension = ""

        search_pattern = fr"*{substr}*"

        if base_dir is None:
           base_dir = self.data

        results = list(base_dir.rglob(search_pattern))
        if extension is not None:
            results = [path for path in results if path.name.endswith(extension)]
        if len(results) > 1:
            print(f"Found {len(results)} matches for {substr}")

        return results
