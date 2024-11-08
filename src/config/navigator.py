from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class Navigator:
    """Project paths manager"""

    root: Path = Path(__file__).parents[2]

    def __post_init__(self):
        """Initialize all project paths"""
        # Data Paths
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"

        # src paths
        self.src = self.root / "src"
        self.prompts = self.src / "prompts"

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
            return_first: bool = True
        ) -> Optional[Union[List[Path], Path]] :
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

        # find results matching pattern
        results = base_dir.rglob(search_pattern)

        # remove directories from the results
        results = [path for path in results if path.is_file()]

        # sort by length so that the shortest is the first
        # thus we avoid picking modified file
        results = list(sorted(
            results,
            key=lambda path: len(path.name),
        ))


        if extension is not None:
            results = [path for path in results if path.name.endswith(extension)]
        if len(results) > 1:
            print(f"Found {len(results)} matches for {substr}")

        if not results:
            return None

        return results[0] if return_first else results
