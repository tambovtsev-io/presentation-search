import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Navigator:
    """Project paths manager"""

    root: Path = Path(__file__).parents[2]

    def __post_init__(self):
        """Initialize all project paths"""

        # Root paths
        self.log = self.root / "log"

        # Data Paths
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"

        self.eval = self.processed / "eval"
        self.eval_artifacts = self.eval / "artifacts"
        self.eval_runs = self.eval / "runs"

        # src paths
        self.src = self.root / "src"
        self.chains = self.src / "chains"
        self.prompt_stor = self.chains / "prompt_stor"

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
        return_first: bool = True,
    ) -> Optional[Union[List[Path], Path]]:
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

        search_pattern = rf"*{substr}*"

        if base_dir is None:
            base_dir = self.data

        # find results matching pattern
        results = base_dir.rglob(search_pattern)

        # remove directories from the results
        results = [path for path in results if path.is_file()]

        # sort by length so that the shortest is the first
        # thus we avoid picking modified file
        results = list(
            sorted(
                results,
                key=lambda path: len(path.name),
            )
        )

        if extension is not None:
            results = [path for path in results if path.name.endswith(extension)]
        if len(results) > 1:
            logger.info(f"Found {len(results)} matches for {substr}")

        if not results:
            return None

        return results[0] if return_first else results

    def get_relative_path(self, abs_path: Path):
        if abs_path.is_absolute():
            return abs_path.relative_to(self.root)
        return abs_path

    def get_absolute_path(self, rel_path: Path):
        return self.root / rel_path
