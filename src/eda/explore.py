from collections import Counter
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Set

import fitz  # PyMuPDF
import pandas as pd

from src.chains import PresentationAnalysis
from src.config.navigator import Navigator


class PresentationMetrics:
    """Class to handle various presentation metrics calculations."""

    def __init__(self, pdf_path: Path):
        """Initialize with PDF path and open document."""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def get_page_metrics(self, page_num: int) -> Dict:
        """
        Get comprehensive metrics for a specific page.

        Returns:
            Dictionary containing page metrics (image count, text length, size)
        """
        page = self.doc[page_num]
        return dict(
            image_count=len(page.get_images()),
            n_words=len(page.get_text().strip().split()),
            size=(page.rect.width, page.rect.height)
        )

    def get_all_metrics(self) -> List[Dict]:
        """
        Calculate metrics for all pages in the presentation.

        Returns:
            List of dictionaries with metrics for each page
        """
        metrics = []
        for page_num in range(len(self.doc)):
            page_metrics = self.get_page_metrics(page_num)
            page_metrics.update(dict(
                page_num=page_num,
                pdf_path=str(self.pdf_path)
            ))
            metrics.append(page_metrics)
        return metrics

    def __del__(self):
        """Ensure proper document closure."""
        if hasattr(self, "doc"):
            self.doc.close()


def parse_pdf_directory(
    root_dir: str | Path,
    topic_first: bool = True,
    include_datasets: Optional[Set[str]] = None,
    exclude_datasets: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Your existing parse_pdf_directory function with metrics integration."""
    if include_datasets and exclude_datasets:
        raise ValueError("Cannot specify both include_datasets and exclude_datasets")

    pdf_files: List[Dict] = []
    root = Path(root_dir)

    for path in root.rglob("*.pdf"):
        rel_path = path.relative_to(root)
        parts = list(rel_path.parts)

        # Get dataset name for filtering (either first or second part depending on topic_first)
        dataset_name = parts[1] if topic_first else parts[0]

        # Apply dataset filters
        if include_datasets and dataset_name not in include_datasets:
            continue
        if exclude_datasets and set(parts).intersection(set(exclude_datasets)):
            continue

        # Initialize empty dict for file info
        pdf_info = dict(filename=parts.pop(), relative_path=str(rel_path))

        if topic_first:
            pdf_info["topic"] = parts.pop(0)

        pdf_info["dataset"] = parts.pop(0)
        pdf_info["nav"] = "/".join(parts) if parts else ""

        try:
            metrics = PresentationMetrics(path)
            all_metrics = metrics.get_all_metrics()

            # Calculate aggregated metrics
            pdf_info["num_pages"] = len(all_metrics)
            pdf_info["total_images"] = sum(m["image_count"] for m in all_metrics)
            pdf_info["total_n_words"] = sum(m["n_words"] for m in all_metrics)

            # Get page sizes
            page_sizes = [(m["size"][0], m["size"][1]) for m in all_metrics]
            common_size = Counter(page_sizes).most_common(1)[0][0]
            pdf_info["page_width"] = common_size[0]
            pdf_info["page_height"] = common_size[1]

            # Handle varying sizes
            unique_sizes = set(page_sizes)
            pdf_info["varying_sizes"] = str(unique_sizes) if len(unique_sizes) > 1 else ""

        except Exception as e:
            pdf_info.update(dict(
                num_pages=0,
                total_images=0,
                total_text_length=0,
                page_width=0,
                page_height=0,
                varying_sizes=""
            ))

        pdf_files.append(pdf_info)

    return pd.DataFrame(pdf_files)

def get_pres_analysis_df(base: Path = Navigator().interim) -> pd.DataFrame:
    descriptions: List[Dict] = []
    for f in base.rglob("*.json"):
        pres = PresentationAnalysis.load(f)
        for slide in pres.slides:
            descriptions.append(
                dict(
                    pres_path=slide.pdf_path,
                    pres_title=pres.name,
                    page=slide.page_num,
                    # Unparsed text
                    llm_output=slide.llm_output,
                    # Parsed texts
                    text_content=slide.parsed_output.text_content,
                    visual_content=slide.parsed_output.visual_content,
                    topic_overview=slide.parsed_output.general_description.topic_overview,
                    conclusions_and_insights=slide.parsed_output.general_description.conclusions_and_insights,
                    layout_and_composition=slide.parsed_output.general_description.layout_and_composition,
                    # Tokens
                    completion_tokens=slide.response_metadata["token_usage"]["completion_tokens"],
                    prompt_tokens=slide.response_metadata["token_usage"]["prompt_tokens"],
                )
            )
    df = pd.DataFrame(descriptions)
    return df


def calculate_image_tokens(width: int, height: int):
    # Source: this openai thread: https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318/6
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048

    if width >= height and height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width and width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)

    return total_tokens


def tokens2price(tokens: int, cost_per_1k_tokens: float = 0.00015):
    # Token prices: https://openai.com/api/pricing/
    return tokens / 1000 * cost_per_1k_tokens
