from collections import Counter
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Set

import fitz  # PyMuPDF
import pandas as pd
from sqlalchemy import text
from sqlalchemy.sql.elements import CompilerElement
from sqlalchemy.sql.expression import desc

from src.chains import PresentationAnalysis
from src.config.navigator import Navigator


def parse_pdf_directory(
    root_dir: str,
    topic_first: bool = True,
    include_datasets: Optional[Set[str]] = None,
    exclude_datasets: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Parse directory of PDFs into a DataFrame using PyMuPDF (fitz).

    Args:
        root_dir: Path to root directory containing PDF files
        topic_first: If True, first folder is topic. If False, topic is not stored
        include_datasets: Set of dataset names to include. If None, include all
        exclude_datasets: Set of dataset names to exclude. If None, exclude none

    Returns:
        DataFrame with columns: [topic (optional)], dataset, nav, filename, relative_path
    """
    if include_datasets and exclude_datasets:
        raise ValueError("Cannot specify both include_datasets and exclude_datasets")

    pdf_files: List[Dict] = []
    root = Path(root_dir)

    for path in root.rglob("*.pdf"):
        rel_path = path.relative_to(root)
        parts = list(rel_path.parts)

        # Get dataset name for filtering (either first or second part depending on topic_first)
        dataset_name = parts[1] if topic_first else parts[0]

        # import pdb; pdb.set_trace()
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
            doc = fitz.open(path)
            pdf_info["num_pages"] = doc.page_count

            # Get metadata
            metadata = doc.metadata
            pdf_info["title"] = metadata.get("title", "")
            pdf_info["author"] = metadata.get("author", "")
            pdf_info["subject"] = metadata.get("subject", "")
            pdf_info["keywords"] = metadata.get("keywords", "")

            # Get all page sizes
            page_sizes = [(page.rect.width, page.rect.height) for page in doc]

            # Get most common page size
            common_size = Counter(page_sizes).most_common(1)[0][0]
            pdf_info["page_width"] = common_size[0]
            pdf_info["page_height"] = common_size[1]

            # If there are different page sizes, store them as a set
            unique_sizes = set(page_sizes)
            if len(unique_sizes) > 1:
                pdf_info["varying_sizes"] = str(unique_sizes)
            else:
                pdf_info["varying_sizes"] = ""

            doc.close()

        except Exception as e:
            pdf_info["num_pages"] = 0
            pdf_info["title"] = ""
            pdf_info["author"] = ""
            pdf_info["subject"] = ""
            pdf_info["keywords"] = ""
            pdf_info["page_width"] = 0
            pdf_info["page_height"] = 0
            pdf_info["varying_sizes"] = ""

        pdf_files.append(pdf_info)

    # Convert to DataFrame
    df = pd.DataFrame(pdf_files)

    # sort_cols = ["dataset", "nav"]
    # df = df.sort_values(sort_cols)
    return df


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
