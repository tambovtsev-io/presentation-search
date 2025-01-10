# dashboard/data_processing.py
from fractions import Fraction
from typing import Dict, List, Tuple

import pandas as pd

from src.eda.explore import calculate_image_tokens, tokens2price


def process_resolutions(
    df: pd.DataFrame, prompt_tokens: int = 300, openai_1k_token_price: float = 0.00015
) -> pd.DataFrame:
    """
    Process presentation resolutions and calculate related metrics.
    """
    df_int_res = df.assign(
        resolution=lambda df_: (
            list(
                df_[["page_width", "page_height"]]
                .astype(int)
                .round(-1)
                .itertuples(index=False, name=None)
            )
        ),
        aspect=lambda df_: (df_["resolution"].apply(lambda x: Fraction(x[0], x[1]))),
        page_vision_tokens=lambda df_: (
            df_["resolution"].apply(lambda x: calculate_image_tokens(*x))
        ),
        page_vision_tokens_mini=lambda df_: df_["page_vision_tokens"] * 33,
        page_total_tokens=lambda df_: (df_["page_vision_tokens_mini"] + prompt_tokens),
        page_cost=lambda df_: (
            df_["page_total_tokens"].apply(
                lambda x: tokens2price(x, openai_1k_token_price)
            )
        ),
        pres_total_tokens=lambda df_: df_["page_total_tokens"] * df_["num_pages"],
        pres_total_cost=lambda df_: (df_["page_cost"] * df_["num_pages"]),
    )

    resolutions = (
        df_int_res.groupby(["resolution", "aspect"])
        .aggregate(
            pres_count=("resolution", "count"),
            page_tokens=("page_vision_tokens", "first"),
        )
        .sort_values("pres_count", ascending=False)
        .reset_index()
    )

    return resolutions


def extract_dataset_from_path(path: str) -> str:
    """Extract dataset name from presentation path."""
    parts = str(path).split("/")
    return parts[3] if len(parts) > 3 else "unknown"
