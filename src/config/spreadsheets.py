import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv


def load_spreadsheet(sheet_id: Optional[str] = None, gid: Optional[str] = None) -> pd.DataFrame:
    if sheet_id is None:
        load_dotenv()
        sheet_id = os.environ.get("BENCHMARK_SPREADSHEET_ID")

    csv_load_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    )
    if gid is not None:
        csv_load_url = f"{csv_load_url}&gid={gid}"
    df = pd.read_csv(csv_load_url)
    return df
