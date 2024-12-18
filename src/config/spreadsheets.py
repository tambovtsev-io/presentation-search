import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from gspread_pandas import Client, Spread
from gspread_pandas.conf import get_config

from src.config import Config


def load_spreadsheet(
    sheet_id: Optional[str] = None, gid: Optional[str] = None
) -> pd.DataFrame:
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


class GoogleSpreadsheetManager:
    def __init__(
        self,
        spread_id: str,
        google_config_dir: Optional[Path] = Config().navigator.root,
        google_config_fname: str = "google_config.json",
        benchmark_spreadsheet_id: Optional[str] = None,
        eval_spreadsheet_id: Optional[str] = None,
    ):
        self.spread_id = spread_id

        google_config = get_config(google_config_dir, google_config_fname)
        self.google_client = Client(config=google_config)

        # if benchmark_spreadsheet_id is None or eval_spreadsheet_id is None:
        #     load_dotenv()
        #     benchmark_spreadsheet_id = os.getenv("BENCHMARK_SPREADSHEET_ID")
        #     eval_spreadsheet_id = os.getenv("EVAL_SPREADSHEET_ID")
        #
        # self.benchmark_spreadsheet_id = benchmark_spreadsheet_id
        # self.eval_spreadsheet_id = self.eval_spreadsheet_id

    def write_spreadsheet(
        self, df: pd.DataFrame, sheet_id: str, sheet_name: str, start: str
    ) -> None:
        spread = Spread(sheet_id, config=self.google_client)
        spread.df_to_sheet(df, index=False, sheet=sheet_name, start=start, replace=True)

    def get_spread(self) -> Spread:
        return Spread(
            self.spread_id,
            client=self.google_client,
            create_spread=True,
            create_sheet=True,
        )
