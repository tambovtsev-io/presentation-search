import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from gspread_pandas import Client, Spread
from gspread_pandas.conf import get_config

from src.config import Config

logger = logging.getLogger(__name__)


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


class GoogleSpreadsheetManagerMLFlow:
    """Manages Google Spreadsheet operations for evaluation results"""

    def __init__(
        self,
        spread_id: str,
        google_config_dir: Optional[Path] = Config().navigator.root,
        google_config_fname: str = "google_config.json",
    ):
        """Initialize spreadsheet manager

        Args:
            spread_id: ID of the target spreadsheet
            google_config_dir: Directory containing Google credentials
            google_config_fname: Name of the Google credentials file
        """
        self.spread_id = spread_id
        self._config = get_config(google_config_dir, google_config_fname)
        self._client = Client(config=self._config)
        self._spread = self.get_spread()

        # Define standard sheet names and layouts
        self._summary_sheet = "Evaluation Summary"
        self._details_sheet = "Detailed Results"
        self._metrics_sheet = "Metrics History"

        # Initialize standard sheets if they don't exist
        self._init_sheets()

    def _init_sheets(self) -> None:
        """Initialize standard sheets with headers if they don't exist"""
        # Check and create summary sheet
        if self._summary_sheet not in self._spread.sheets:
            summary_headers = [
                "Timestamp",
                "Experiment",
                "Retriever",
                "Scorer",
                "Questions Count",
                "Mean Metrics",
            ]
            self._create_sheet(self._summary_sheet, summary_headers)

        # Check and create details sheet
        if self._details_sheet not in self._spread.sheets:
            details_headers = [
                "Timestamp",
                "Experiment",
                "Retriever",
                "Scorer",
                "Question",
                "Expected Presentation",
                "Expected Pages",
                "Retrieved Presentations",
                "Retrieved Pages",
                "Metric Scores",
                "Metric Explanations",
            ]
            self._create_sheet(self._details_sheet, details_headers)

        # Check and create metrics history sheet
        if self._metrics_sheet not in self._spread.sheets:
            metrics_headers = [
                "Timestamp",
                "Experiment",
                "Retriever",
                "Scorer",
                "Metric Name",
                "Mean Score",
            ]
            self._create_sheet(self._metrics_sheet, metrics_headers)

    def _create_sheet(self, sheet_name: str, headers: List[str]) -> None:
        """Create new sheet with headers

        Args:
            sheet_name: Name for the new sheet
            headers: List of column headers
        """
        try:
            spread = self._spread.find_sheet(sheet_name)
            if spread:
                logger.info(f"Using existing sheet '{sheet_name}'")
            else:
                self._spread.create_sheet(sheet_name)
                worksheet = self._spread.find_sheet(sheet_name)
                if worksheet:
                    worksheet.update([headers])
                logger.info(f"Created sheet '{sheet_name}' with headers")
        except Exception as e:
            logger.error(f"Failed to create sheet '{sheet_name}': {str(e)}")
            raise

    def get_spread(self) -> Spread:
        """Get Spread instance for the target spreadsheet"""
        return Spread(
            self.spread_id,
            client=self._client,
            create_spread=True,
            create_sheet=True,
        )

    def write_evaluation_results(
        self,
        results_df: pd.DataFrame,
        metric_values: Dict[str, List[float]],
        experiment_name: str,
    ) -> None:
        """Write evaluation results to spreadsheet

        Args:
            results_df: DataFrame with detailed evaluation results
            metric_values: Dictionary mapping metric names to score lists
            experiment_name: Name of the experiment
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retriever = results_df["retriever"].iloc[0]
            scorer = results_df["scorer"].iloc[0]

            # Write summary
            summary_row = {
                "Timestamp": timestamp,
                "Experiment": experiment_name,
                "Retriever": retriever,
                "Scorer": scorer,
                "Questions Count": len(results_df),
                "Mean Metrics": ", ".join(
                    f"{name}: {sum(values)/len(values):.3f}"
                    for name, values in metric_values.items()
                ),
            }
            self._append_rows(self._summary_sheet, [summary_row])

            # Write detailed results
            details = []
            for _, row in results_df.iterrows():
                detail_row = {
                    "Timestamp": timestamp,
                    "Experiment": experiment_name,
                    "Retriever": retriever,
                    "Scorer": scorer,
                    "Question": row["question"],
                    "Expected Presentation": row["expected_presentation"],
                    "Expected Pages": row["expected_pages"],
                    "Retrieved Presentations": row["retrieved_presentations"],
                    "Retrieved Pages": row["retrieved_pages"],
                    "Metric Scores": ", ".join(
                        f"{col.replace('metric_', '').replace('_score', '')}: {row[col]}"
                        for col in results_df.columns
                        if col.endswith("_score")
                    ),
                    "Metric Explanations": "\n".join(
                        f"{col.replace('metric_', '').replace('_explanation', '')}: {row[col]}"
                        for col in results_df.columns
                        if col.endswith("_explanation")
                    ),
                }
                details.append(detail_row)
            self._append_rows(self._details_sheet, details)

            # Write metrics history
            metrics = []
            for metric_name, values in metric_values.items():
                metrics.append(
                    {
                        "Timestamp": timestamp,
                        "Experiment": experiment_name,
                        "Retriever": retriever,
                        "Scorer": scorer,
                        "Metric Name": metric_name,
                        "Mean Score": sum(values) / len(values),
                    }
                )
            self._append_rows(self._metrics_sheet, metrics)

            logger.info(
                f"Successfully wrote evaluation results to sheets: "
                f"{self._summary_sheet}, {self._details_sheet}, {self._metrics_sheet}"
            )

        except Exception as e:
            logger.error(f"Failed to write evaluation results: {str(e)}")
            raise

    def _append_rows(self, sheet_name: str, rows: List[Dict]) -> None:
        """Append rows to specified sheet

        Args:
            sheet_name: Target sheet name
            rows: List of dictionaries representing rows
        """
        try:
            df = pd.DataFrame(rows)
            worksheet = self._spread.find_sheet(sheet_name)
            if worksheet:
                start = f"A{len(worksheet.get_all_values()) + 1}"
                self.write_spreadsheet(df, self.spread_id, sheet_name, start)
        except Exception as e:
            logger.error(f"Failed to append rows to '{sheet_name}': {str(e)}")
            raise

    def write_spreadsheet(
        self, df: pd.DataFrame, sheet_id: str, sheet_name: str, start: str
    ) -> None:
        """Write DataFrame to spreadsheet at specified location

        Args:
            df: DataFrame to write
            sheet_id: Target spreadsheet ID
            sheet_name: Target sheet name
            start: Starting cell (e.g. 'A1')
        """
        spread = Spread(sheet_id, client=self._client)
        spread.df_to_sheet(
            df, index=False, headers=False, sheet=sheet_name, start=start, replace=False
        )
