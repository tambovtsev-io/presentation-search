import re
from dataclasses import dataclass
from typing import Optional

import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords


class RegexQueryPreprocessor:
    """Preprocesses search queries by removing common patterns and standardizing format."""

    @dataclass
    class QueryPattern:
        """Represents a query pattern with its regex and replacement."""

        pattern: str
        replacement: str = ""
        description: str = ""

    def __init__(self, remove_stopwords: bool = True) -> None:
        # Download required NLTK data if not already present
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        self._remove_stopwords = remove_stopwords
        self._stopwords = set(stopwords.words("russian"))

        # Add custom Russian stopwords
        # fmt: off
        self._custom_stopwords = {
            "разные", "какие", "когда",
            "который", "которой", "которая", "которые", "был", "была", "были",
            "также", "именно", "либо", "или", "где", "как", "какой", "какая",
            "быть", "есть", "это", "эта", "эти", "для", "при", "про"
        }
        self._stopwords.update(self._custom_stopwords)
        # fmt: on

        # Define query patterns
        self._patterns = {
            "presentation_patterns": [
                self.QueryPattern(
                    r"^в какой презентации (?:был[аи]?|рассматривали?|говорили?|обсуждали?|показывали?|рассказывали?|перечисляли?) ",
                ),
                self.QueryPattern(
                    r"^в презентации (?:был[аио]?|рассматривал?|говорил?|обсуждал?|показывал?|сравнивал?)(?:и?|ась|ось|а) ",
                ),
                self.QueryPattern(
                    r"^презентаци(?:я|и) (?:про|с|в которой|где|со?) ",
                ),
            ],
            "slide_patterns": [
                self.QueryPattern(
                    r"(?:на )?слайд(?:е|ы)? (?:с|был[аи]?|про|где|со)? ",
                ),
                # self.QueryPattern(
                #     r"слайд(?:ы)? с заголовк(?:ом|ами) ",
                # ),
            ],
            "question_patterns": [
                self.QueryPattern(
                    r"^где (?:был[аи]?|обсуждали?|говорили про) ",
                ),
                self.QueryPattern(
                    r"^о чем (?:рассказывал[аи]?|говорил[аи]?) ",
                ),
            ],
        }

        # Compile patterns
        self._compiled_patterns = {}
        for category, patterns in self._patterns.items():
            self._compiled_patterns[category] = [
                re.compile(p.pattern, re.IGNORECASE) for p in patterns
            ]

    @property
    def id(self):
        return self.__class__.__name__

    def remove_stopwords_from_text(self, text: str) -> str:
        """Remove stopwords while preserving protected terms."""
        tokens = text.split()
        filtered_tokens = [
            token for token in tokens if token.lower() not in self._stopwords
        ]
        return " ".join(filtered_tokens)

    def clean_query(self, query: str) -> str:
        """
        Remove common patterns, stopwords, and standardize the query.

        Args:
            query: Input search query

        Returns:
            Cleaned query with removed patterns and standardized format
        """
        # Convert to lowercase ? and remove punctuation
        # query = query.lower().strip()
        query = query.strip()
        query = re.sub(r"[?,!.]", "", query)

        # Apply all pattern categories
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                query = pattern.sub("", query)

        # Remove extra spaces
        query = re.sub(r"\s+", " ", query).strip()

        # Remove stopwords if enabled
        if self._remove_stopwords:
            query = self.remove_stopwords_from_text(query)

        return query

    def __call__(self, query, *args, **kwargs):
        return self.clean_query(query, *args, **kwargs)


if __name__ == "__main__":

    import fire

    load_dotenv()

    class CLI:
        """Command line interface for QueryPreprocessor."""

        def __init__(self):
            self.preprocessor = RegexQueryPreprocessor()

        def clean(self, *queries: str, remove_stopwords: bool = True) -> None:
            """
            Clean queries and show original->cleaned pairs.

            Args:
                queries: Single query string or list of queries
                remove_stopwords: Whether to remove stopwords
            """
            self.preprocessor._remove_stopwords = remove_stopwords

            # Process each query
            print("Original -> Cleaned")
            print("-" * 50)
            for query in queries:
                cleaned = self.preprocessor.clean_query(query)
                print(f"{query} -> \033[94m{cleaned} \033[0m")

        def clean_gsheets(
            self,
            sheet_id: Optional[str] = None,
            gid: Optional[str] = None,
            remove_stopwords: bool = True,
        ):
            from src.config.spreadsheets import load_spreadsheet

            df = load_spreadsheet(sheet_id, gid)
            questions = df["question"]
            return self.clean(*questions, remove_stopwords=remove_stopwords)

    # Start CLI
    fire.Fire(CLI)
