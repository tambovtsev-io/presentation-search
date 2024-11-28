from typing import List, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BaseScorer(BaseModel):
    """Base class for scoring mechanisms.
    Scoring is an abstraction over distances returned from ChromaDB.
    """

    @property
    def id(self) -> str:
        """Unique identifier for the scoring method"""
        return self.__class__.__name__.lower()

    def compute_score(self, distances: List[float]) -> float:
        """Compute aggregated score from distances"""
        return float(np.min(distances))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MinScorer(BaseScorer):
    """Min scorer"""

    def compute_score(self, distances: List[float]) -> float:
        return float(np.min(distances))


class WeightedScorer(BaseScorer):
    """Weighted mean scoring. First elements give more impact"""

    def compute_score(self, distances: List[float]) -> float:
        dists = np.array(distances)
        weights = np.arange(len(dists))[::-1] + 1  # weights are inversed indices
        return (dists * weights).sum() / weights.sum()


class HyperbolicScorer(BaseScorer):
    """Scorer with factor adjustment based on number of slides"""

    k: float = 2.0
    p: float = 3.0

    @property
    def id(self) -> str:
        return f"hyperbolic_k{self.k}_p{self.p}"

    def compute_score(self, distances: List[float]) -> float:
        """Adjust weighted score with hyperbolic function."""
        dists = np.array(distances)
        n = len(distances)

        # Compute weighted score
        weights = np.arange(n)[::-1] + 1
        weighted_score = (dists * weights).sum() / weights.sum()

        # Apply adjustment factor
        factor = -self.k * n / (1 - self.p * n)
        return factor * weighted_score


ScorerTypes = Union[BaseScorer, MinScorer, WeightedScorer, HyperbolicScorer]


class ScoringFactory:
    """Factory for creating scorer instances"""

    @staticmethod
    def create_default() -> BaseScorer:
        """Create default scorer"""
        return HyperbolicScorer()

    @staticmethod
    def create_from_id(scorer_id: str) -> BaseScorer:
        """Create scorer from identifier string"""
        if scorer_id == "weightedscorer":
            return WeightedScorer()
        elif scorer_id.startswith("Hyperbolic"):
            k = float(scorer_id.split("k")[1].split("_")[0])
            p = float(scorer_id.split("p")[1])
            return HyperbolicScorer(k=k, p=p)
        raise ValueError(f"Unknown scorer id: {scorer_id}")
