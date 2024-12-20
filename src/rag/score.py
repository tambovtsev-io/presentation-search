import re
from typing import ClassVar, Dict, List, Pattern, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BaseScorer(BaseModel):
    """Base class for scoring mechanisms.
    Scoring is an abstraction over distances returned from ChromaDB.
    """

    @property
    def id(self) -> str:
        """Unique identifier for the scoring method"""
        return self.__class__.__name__.lower().replace("scorer", "")

    def compute_score(self, distances: List[float]) -> float:
        """Compute aggregated score from distances"""
        return float(np.min(distances))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MinScorer(BaseScorer):
    """Min scorer"""

    def compute_score(self, distances: List[float]) -> float:
        return float(np.min(distances))


class WeightedScorer(BaseScorer):
    """Weighted mean scoring.
    Idea: elements with lower distances contribute more to the result
    """

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
        class_name = super(HyperbolicScorer, self).id
        return f"{class_name}_k{self.k}_p{self.p}"

    def adjustment_factor(self, n: float) -> float:
        factor = -self.k * n / (1 - self.p * n)
        return factor

    def compute_score(self, distances: List[float]) -> float:
        n = len(distances)
        score = min(distances)
        factor = self.adjustment_factor(n)
        return factor * score


class HyperbolicWeightedScorer(HyperbolicScorer):
    def compute_score(self, distances: List[float]) -> float:
        n = len(distances)
        w_scorer = WeightedScorer()
        w_score = w_scorer.compute_score(distances)

        return self.adjustment_factor(n) * w_score


class ExponentialScorer(BaseScorer):
    """Exponentially decreases score based on amount of slides.
    Core Function:
        y = x^2 exp(-x)
    Shifted and scaled function:
        y = a + (1-a) * (x+s)^2 * exp(-x/w) / ( exp(-1/w) * (s+1) )

    The function follows these criteria:
        - Passes (1, 1) - so if we have one slide it does not affect score
        - Declines down to specified asymptote - so we do not allow hack by a lot of slides with big distance
        - Declines slowly in the beginning and more with the growth of number of matches

    Params:
        a: Asymptote - lim(y) as x -> +inf
        w: Width parameter
        s: Shift in x
    """

    a: float = 0.7  # Asymptote
    w: float = 1.7  # Width
    s: float = 2.8  # x-shift

    @property
    def id(self) -> str:
        class_name = super().id
        return f"{class_name}_a{self.a}_w{self.w}_s{self.s}"

    def adjustment_factor(self, n: float):
        a, w, s = self.a, self.w, self.s
        factor = a + (1 - a) * (n + s) ** 2 * np.exp(-n / s) / (
            (1 + s) ** 2 * np.exp(-1 / w)
        )
        return factor

    def compute_score(self, distances: List[float]) -> float:
        n = len(distances)
        score = min(distances)
        return self.adjustment_factor(n) * score


class ExponentialWeightedScorer(ExponentialScorer):
    def compute_score(self, distances: List[float]) -> float:
        n = len(distances)
        w_scorer = WeightedScorer()
        w_score = w_scorer.compute_score(distances)
        return self.adjustment_factor(n) * w_score


class StepScorer(BaseScorer):
    """Step-wise scoring based on predefined ranges.
    For each threshold a specific score value is assigned.
    Default ranges: [(1, 1), (3, 0.9), (8, 0.7)]
    """

    ranges: List[Tuple[int, float]] = Field(
        default=[(1, 1.0), (3, 0.9), (8, 0.7)],
        description="List of tuples (threshold, value)",
    )

    def compute_score(self, distances: List[float]) -> float:
        n = len(distances)
        # Get weighted score first
        w_scorer = WeightedScorer()
        weighted_score = w_scorer.compute_score(distances)

        # Apply step adjustment
        for threshold, value in sorted(self.ranges, reverse=True):
            if n >= threshold:
                return value * weighted_score
        # If no threshold matched, return last defined value
        return self.ranges[-1][1] * weighted_score

    @property
    def id(self) -> str:
        """Create unique id based on ranges"""
        return f"step_{'_'.join(f'{t}-{v}' for t, v in self.ranges)}"


class LinearScorer(BaseScorer):
    """Linear interpolation scoring based on predefined points.
    Performs piecewise linear interpolation between points.
    Default points: [(1, 1), (3, 0.9), (8, 0.7)]
    """

    points: List[Tuple[int, float]] = Field(
        default=[(1, 1), (3, 0.9), (8, 0.7)],
        description="List of points for linear interpolation",
    )

    def compute_score(self, distances: List[float]) -> float:
        n = len(distances)
        # Get weighted score first
        w_scorer = WeightedScorer()
        weighted_score = w_scorer.compute_score(distances)

        # Handle boundary cases
        if n <= self.points[0][0]:
            return self.points[0][1] * weighted_score
        if n >= self.points[-1][0]:
            return self.points[-1][1] * weighted_score

        # Find and interpolate relevant segment
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]

            if x1 <= n <= x2:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                factor = m * n + b
                return factor * weighted_score

        return weighted_score  # fallback

    @property
    def id(self) -> str:
        """Create unique id based on interpolation points"""
        return f"linear_{'_'.join(f'{x}-{y}' for x, y in self.points)}"


ScorerTypes = Union[
    BaseScorer,
    MinScorer,
    WeightedScorer,
    HyperbolicScorer,
    HyperbolicWeightedScorer,
    ExponentialScorer,
    ExponentialWeightedScorer,
    StepScorer,
    LinearScorer,
]


class ScorerFactory(BaseModel):
    """Factory for creating scorer instances from specifications using regex patterns"""

    patterns: ClassVar[Dict[str, Pattern]] = {
        "min": re.compile(r"^min$"),
        "weighted": re.compile(r"^weighted$"),
        "hyperbolic": re.compile(
            r"""
             ^hyperbolic          # Base name
             (?:_weighted)?       # Optional weighted variant
             _k(?P<k>\d+\.?\d*)   # k parameter (float)
             _p(?P<p>\d+\.?\d*)$  # p parameter (float)
             """,
            re.VERBOSE,
        ),
        "exponential": re.compile(
            r"""
             ^exponential         # Base name
             (?:_weighted)?       # Optional weighted variant
             _a(?P<a>\d+\.?\d*)   # a parameter (float)
             _w(?P<w>\d+\.?\d*)   # w parameter (float)
             _s(?P<s>\d+\.?\d*)$  # s parameter (float)
             """,
            re.VERBOSE,
        ),
        "step": re.compile(
            r"""
             ^step_               # Base name
             (?P<points>          # Start points capture
                 (?:              # Start group for each point
                     \d+          # x value (int)
                     -            # separator
                     \d+\.?\d*    # y value (float)
                     _?           # Optional trailing underscore
                 )+               # One or more points
             )$                   # End points capture
             """,
            re.VERBOSE,
        ),
        "linear": re.compile(
            r"""
             ^linear_             # Base name
             (?P<points>          # Start points capture
                 (?:              # Start group for each point
                     \d+          # x value (int)
                     -            # separator
                     \d+\.?\d*    # y value (float)
                     _?           # Optional trailing underscore
                 )+               # One or more points
             )$                   # End points capture
             """,
            re.VERBOSE,
        ),
    }

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _parse_points(points_str: str) -> List[Tuple[int, float]]:
        """Parse points string into list of tuples.
        Format: "1-1.0_3-0.9_8-0.7" -> [(1, 1.0), (3, 0.9), (8, 0.7)]
        """
        points = []
        for point in points_str.rstrip("_").split("_"):
            x, y = point.split("-")
            points.append((int(x), float(y)))
        return points

    @staticmethod
    def create_default() -> BaseScorer:
        """Create default scorer"""
        return HyperbolicScorer()

    @classmethod
    def create_from_id(cls, scorer_id: str) -> BaseScorer:
        """Create scorer from identifier string using regex patterns

        Args:
            scorer_id: Scorer identifier string
                Examples:
                - "min"
                - "hyperbolic_k2.0_p3.0"
                - "hyperbolic_weighted_k2.0_p3.0"
                - "exponential_a0.7_w1.7_s2.8"
                - "step_1-1.0_3-0.9_8-0.7"
                - "linear_1-1.0_3-0.9_8-0.7"

        Returns:
            Configured scorer instance

        Raises:
            ValueError: If scorer_id format is invalid
        """
        scorer_id = scorer_id.lower()

        for scorer_type, pattern in cls.patterns.items():
            match = pattern.match(scorer_id)
            if not match:
                continue

            if scorer_type == "min":
                return MinScorer()

            elif scorer_type == "weighted":
                return WeightedScorer()

            elif scorer_type == "hyperbolic":
                k = float(match.group("k"))
                p = float(match.group("p"))
                return (
                    HyperbolicWeightedScorer(k=k, p=p)
                    if "_weighted" in scorer_id
                    else HyperbolicScorer(k=k, p=p)
                )

            elif scorer_type == "exponential":
                a = float(match.group("a"))
                w = float(match.group("w"))
                s = float(match.group("s"))
                return (
                    ExponentialWeightedScorer(a=a, w=w, s=s)
                    if "_weighted" in scorer_id
                    else ExponentialScorer(a=a, w=w, s=s)
                )

            elif scorer_type in ["step", "linear"]:
                points = cls._parse_points(match.group("points"))
                return (
                    StepScorer(ranges=points)
                    if scorer_type == "step"
                    else LinearScorer(points=points)
                )

        raise ValueError(f"Invalid scorer id format: {scorer_id}")

    @classmethod
    def parse_scorer_specs(cls, specs: List[str]) -> List[BaseScorer]:
        """Parse scorer specifications into scorer instances

        Args:
            specs: List of scorer specifications in format:
                - Simple: "min", "weighted"
                - Parameterized: "hyperbolic_k2.0_p3.0"
                - With suffix: "hyperbolic_weighted_k2.0_p3.0"
                - Step/Linear: "step_1-1.0_3-0.9_8-0.7"

        Returns:
            List of configured scorer instances
        """
        scorers = []
        for spec in specs:
            try:
                scorer = cls.create_from_id(spec.lower())
                scorers.append(scorer)
            except Exception as e:
                raise ValueError(f"Failed to parse scorer spec '{spec}': {str(e)}")
        return scorers


class ScorerPresets:
    """Predefined scorer configurations"""

    DEFAULT = ["min", "exponential_a0.7_w1.7_s2.8"]
    WEIGHTEDALL = ["weighted", "hyperbolic_weighted_k2.0_p3.0", "exponential_weighted_a0.7_w1.7_s2.8"]
    HYPERBOLIC = ["hyperbolic_k2.0_p3.0"]
    EXPONENTIAL = ["exponential_a0.7_w1.7_s2.8"]
    STEP = ["step_1-1.0_3-0.9_8-0.7"]
    LINEAR = ["linear_1-1.0_3-0.9_8-0.7"]
    ALL = DEFAULT + WEIGHTEDALL + EXPONENTIAL + STEP + LINEAR
