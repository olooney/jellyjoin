from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Callable, Literal, TypeAlias

import numpy as np


class SimilarityStrategy(ABC):
    @abstractmethod
    def __call__(
        self,
        left_texts: Collection[str],
        right_texts: Collection[str],
    ) -> np.ndarray:
        """
        Abstract Base Class for all similarity strategy classes.
        """


# function signatures
StrategyCallable: TypeAlias = Callable[[Collection[str], Collection[str]], np.ndarray]
SimilarityCallable: TypeAlias = Callable[[str, str], float]
PreprocessorCallable: TypeAlias = Callable[[str], str]

# Lists of things that can be coerced into specific types
SimilarityLiteral: TypeAlias = Literal[
    "hamming",
    "levenshtein",
    "damerau_levenshtein",
    "jaro",
    "jaro_winkler",
]
SimilarityLike: TypeAlias = SimilarityLiteral | SimilarityCallable

StrategyLiteral: TypeAlias = Literal[
    "openai",
    "nomic",
]
StrategyLike: TypeAlias = (
    SimilarityStrategy | StrategyLiteral | SimilarityLiteral | StrategyCallable
)

# types used by join
HowLiteral = Literal["inner", "left", "right", "outer"]
AllowManyLiteral = Literal["neither", "left", "right", "both"]
