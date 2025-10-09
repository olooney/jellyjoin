from importlib.metadata import version, PackageNotFoundError

from .strategy import (
    SimilarityStrategy,
    SimilarityStrategyCallable,
    OpenAIEmbeddingSimilarityStrategy,
    PairwiseSimilarityStrategy,
)
from .join import jellyjoin

__all__ = [
    "__version__",
    "SimilarityStrategy",
    "SimilarityStrategyCallable",
    "OpenAIEmbeddingSimilarityStrategy",
    "PairwiseSimilarityStrategy",
    "jellyjoin",
]


# set the version dynamically
try:
    __version__ = version("jellyjoin")
except PackageNotFoundError:
    __version__ = "0.0.0"
