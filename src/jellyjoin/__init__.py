from importlib.metadata import PackageNotFoundError, version

from .join import jellyjoin
from .strategy import (
    OpenAIEmbeddingSimilarityStrategy,
    PairwiseSimilarityStrategy,
    SimilarityStrategy,
)
from .type_definitions import (
    PreprocessorCallable,
    SimilarityCallable,
    SimilarityStrategyCallable,
)

__all__ = [
    "__version__",
    "OpenAIEmbeddingSimilarityStrategy",
    "PairwiseSimilarityStrategy",
    "PreprocessorCallable",
    "SimilarityCallable",
    "SimilarityStrategy",
    "SimilarityStrategyCallable",
    "SimilarityStrategyCallable",
    "jellyjoin",
]


# set the version dynamically
try:
    __version__ = version("jellyjoin")
except PackageNotFoundError:
    __version__ = "0.0.0"
