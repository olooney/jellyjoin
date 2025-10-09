from importlib.metadata import version, PackageNotFoundError

from .strategy import (
    SimilarityStrategy,
    OpenAIEmbeddingSimilarityStrategy,
    PairwiseSimilarityStrategy,
)
from .join import jellyjoin
from .type_definitions import (
    SimilarityStrategyCallable,
    SimilarityCallable,
    PreprocessorCallable,
)

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
