import importlib.metadata as _imd

from ._join import (
    DROP,
    Jelly,
    jellyjoin,
)
from .similarity import (
    damerau_levenshtein_similarity,
    get_similarity_function,
    hamming_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_similarity,
)
from .strategy import (
    EmbeddingStrategy,
    NomicEmbeddingStrategy,
    OllamaEmbeddingStrategy,
    OpenAIEmbeddingStrategy,
    PairwiseStrategy,
    SimilarityStrategy,
    get_automatic_strategy,
    get_similarity_strategy,
)

__all__ = [
    "jellyjoin",
    "Jelly",
    "DROP",
    "hamming_similarity",
    "levenshtein_similarity",
    "damerau_levenshtein_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "get_similarity_function",
    "SimilarityStrategy",
    "EmbeddingStrategy",
    "OpenAIEmbeddingStrategy",
    "NomicEmbeddingStrategy",
    "OllamaEmbeddingStrategy",
    "PairwiseStrategy",
    "get_similarity_strategy",
    "get_automatic_strategy",
    "__version__",
]

# set the version dynamically
try:
    __version__ = _imd.version("jellyjoin")
except _imd.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

del _imd
