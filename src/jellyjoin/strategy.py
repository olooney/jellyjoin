import logging
from collections.abc import Collection
from typing import List, Literal

import numpy as np
import tiktoken
from numpy.typing import ArrayLike, DTypeLike

from .similarity import get_similarity_function
from .typing import (
    PreprocessorCallable,
    SimilarityCallable,
    SimilarityLike,
    Strategy,
    StrategyCallable,
    StrategyLike,
)

__all__ = [
    "Strategy",
    "EmbeddingStrategy",
    "OpenAIEmbeddingStrategy",
    "NomicEmbeddingStrategy",
    "PairwiseStrategy",
    "get_similarity_strategy",
    "get_automatic_strategy",
]


logger = logging.getLogger(__name__)


# global used to cache the automatic strategy to prevent instantiating a new
# OpenAI client for every call.
_cached_strategy = None


# identity function used as a default argument to several functions
def identity(x: str) -> str:
    return x


class EmbeddingStrategy(Strategy):
    """
    Base class for Strategies that use an embedding model to calculate
    similarities. Implement the abstract `call_embedding_api()` method
    to provide a concrete implmenetation.

    The EmbeddingStrategy will ensure that `call_embedding_api()` is
    never called with at least one but not more than `batch_size`
    strings, and that each of the these strings will have fewer than
    `max_tokens` tokens according to the tiktoken `encoding`.
    """

    def __init__(
        self,
        preprocessor: PreprocessorCallable = identity,
    ):
        """
        Initialize an EmbeddingStrategy.

        Parameters
        ----------
        preprocessor : PreprocessorCallable, optional
            A callable applied to each text before embedding. Defaults to `identity`.
        """
        self.preprocessor = preprocessor

    def __call__(
        self,
        left_texts: Collection[str],
        right_texts: Collection[str],
    ) -> np.ndarray:
        """
        Compute an NxM matrix of similarities using an embedding model.
        """
        left_texts = [self.preprocessor(text) for text in left_texts]
        right_texts = [self.preprocessor(text) for text in right_texts]

        if not left_texts:
            return np.zeros((0, len(right_texts)))
        elif not right_texts:
            return np.zeros((len(left_texts), 0))

        # compute embeddings
        left_embeddings = self.embed(left_texts)
        right_embeddings = self.embed(right_texts)

        # calculate similarity matrix
        similarity_matrix = left_embeddings @ right_embeddings.T

        return similarity_matrix


class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """
    Uses an OpenAI embedding model (text-embedding-3-large by default) to
    calculate the embedding vectors used for the embedding vector matrix
    multiplication similarity strategy.
    """

    def __init__(
        self,
        client=None,
        embedding_model: str = "text-embedding-3-large",
        preprocessor: PreprocessorCallable = identity,
        batch_size: int = 2048,
        max_tokens: int = 6000,
        encoding: str = "cl100k_base",
        dtype: DTypeLike = np.float32,
    ):
        """
        client:
            An OpenAI API client instance used to perform embedding requests.
        preprocessor:
            A callable applied to each text before embedding. Defaults to `identity`.
        embedding_model:
            Name of the OpenAI embedding model to use. Defaults to `"text-embedding-3-large"`.
        batch_size:
            Maximum number of strings to send in one API call. Defaults to 2048.
        max_tokens:
            Maximum number of tokens allowed per string according to the encoding. Defaults to 8191.
        encoding:
            Name of the tiktoken encoding to use for token counting. Defaults to `"cl100k_base"`.
        dtype:
            Data type of the returned embedding arrays. Defaults to `np.float32`.
        """
        if client is None:
            import openai

            self.client = openai.OpenAI()
        else:
            self.client = client

        self.embedding_model = embedding_model
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.encoding = tiktoken.get_encoding(encoding)
        self.dtype = dtype

        super().__init__(preprocessor=preprocessor)

    def _truncate(
        self,
        text: str,
    ) -> str:
        """
        Return the text if it is short enough, otherwise truncate it.
        """
        # tokens are always at least one character, so we can short-circuit if
        # the number of characters is less than max_tokens
        if len(text) <= self.max_tokens:
            return text

        tokens = self.encoding.encode(text)

        # we're ok; return a reference to the original string
        if len(tokens) <= self.max_tokens:
            return text

        # truncate and re-encode to string.
        logger.debug("Truncating %d tokens to %d.", len(tokens), self.max_tokens)
        truncated = tokens[: self.max_tokens]
        return self.encoding.decode(truncated)

    def _call_embedding_api(self, batch: List[str]) -> List[ArrayLike]:
        logger.debug("Calling OpenAI embeddings API with %d strings.", len(batch))

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=batch,
            encoding_format="float",
        )

        return [e.embedding for e in response.data]

    def embed(self, texts: Collection[str]) -> np.ndarray:
        """
        Get embeddings from the OpenAI client in batches of size `self.batch_size`.
        Returns a matrix of shape `(len(texts), D)`, where `D` is the number of
        dimensions of the embedding vectors.
        """
        all_vectors = []

        for i in range(0, len(texts), self.batch_size):
            raw_batch = texts[i : i + self.batch_size]
            batch = [self._truncate(text) for text in raw_batch]

            vectors = self._call_embedding_api(batch)
            all_vectors.extend(vectors)

        return np.array(all_vectors, dtype=self.dtype)


class NomicEmbeddingStrategy(EmbeddingStrategy):
    """
    Uses a Nomic embedding model locally (via GPT4All backend) to compute embeddings.
    """

    TaskTypeLiteral = Literal[
        "search_document", "search_query", "classification", "clustering"
    ]

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text-v1.5",
        preprocessor: PreprocessorCallable = identity,
        task_type: TaskTypeLiteral = "search_document",
        dimensionality: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        allow_download: bool = True,
        dtype: DTypeLike = np.float32,
    ):
        """
        Local Nomic embeddings (no network calls after first model download).

        Parameters
        ----------
        embedding_model:
            Nomic model name, e.g. "nomic-embed-text-v1.5".
        preprocessor:
            A callable applied to each text before embedding. Defaults to `identity`.
        task_type:
            One of {"search_document","search_query","classification","clustering"}.
        dimensionality : int | None
            Output embedding size (v1.5 supports 64..768). None = model default.
        device:
            Device for local mode (e.g., 'gpu'). `None` for automatic.
        allow_download:
            True if nomic should download and cache the model automatically if
            it's not available locally.
        dtype:
            Data type of the returned embedding arrays. Defaults to `np.float32`.
        """
        # deferred import of nomic.
        import nomic.embed as nomic_embed

        self.nomic_embed = nomic_embed

        self.embedding_model = embedding_model
        self.task_type = task_type
        self.dimensionality = dimensionality
        self.device = device
        self.allow_download = bool(allow_download)
        self.dtype = dtype

        super().__init__(preprocessor=preprocessor)

    def embed(self, texts: Collection[str]) -> np.ndarray:
        result = self.nomic_embed.text(
            texts=texts,
            model=self.embedding_model,
            task_type=self.task_type,
            inference_mode="local",
            long_text_mode="truncate",
            device=self.device,
            dimensionality=self.dimensionality,
            allow_download=self.allow_download,
        )

        return np.array(result["embeddings"], dtype=self.dtype)


class PairwiseStrategy(Strategy):
    """
    A simpler, non-vectorized similarity strategy which calls a
    similarity function for every possible pair of strings from the left
    and right sides.
    """

    def __init__(
        self,
        similarity_function: SimilarityLike = None,
        preprocessor: PreprocessorCallable = identity,
    ):
        """
        similarity_func:
            A callable that computes similarity between two strings (e.g., jellyfish.jaro_winkler),
            or a string identifier for one of the standard similarity functions. Defaults to
            `jellyjoin.similarity.damerau_levenshtein_similarity`.
        preprocessor:
            A callable that preprocesses each input string (e.g., soundex or lowercase.).
        """
        self.preprocessor = preprocessor
        self.similarity_function: SimilarityCallable = get_similarity_function(
            similarity_function
        )

    def __call__(
        self,
        left_texts: Collection[str],
        right_texts: Collection[str],
    ) -> np.ndarray:
        """
        Compute an NxM matrix of similarities using the specified preprocessor and similarity function.
        """
        size = (len(left_texts), len(right_texts))
        similarity_matrix = np.zeros(size)

        for row, left_text in enumerate(left_texts):
            left = self.preprocessor(left_text)
            for column, right_text in enumerate(right_texts):
                right = self.preprocessor(right_text)
                similarity_matrix[row, column] = self.similarity_function(left, right)

        return similarity_matrix


def get_similarity_strategy(
    strategy: StrategyLike | None = None,
) -> Strategy | StrategyCallable:
    """
    Resolves a strategy identifier to a strategy class.

    - "openai" or "nomic" will instantiate those strategies with default arguments.
    - any other string will be passed to `PairwiseStrategy` which will
      interpret it as the name of a similarity function, e.g. 'jaro-winkler'
    - Any callable will be interpreted as a StrategyCallable and returned as-is.
    - `None` will call `get_automatic_strategy()`.
    """
    match strategy:
        case None:
            return get_automatic_strategy()
        case Strategy():
            return strategy
        case str() as strategy_name:
            strategy_name = strategy_name.strip().lower().replace("-", "_")
            if strategy_name == "openai":
                return OpenAIEmbeddingStrategy()
            elif strategy_name == "nomic":
                return NomicEmbeddingStrategy()
            else:
                try:
                    return PairwiseStrategy(strategy_name)
                except KeyError:
                    raise ValueError(
                        f"Strategy name {strategy_name!r} must be "
                        '"openai", '
                        '"nomic", '
                        "or any valid similarity function name, "
                        'e.g. "jaro_winkler"'
                    )
        case _ if callable(strategy):
            return strategy
        case _:
            raise TypeError(
                "strategy argument must be None, "
                "a strategy name, "
                "a similarity function name, "
                "a Strategy instance, "
                "or any compatible callable."
            )


def get_automatic_strategy() -> Strategy:
    """
    Tries to instantiate an similarity Strategy in this order:
        1. `OpenAIEmbeddingStrategy`
        2.`PairwiseStrategy`
    """
    global _cached_strategy

    if _cached_strategy:
        logger.debug("Using cached jellyjoin.Strategy.")
        return _cached_strategy

    try:
        strategy = OpenAIEmbeddingStrategy()
        _cached_strategy = strategy
        logger.debug("Instantiated and cached OpenAIEmbeddingStrategy.")
        return strategy
    except Exception:
        logger.warning("OpenAI unavailable; trying next strategy...")
        logging.debug("Failed to instantiate OpenAI client.", exc_info=True)

    return PairwiseStrategy()
