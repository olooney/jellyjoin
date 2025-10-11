import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import List

import numpy as np
import tiktoken
from numpy.typing import DTypeLike

from .similarity import get_similarity_function
from .type_definitions import (
    PreprocessorCallable,
    SimilarityCallable,
    SimilarityIdentifier,
)

__all__ = [
    "SimilarityStrategy",
    "OpenAIEmbeddingSimilarityStrategy",
    "PairwiseSimilarityStrategy",
    "get_automatic_similarity_strategy",
]


logger = logging.getLogger(__name__)


# global used to cache the automatic strategy to prevent instantiating a new
# OpenAI client for every call.
cached_strategy = None


# identity function used as a default argument to several functions
def identity(x: str) -> str:
    return x


class SimilarityStrategy(ABC):
    @abstractmethod
    def __call__(
        self,
        left_texts: Collection[str],
        right_texts: Collection[str],
    ) -> np.ndarray:
        """
        Computes the NxM similarity matrix between N left_texts and M right_texts.
        """
        pass


class OpenAIEmbeddingSimilarityStrategy(SimilarityStrategy):
    def __init__(
        self,
        client,
        preprocessor: PreprocessorCallable = identity,
        embedding_model: str = "text-embedding-3-large",
        batch_size: int = 2048,
        max_tokens: int = 8191,
        encoding: str = "cl100k_base",
        dtype: DTypeLike = np.float32,
    ):
        """
        Uses an OpenAI embedding model (text-embedding-3-large by default) to
        calculate the embeddings, then uses a matrix product to quickly
        calculate all cosine similarities. OpenAI embeddings are already
        normalized, so this inner product is the same as the cosine similarity.
        """
        self.client = client
        self.preprocessor = preprocessor
        self.embedding_model = embedding_model
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.encoding = tiktoken.get_encoding(encoding)
        self.dtype = dtype

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

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from the OpenAI client in batches of size `self.batch_size`.
        """
        all_vectors = []

        for i in range(0, len(texts), self.batch_size):
            raw_batch = texts[i : i + self.batch_size]
            batch = [self._truncate(text) for text in raw_batch]

            # embedding API call
            logger.warning("Calling OpenAI embeddings API with %d strings.", len(batch))
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
                encoding_format="float",
            )

            # stack up vectors
            vectors = [e.embedding for e in response.data]
            all_vectors.extend(vectors)

        return np.array(all_vectors, dtype=self.dtype)

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


class PairwiseSimilarityStrategy(SimilarityStrategy):
    def __init__(
        self,
        similarity_function: SimilarityIdentifier = None,
        preprocessor: PreprocessorCallable = identity,
    ):
        """
        preprocessor: A callable that preprocesses each input string (e.g., soundex or metaphone).
        similarity_func: A callable that computes similarity between two strings (e.g., jellyfish.jaro_winkler).
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


def get_automatic_similarity_strategy() -> SimilarityStrategy:
    """
    Instantiate the `OpenAIEmbeddingSimilarityStrategy`, if possible, or
    default to `PairwiseSimilarityStrategy` with the Damerau-levenshtein.
    """
    global cached_strategy

    if cached_strategy:
        logger.debug("Using cached SimilarityStrategy.")
        return cached_strategy

    try:
        import openai

        # will usually succeed if OPENAI_API_KEY is defined
        client = openai.OpenAI()
        strategy = OpenAIEmbeddingSimilarityStrategy(client)
        cached_strategy = strategy
        logger.debug("Instantiated and cached OpenAIEmbeddingSimilarityStrategy.")

        return strategy
    except Exception:
        logger.warning(
            "OpenAI unavailable; falling back to PairwiseSimilarityStrategy.",
        )
        logging.debug("Failed to instantiate OpenAI client.", exc_info=True)

    return PairwiseSimilarityStrategy()
