from typing import Union, List, Optional, Iterable, Any, Tuple, Literal, Callable
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod
from importlib.metadata import version, PackageNotFoundError

from .similarity import damerau_levenshtein_similarity
from .similarity import get_similarity_function

__all__ = [
    "__version__",
    "SimilarityStrategy",
    "OpenAIEmbeddingSimilarityStrategy",
    "PairwiseSimilarityStrategy",
    "jellyjoin",
]


# set the version dynamically
try:
    __version__ = version("jellyjoin")
except PackageNotFoundError:
    __version__ = "0.0.0"


identity = lambda x: x

class SimilarityStrategy(ABC):
    @abstractmethod
    def __call__(self, left_texts: Iterable[str], right_texts: Iterable[str]) -> np.ndarray:
        """
        Computes the NxM similarity matrix between N left_texts and M right_texts.
        """
        pass


class OpenAIEmbeddingSimilarityStrategy(SimilarityStrategy):
    def __init__(
        self,
        client,
        embedding_model: str = "text-embedding-3-large",
        preprocessor: Optional[Callable[[str], str]] = identity,
    ):
        """
        Uses an OpenAI embedding model (text-embedding-3-large by default) to
        calculate the embeddings, then uses a matrix product to quickly 
        calculate all cosine similarities. OpenAI embeddings are already
        normalized, so this inner product is the same as the cosine similarity.
        """
        self.client = client
        self.embedding_model = embedding_model
        self.preprocessor = preprocessor

    def __call__(self, left_texts: Iterable[str], right_texts: Iterable[str]) -> np.ndarray:
        """
        Compute an NxM matrix of similarities using an embedding model.
        """
        left_texts = [ self.preprocessor(text) for text in left_texts ]
        right_texts = [ self.preprocessor(text) for text in right_texts ]

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
        Helper function to get embeddings from the OpenAI client.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float",
        )
        vectors = [ np.array(e.embedding) for e in response.data ]
        return np.stack(vectors)


class PairwiseSimilarityStrategy(SimilarityStrategy):
    def __init__(
        self,
        similarity_func: Callable[[str, str], float] = None,
        preprocessor: Optional[Callable[[str], str]] = identity,
    ):
        """
        preprocessor: A callable that preprocesses each input string (e.g., soundex or metaphone).
        similarity_func: A callable that computes similarity between two strings (e.g., jellyfish.jaro_winkler).
        """
        self.preprocessor = preprocessor

        if similarity_func is None:
            self.similarity_func = damerau_levenshtein_similarity
        else:
            self.similarity_func = get_similarity_function(similarity_func)

    def __call__(self, left_texts: List[str], right_texts: List[str]) -> np.ndarray:
        """
        Compute an NxM matrix of similarities using the specified preprocessor and similarity function.
        """
        size= (len(left_texts), len(right_texts))
        similarity_matrix = np.zeros(size)

        for row, left_text in enumerate(left_texts):
            left = self.preprocessor(left_text)
            for column, right_text in enumerate(right_texts):
                right = self.preprocessor(right_text)
                similarity_matrix[row, column] = self.similarity_func(right, left)

        return similarity_matrix


def get_automatic_similarity_strategy() -> SimilarityStrategy:
    """
    Instantiate the `OpenAIEmbeddingSimilarityStrategy`, if possible, or
    default to `PairwiseSimilarityStrategy` with the Damerau-levenshtein.
    """
    try:
        import openai
        # will usually succeed if OPENAI_API_KEY is defined
        client = openai.OpenAI()
        return OpenAIEmbeddingSimilarityStrategy(client)
    except:
        pass
    return PairwiseSimilarityStrategy()
        

def find_extra_assignments(similarity_matrix, unassigned, threshold, transpose=False):
    """
    Scans the similarity matrix for matches that are currently unassigned but
    above the threshold.
    """
    if transpose:
        similarity_matrix = similarity_matrix.T
    
    extra_assignments = []
    for row in unassigned:
        column = np.argmax(similarity_matrix[row, :])
        score = similarity_matrix[row, column]
        if score > threshold:
            if transpose:
                row, column = column, row
            extra_assignments.append((row, column, score))
            
    return extra_assignments


def all_extra_assignments(
    allow_many:str,
    assignments: np.ndarray,
    similarity_matrix: np.ndarray,
    threshold: float
) -> list[int]:
    """
    Finds all extra assignments left, right, or both. This allows for
    many-to-one, one-to-many, and many-to-many matches respectively.
    """
    new_assignments = []
    
    n_left, n_right = similarity_matrix.shape
    
    # For each unassigned right item, find best left match if above threshold
    if allow_many in ["right", "both"]:
        unassigned_right = list(set(range(n_right)) - set(a[1] for a in assignments))
        extra_assignments = find_extra_assignments(
            similarity_matrix,
            unassigned_right,
            threshold,
            transpose=True
        )
        new_assignments.extend(extra_assignments)
    
    # For each unassigned left item, find best right match if above threshold
    if allow_many in ["left", "both"]:
        unassigned_left = list(set(range(n_left)) - set(a[0] for a in assignments))
        extra_assignments = find_extra_assignments(
            similarity_matrix,
            unassigned_left,
            threshold,
            transpose=False
        )
        new_assignments.extend(extra_assignments)
    return new_assignments


def double_join(
    left: pd.DataFrame,
    middle: pd.DataFrame,
    right: pd.DataFrame,
    how: Literal["inner", "left", "right", "outer"]
) -> pd.DataFrame:
    """
    Joins three dataframes together, with the associations in the middle.
    """
    left_how = "outer" if how in ["left", "outer"] else "left"
    right_how = "outer" if how in ["right", "outer"] else "left"

    # Join with original dataframes
    intermediate_df = middle.merge(
        left.reset_index(drop=True),
        left_on="Left",
        right_index=True,
        how=left_how,
    )
    return intermediate_df.merge(
        right.reset_index(drop=True),
        left_on="Right",
        right_index=True, 
        suffixes=('_left', '_right'),
        how=right_how,
    )


def jellyjoin(
    left: Union[pd.DataFrame, Iterable],
    right: Union[pd.DataFrame, Iterable],
    left_column: Optional[str] = None,
    right_column: Optional[str] = None,
    similarity_strategy: Optional[Callable] = None,
    threshold: float = 0.0,
    allow_many:  Literal[None, "left", "right", "both"] = None,
    how: Literal["inner", "left", "right", "outer"] = "inner",
) -> pd.DataFrame:
    """
    Join dataframes or lists based on semantic similarity.
    
    Args:
        left: Left dataframe or iterable of strings
        right: Right dataframe or iterable of strings
        left_column: Column name to use for left dataframe (required if left is DataFrame)
        right_column: Column name to use for right dataframe (required if right is DataFrame)
        threshold: Minimum similarity score to consider a match (default: 0.5)
        allow_many: 
    
    Returns:
        DataFrame with joined data sorted by (Left, Right) indices
    """
    if similarity_strategy is None:
        similarity_strategy = get_automatic_similarity_strategy()
    
    # Convert inputs to dataframes if they aren't already
    if not isinstance(left, pd.DataFrame):
        left = pd.DataFrame({left_column or "Left Value": list(left)})
    if not isinstance(right, pd.DataFrame):
        right = pd.DataFrame({right_column or "Right Value": list(right)})

    # default to the first column if not explicitly named
    if not left_column:
        left_column = left.columns[0]
    if not right_column:
        right_column = right.columns[0]
    
    # Calculate similarity matrix
    similarity_matrix = similarity_strategy(left[left_column], right[right_column])
    
    # Find optimal assignments using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    scores = similarity_matrix[row_indices, col_indices]
    
    # Filter by threshold
    mask = scores > threshold
    assignments = list(zip(row_indices[mask], col_indices[mask], scores[mask]))
    
    if allow_many:
        extra_assignments = all_extra_assignments(
            allow_many,
            assignments,
            similarity_matrix,
            threshold
        )
        assignments.extend(extra_assignments)
    
    # Create dataframe from assignments
    assignment_df = pd.DataFrame(assignments, columns=["Left", "Right", "Similarity"])

    # join all three data frames together
    result = double_join(left, assignment_df, right, how)
    
    # Sort and reset index
    result = result.sort_values(by=["Left", "Right"]).reset_index(drop=True)
    
    return result



