import logging
from collections.abc import Collection
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .strategy import get_automatic_strategy
from .typing import StrategyCallable

__all__ = [
    "jellyjoin",
]

logger = logging.getLogger(__name__)

# types
HowLiteral = Literal["inner", "left", "right", "outer"]
AllowManyLiteral = Literal["neither", "left", "right", "both"]
AssignmentList = List[Tuple[int, int, float]]


def find_extra_assignments(
    similarity_matrix: np.ndarray,
    unassigned: Iterable[int],
    threshold: float,
    transpose: bool = False,
) -> AssignmentList:
    """
    Scans the similarity matrix for matches that are currently unassigned but
    above the threshold.
    """
    if transpose:
        similarity_matrix = similarity_matrix.T

    extra_assignments: AssignmentList = []
    for row in unassigned:
        column = int(np.argmax(similarity_matrix[row, :]))
        score = float(similarity_matrix[row, column])
        if score >= threshold:
            if transpose:
                row, column = column, row
            extra_assignments.append((row, column, score))

    return extra_assignments


def all_extra_assignments(
    allow_many: AllowManyLiteral,
    assignments: AssignmentList,
    similarity_matrix: np.ndarray,
    threshold: float,
) -> AssignmentList:
    """
    Finds all extra assignments left, right, or both. This allows for
    many-to-one, one-to-many, and many-to-many matches respectively.
    """
    new_assignments = []

    n_left, n_right = similarity_matrix.shape

    # For each unassigned right item, find best left match if above threshold
    if allow_many in ["right", "both"]:
        logger.debug("Searching for extra right (one-to-many) assignments.")
        unassigned_right = list(set(range(n_right)) - set(a[1] for a in assignments))
        extra_assignments = find_extra_assignments(
            similarity_matrix, unassigned_right, threshold, transpose=True
        )
        new_assignments.extend(extra_assignments)

    # For each unassigned left item, find best right match if above threshold
    if allow_many in ["left", "both"]:
        logger.debug("Searching for extra left (many-to-one) assignments.")
        unassigned_left = list(set(range(n_left)) - set(a[0] for a in assignments))
        extra_assignments = find_extra_assignments(
            similarity_matrix, unassigned_left, threshold, transpose=False
        )
        new_assignments.extend(extra_assignments)
    return new_assignments


def triple_join(
    left: pd.DataFrame,
    middle: pd.DataFrame,
    right: pd.DataFrame,
    how: HowLiteral,
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
        suffixes=("_left", "_right"),
        how=right_how,
    )


def jellyjoin(
    left: Union[pd.DataFrame, Collection],
    right: Union[pd.DataFrame, Collection],
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    strategy: Optional[StrategyCallable] = None,
    threshold: float = 0.0,
    allow_many: AllowManyLiteral = "neither",
    how: HowLiteral = "inner",
) -> pd.DataFrame:
    """
    Join dataframes or lists based on semantic similarity.

    Args:
        left: Left DataFrame or Collection of strings
        right: Right DataFrame or Collection of strings
        left_on: Column name to use for left dataframe
        right_on: Column name to use for right dataframe
        threshold: Minimum similarity score to consider a match (default: 0.0)
        allow_many: Find one-to-many assocations

    Returns:
        DataFrame with joined data sorted by (Left, Right) indices
    """
    if strategy is None:
        strategy = get_automatic_strategy()

    # Convert inputs to dataframes if they aren't already
    if not isinstance(left, pd.DataFrame):
        left = pd.DataFrame({left_on or "Left Value": list(left)})
    if not isinstance(right, pd.DataFrame):
        right = pd.DataFrame({right_on or "Right Value": list(right)})

    # default to the first column if not explicitly named
    if not left_on:
        left_on = left.columns[0]
    if not right_on:
        right_on = right.columns[0]

    # Calculate similarity matrix
    similarity_matrix = strategy(left[left_on], right[right_on])

    # Find optimal assignments using Hungarian algorithm
    logger.debug("Solving assignment problem for %s matrix.", similarity_matrix.shape)
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    scores = similarity_matrix[row_indices, col_indices]

    # Filter by threshold
    mask = scores >= threshold
    assignments = list(zip(row_indices[mask], col_indices[mask], scores[mask]))

    if allow_many != "neither":
        extra_assignments = all_extra_assignments(
            allow_many, assignments, similarity_matrix, threshold
        )
        assignments.extend(extra_assignments)

    # Create dataframe from assignments
    assignment_df = pd.DataFrame(assignments, columns=["Left", "Right", "Similarity"])

    # join all three data frames together
    result = triple_join(left, assignment_df, right, how)

    # Sort and reset index
    result = result.sort_values(by=["Left", "Right"]).reset_index(drop=True)

    return result
