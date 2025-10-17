import logging
from collections.abc import Collection
from typing import Iterable, List, Tuple, get_args

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .strategy import get_similarity_strategy
from .typing import (
    AllowManyLiteral,
    HowLiteral,
    StrategyLike,
)

__all__ = [
    "jellyjoin",
]

logger = logging.getLogger(__name__)

# internal type only used by private functions
AssignmentList = List[Tuple[int, int, float]]


def _find_extra_assignments(
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


def _all_extra_assignments(
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
        extra_assignments = _find_extra_assignments(
            similarity_matrix, unassigned_right, threshold, transpose=True
        )
        new_assignments.extend(extra_assignments)

    # For each unassigned left item, find best right match if above threshold
    if allow_many in ["left", "both"]:
        logger.debug("Searching for extra left (many-to-one) assignments.")
        unassigned_left = list(set(range(n_left)) - set(a[0] for a in assignments))
        extra_assignments = _find_extra_assignments(
            similarity_matrix, unassigned_left, threshold, transpose=False
        )
        new_assignments.extend(extra_assignments)
    return new_assignments


def _triple_join(
    left: pd.DataFrame,
    middle: pd.DataFrame,
    right: pd.DataFrame,
    how: HowLiteral,
    suffixes: Iterable,
) -> pd.DataFrame:
    """
    Joins three dataframes together, with the associations in the middle.
    """
    left_how = "outer" if how in ["left", "outer"] else "left"
    right_how = "outer" if how in ["right", "outer"] else "left"

    # ensure unique enough column names
    left_dupes = set(left.columns) & (set(middle.columns) | set(right.columns))
    right_dupes = (set(left.columns) | set(middle.columns)) & set(right.columns)
    duplicate_columns = left_dupes | right_dupes
    left = left.rename(columns={c: c + suffixes[0] for c in duplicate_columns})
    right = right.rename(columns={c: c + suffixes[1] for c in duplicate_columns})

    # Join with original dataframes
    left_middle = middle.merge(
        left.reset_index(drop=True),
        left_on=middle.columns[0],
        right_index=True,
        how=left_how,
    )
    return left_middle.merge(
        right.reset_index(drop=True),
        left_on=middle.columns[1],
        right_index=True,
        how=right_how,
    )


def _coerce_to_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | None,
    left_on: str | None,
    right_on: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    # handle the shared "on" column name
    if on:
        if left_on or right_on:
            raise ValueError('Pass only "on" or "left_on" and "right_on", not both.')
        left_on = on
        right_on = on

    # Convert inputs to dataframes if they aren't already
    if not isinstance(left, pd.DataFrame):
        left = pd.DataFrame({left_on or "Left Value": list(left)})

    if not isinstance(right, pd.DataFrame):
        right = pd.DataFrame({right_on or "Right Value": list(right)})

    # TODO: magic case where the two dataframes share exactly one column name?

    # default to joining on the first column if not explicitly named
    if not left_on:
        left_on = left.columns[0]

    if not right_on:
        right_on = right.columns[0]

    return left, right, left_on, right_on


def _validate_jellyjoin_arguments(
    suffixes: Collection,
    association_column_names: Collection,
    allow_many: AllowManyLiteral,
    how: HowLiteral,
):
    """
    Raises exception for any invalid arguments.
    """
    if len(suffixes) != 2:
        raise ValueError("Pass exactly two suffixes.")

    if len(association_column_names) != 3:
        raise ValueError(
            "Pass exactly three association_column_names: left index, right index, and similarity score."
        )

    if allow_many not in get_args(AllowManyLiteral):
        raise ValueError('allow_many must be "left", "right", "both", or "neither".')

    if how not in get_args(HowLiteral):
        raise ValueError('how argument must be "inner", "left", "right", or "outer".')


def jellyjoin(
    left: pd.DataFrame | Collection,
    right: pd.DataFrame | Collection,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    strategy: StrategyLike | None = None,
    threshold: float = 0.0,
    allow_many: AllowManyLiteral = "neither",
    how: HowLiteral = "inner",
    association_column_names: Tuple[str, str, str] = ("Left", "Right", "Similarity"),
    suffixes: Collection = ("_left", "_right"),
    return_similarity_matrix: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """
    Join dataframes or lists based on semantic similarity.

    Args:
        left: Left DataFrame or Collection of strings
        right: Right DataFrame or Collection of strings
        on: Join column name to use for both left and right dataframes
        left_on: Column name to use for left dataframe
        right_on: Column name to use for right dataframe
        strategy: `jelljoin.Strategy` to use to calculate similarity
        threshold: Minimum similarity score to consider a match (default: 0.0)
        allow_many: Find one-to-many assocations
        association_column_names:
            names for the three columns jellyjoins adds to the dataframe.
        suffixes:
            column suffixes added on the left and right sides to ensure
            uniqueness.
        return_similarity_matrix:
            pass True to return the pair (dataframe, similarity_matrix)
            instead of just the dataframe (default)

    Returns:
        DataFrame with joined data sorted by (Left, Right) indices
    """
    _validate_jellyjoin_arguments(suffixes, association_column_names, allow_many, how)

    # resolve StrategyLike to specific Strategy instance
    strategy = get_similarity_strategy(strategy)

    # standardize to dataframes joined on specific columns
    left, right, left_on, right_on = _coerce_to_dataframes(
        left, right, on, left_on, right_on
    )

    # Calculate similarity matrix
    similarity_matrix = strategy(left[left_on], right[right_on])

    # Find optimal one-to-one assignments using Hungarian algorithm
    logger.debug("Solving assignment problem for %s matrix.", similarity_matrix.shape)
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    scores = similarity_matrix[row_indices, col_indices]
    mask = scores >= threshold
    assignments = list(zip(row_indices[mask], col_indices[mask], scores[mask]))

    # Add on extra one-to-many or many-to-one assignments if desired
    if allow_many != "neither":
        extra_assignments = _all_extra_assignments(
            allow_many, assignments, similarity_matrix, threshold
        )
        assignments.extend(extra_assignments)

    # join left to right with the assignments in the middle
    middle = pd.DataFrame(assignments, columns=association_column_names)
    result = _triple_join(left, middle, right, how, suffixes)

    # Sort and reset index
    result = result.sort_values(by=list(association_column_names))
    result = result.reset_index(drop=True)

    if return_similarity_matrix:
        return result, similarity_matrix
    else:
        return result
