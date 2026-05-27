import logging
from collections.abc import Collection, Iterable, Sequence
from typing import Literal, TypedDict, get_args, overload

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
    "Jelly",
    "DROP",
]

logger = logging.getLogger(__name__)

# internal type only used by private functions
AssignmentList = list[tuple[int, int, float]]
SuffixPair = tuple[str, str]


class JellyDefaults(TypedDict):
    on: str | None
    left_on: str | None
    right_on: str | None
    strategy: StrategyLike | None
    threshold: float
    allow_many: AllowManyLiteral
    how: HowLiteral
    left_index_column: str
    right_index_column: str
    similarity_column: str
    suffixes: SuffixPair
    return_similarity_matrix: bool


# sentinal value indicating a column should be dropped
DROP = ""

# these column names are used for in the middle association dataframe are long
# to reduce the probability of conflicting with user column names.
LEFT_COLUMN = "Left Jellyjoin Index"
RIGHT_COLUMN = "Right Jellyjoin Index"
SIMILARITY_COLUMN = "Jellyjoin Similarity Score"


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
    suffixes: SuffixPair,
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
    left: pd.DataFrame | Collection,
    right: pd.DataFrame | Collection,
    on: str | None,
    left_on: str | None,
    right_on: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    # flag the special case where they're identical
    identical = left is right

    # Convert inputs to dataframes if they aren't already
    if not isinstance(left, pd.DataFrame):
        left = pd.DataFrame({left_on or "Left Value": list(left)})

    if identical:
        right = left
    elif not isinstance(right, pd.DataFrame):
        right = pd.DataFrame({right_on or "Right Value": list(right)})

    # handle the shared "on" column name
    if on:
        if left_on or right_on:
            raise ValueError(
                "If the `on` argument is passed, `left_on` and `right_on` must not be passed."
            )
        left_on = on
        right_on = on

    # default to joining on the first column if not explicitly named
    if not left_on:
        left_on = left.columns[0]

    if not right_on:
        right_on = right.columns[0]

    if not isinstance(left_on, str) or not isinstance(right_on, str):
        raise TypeError(
            "Arguments `on`, `left_on`, and `right_on` must be strings if supplied."
        )

    return left, right, left_on, right_on


def _validate_jellyjoin_arguments(
    suffixes: Sequence[object],
    left_index_column: str | None,
    right_index_column: str | None,
    similarity_column: str | None,
    allow_many: AllowManyLiteral,
    how: HowLiteral,
):
    """
    Raises exception for any invalid arguments.
    """
    # suffixes added for disambiguation
    if len(suffixes) != 2:
        raise ValueError("Pass exactly two suffixes.")

    for index, suffix in enumerate(suffixes):
        if not isinstance(suffix, str):
            raise TypeError(f"suffixes[{index}] must be a string.")
        if not suffix:
            raise TypeError(f"suffixes[{index}] cannot be an empty string.")
    if suffixes[0] == suffixes[1]:
        raise ValueError("suffixes cannot be the same.")

    # column names
    if similarity_column is not None:
        if not isinstance(similarity_column, str):
            raise TypeError("similarity_column must be a string.")

    if left_index_column is not None:
        if not isinstance(left_index_column, str):
            raise TypeError("left_index_column must be a string.")

    if right_index_column is not None:
        if not isinstance(right_index_column, str):
            raise TypeError("right_index_column must be a string.")

    # allow many values
    if allow_many not in get_args(AllowManyLiteral):
        raise ValueError('allow_many must be "left", "right", "both", or "neither".')

    # allow many values
    if how not in get_args(HowLiteral):
        raise ValueError('how argument must be "inner", "left", "right", or "outer".')


def _prepare_middle_columns(
    left_index_column: str,
    right_index_column: str,
    similarity_column: str,
) -> tuple[list[str], list[str]]:
    """
    Figures out the permanent or temporary names to use for the middle column,
    and which temporary columns to drop afterwarsd
    """
    middle_columns = [
        left_index_column or LEFT_COLUMN,
        right_index_column or RIGHT_COLUMN,
        similarity_column or SIMILARITY_COLUMN,
    ]

    # if the user supplied column names are None, the column is temporary
    # and will be dropped from the result dataframe.
    drop_columns = [
        LEFT_COLUMN if not left_index_column else None,
        RIGHT_COLUMN if not right_index_column else None,
        SIMILARITY_COLUMN if not similarity_column else None,
    ]
    drop_columns = [column for column in drop_columns if column]

    return middle_columns, drop_columns


@overload
def jellyjoin(
    left: pd.DataFrame | Collection,
    right: pd.DataFrame | Collection,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    strategy: StrategyLike | None = None,
    threshold: float = 0.0,
    allow_many: AllowManyLiteral = "neither",
    how: HowLiteral = "inner",
    left_index_column: str = "Left",
    right_index_column: str = "Right",
    similarity_column: str = "Similarity",
    suffixes: SuffixPair = ("_left", "_right"),
    return_similarity_matrix: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def jellyjoin(
    left: pd.DataFrame | Collection,
    right: pd.DataFrame | Collection,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    strategy: StrategyLike | None = None,
    threshold: float = 0.0,
    allow_many: AllowManyLiteral = "neither",
    how: HowLiteral = "inner",
    left_index_column: str = "Left",
    right_index_column: str = "Right",
    similarity_column: str = "Similarity",
    suffixes: SuffixPair = ("_left", "_right"),
    return_similarity_matrix: Literal[True],
) -> tuple[pd.DataFrame, np.ndarray]: ...


def jellyjoin(
    left: pd.DataFrame | Collection,
    right: pd.DataFrame | Collection,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    strategy: StrategyLike | None = None,
    threshold: float = 0.0,
    allow_many: AllowManyLiteral = "neither",
    how: HowLiteral = "inner",
    left_index_column: str = "Left",
    right_index_column: str = "Right",
    similarity_column: str = "Similarity",
    suffixes: SuffixPair = ("_left", "_right"),
    return_similarity_matrix: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Join two data sources by computing pairwise semantic similarity.

    This function extends fuzzy or semantic matching to perform a join
    between two datasets (lists or DataFrames). Similarity between records
    is computed using the specified strategy, and the results are combined
    into a joined DataFrame, optionally along with the full similarity matrix.

    Parameters
    ----------
    left : pandas.DataFrame or Collection of str
        Left input data. Can be a DataFrame or a list/series of strings.
    right : pandas.DataFrame or Collection of str
        Right input data. Can be a DataFrame or a list/series of strings.
    on : str, optional
        Column name to join on when both left and right are DataFrames.
        Mutually exclusive with `left_on` and `right_on`.
    left_on : str, optional
        Column name in the left DataFrame to use for matching.
    right_on : str, optional
        Column name in the right DataFrame to use for matching.

    strategy : StrategyLike, optional
        Similarity strategy to use (e.g., `"jaro_winkler"`, `"openai"`,
        or a `jellyjoin.Strategy` instance). If `None`, an automatic
        strategy is selected based on availability.
    threshold : float, default=0.0
        Minimum similarity score (0.0–1.0) required to consider a pair
        as a valid match.
    allow_many : {"neither", "left", "right", "both"}, default="neither"
        Controls whether multiple matches per row are allowed on each side.
    how : {"inner", "left", "right", "outer"}, default="inner"
        Join type determining which rows to include in the final output.

    left_index_column : str , default="Left"
        Name for the output column holding the `.iloc` index of each left record.
        Pass `jellyjoin.DROP` to omit this column.
    right_index_column : str, default="Right"
        Name for the output column holding the `.iloc` index of each right record.
        Pass `jellyjoin.DROP` to omit this column.
    similarity_column : str, default="Similarity"
        Name for the output column containing similarity scores.
        Pass `jellyjoin.DROP` to omit this column.
    suffixes : Collection of str, default=("_left", "_right")
        Suffixes to append to overlapping column names from the left and
        right DataFrames to ensure uniqueness.
    return_similarity_matrix : bool, default=False
        If True, return a tuple `(DataFrame, ndarray)` where the second
        element is the full similarity matrix. Otherwise, return only
        the joined DataFrame.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, numpy.ndarray)
        Joined DataFrame, optionally with the similarity matrix when
        `return_similarity_matrix=True`.

    Raises
    ------
    ValueError
        If invalid or conflicting join arguments are provided.
    TypeError
        If argument types are incompatible with expected input formats.

    Examples
    --------
    >>> import jellyjoin
    >>> left = ["cat", "dog", "piano"]
    >>> right = ["CAT", "Dgo", "Whiskey"]
    >>> df = jellyjoin.jellyjoin(left, right, strategy="jaro_winkler")
    >>> df.head()
         Left  Right  Similarity Left Value Right Value
    0      0      0    1.000000         cat         CAT
    1      1      1    0.555556         dog         Dgo
    2      2      2    0.000000       piano     Whiskey
    """
    _validate_jellyjoin_arguments(
        suffixes,
        left_index_column,
        right_index_column,
        similarity_column,
        allow_many,
        how,
    )

    # resolve StrategyLike to specific Strategy instance
    strategy = get_similarity_strategy(strategy)

    # standardize to dataframes joined on specific columns
    left, right, left_on, right_on = _coerce_to_dataframes(
        left, right, on, left_on, right_on
    )

    left_texts = left[left_on]
    if left is right and left_on == right_on:
        right_texts = left_texts
    else:
        right_texts = right[right_on]

    # Calculate similarity matrix
    similarity_matrix = strategy(left_texts, right_texts)

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

    middle_columns, drop_columns = _prepare_middle_columns(
        left_index_column, right_index_column, similarity_column
    )

    # join left to right with the assignments in the middle
    middle = pd.DataFrame(assignments, columns=middle_columns)
    result = _triple_join(left, middle, right, how, suffixes)

    # Sort and reset index
    result = result.sort_values(by=list(middle_columns)).reset_index(drop=True)

    # drop any temporary columns the user didn't ask for.
    if drop_columns:
        result = result.drop(columns=drop_columns)

    if return_similarity_matrix:
        return result, similarity_matrix
    else:
        return result


class Jelly:
    def __init__(
        self,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        strategy: StrategyLike | None = None,
        threshold: float = 0.0,
        allow_many: AllowManyLiteral = "neither",
        how: HowLiteral = "inner",
        left_index_column: str = "Left",
        right_index_column: str = "Right",
        similarity_column: str = "Similarity",
        suffixes: SuffixPair = ("_left", "_right"),
        return_similarity_matrix: bool = False,
    ) -> None:
        """
        Create a curried Jelly joiner with default options.

        Parameters
        ----------
        on : str, optional
            Column name to join on when both inputs are DataFrames.
        left_on : str, optional
            Column name in the left DataFrame to use for matching.
        right_on : str, optional
            Column name in the right DataFrame to use for matching.

        strategy : StrategyLike, optional
            Similarity strategy to use (e.g., "jaro_winkler", "openai", or a Strategy instance).
        threshold : float, default=0.0
            Minimum similarity score (0.0–1.0) required to consider a pair a valid match.
        allow_many : {"neither", "left", "right", "both"}, default="neither"
            Controls whether multiple matches per row are allowed on each side.
        how : {"inner", "left", "right", "outer"}, default="inner"
            Join type determining which rows to include in the final output.

        left_index_column : str, default="Left"
            Name for the column holding the `.iloc` index of each left record.
            Pass `jellyjoin.DROP` to omit this column in the output dataframe.
        right_index_column : str, default="Right"
            Name for the column holding the `.iloc` index of each right record.
            Pass `jellyjoin.DROP` to omit this column in the output dataframe.
        similarity_column : str, default="Similarity"
            Name for the column containing similarity scores.
            Pass `jellyjoin.DROP` to omit this column in the output dataframe.
        suffixes : Collection of str, default=("_left", "_right")
            Suffixes appended to overlapping column names from left/right DataFrames.
        return_similarity_matrix : bool, default=False
            If True, `join()` returns `(DataFrame, ndarray)`; otherwise just the DataFrame.
        """
        _validate_jellyjoin_arguments(
            suffixes,
            left_index_column,
            right_index_column,
            similarity_column,
            allow_many,
            how,
        )
        return_similarity_matrix = bool(return_similarity_matrix)

        self.defaults: JellyDefaults = {
            "on": on,
            "left_on": left_on,
            "right_on": right_on,
            "strategy": strategy,
            "threshold": threshold,
            "allow_many": allow_many,
            "how": how,
            "left_index_column": left_index_column,
            "right_index_column": right_index_column,
            "similarity_column": similarity_column,
            "suffixes": suffixes,
            "return_similarity_matrix": return_similarity_matrix,
        }

    @overload
    def join(
        self,
        left: pd.DataFrame | Collection,
        right: pd.DataFrame | Collection,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        strategy: StrategyLike | None = None,
        threshold: float | None = None,
        allow_many: AllowManyLiteral | None = None,
        how: HowLiteral | None = None,
        left_index_column: str | None = None,
        right_index_column: str | None = None,
        similarity_column: str | None = None,
        suffixes: SuffixPair | None = None,
        return_similarity_matrix: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @overload
    def join(
        self,
        left: pd.DataFrame | Collection,
        right: pd.DataFrame | Collection,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        strategy: StrategyLike | None = None,
        threshold: float | None = None,
        allow_many: AllowManyLiteral | None = None,
        how: HowLiteral | None = None,
        left_index_column: str | None = None,
        right_index_column: str | None = None,
        similarity_column: str | None = None,
        suffixes: SuffixPair | None = None,
        return_similarity_matrix: Literal[True],
    ) -> tuple[pd.DataFrame, np.ndarray]: ...

    def join(
        self,
        left: pd.DataFrame | Collection,
        right: pd.DataFrame | Collection,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        strategy: StrategyLike | None = None,
        threshold: float | None = None,
        allow_many: AllowManyLiteral | None = None,
        how: HowLiteral | None = None,
        left_index_column: str | None = None,
        right_index_column: str | None = None,
        similarity_column: str | None = None,
        suffixes: SuffixPair | None = None,
        return_similarity_matrix: bool | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """
        Join two data sources by computing pairwise semantic similarity,
        using the defaults set on the `Jelly` instance unless overridden.

        Parameters
        ----------
        left : pandas.DataFrame or Collection of str
            Left input data. Can be a DataFrame or a list/series of strings.
        right : pandas.DataFrame or Collection of str
            Right input data. Can be a DataFrame or a list/series of strings.
        on : str, optional
            Column name to join on when both left and right are DataFrames.
            Mutually exclusive with `left_on` and `right_on`.
        left_on : str, optional
            Column name in the left DataFrame to use for matching.
        right_on : str, optional
            Column name in the right DataFrame to use for matching.

        strategy : StrategyLike, optional
            Similarity strategy to use (e.g., "jaro_winkler", "openai",
            or a Strategy instance).
        threshold : float, optional
            Minimum similarity score (0.0–1.0) required to consider a pair as a valid match.
        allow_many : {"neither", "left", "right", "both"}, optional
            Controls whether multiple matches per row are allowed on each side.
        how : {"inner", "left", "right", "outer"}, optional
            Join type determining which rows to include in the final output.

        left_index_column : str, optional
            Name for the output column holding the `.iloc` index of each left record.
            Normally defaults to "Left".
            Pass `jellyjoin.DROP` to omit this column.
        right_index_column : str, optional
            Name for the output column holding the `.iloc` index of each right record.
            Normally defaults to "Right".
            Pass `jellyjoin.DROP` to omit this column.
        similarity_column : str, optional
            Name for the output column containing similarity scores.
            Normally defaults to "Similarity".
            Pass `jellyjoin.DROP` to omit this column.
        suffixes : Collection of str, optional
            Suffixes to append to overlapping column names from the left and right DataFrames.
        return_similarity_matrix : bool, optional
            If True, return `(DataFrame, ndarray)`; otherwise, only the joined DataFrame.

        Returns
        -------
        pandas.DataFrame or (pandas.DataFrame, numpy.ndarray)
            Joined DataFrame, optionally with the similarity matrix.

        Raises
        ------
        ValueError
            If invalid or conflicting join arguments are provided.
        TypeError
            If argument types are incompatible with expected input formats.
        """
        resolved_on = self.defaults["on"] if on is None else on
        resolved_left_on = self.defaults["left_on"] if left_on is None else left_on
        resolved_right_on = self.defaults["right_on"] if right_on is None else right_on

        if on is not None:
            resolved_left_on = None
            resolved_right_on = None
        elif left_on is not None or right_on is not None:
            resolved_on = None

        resolved_strategy = self.defaults["strategy"] if strategy is None else strategy
        resolved_threshold = (
            self.defaults["threshold"] if threshold is None else threshold
        )
        resolved_allow_many = (
            self.defaults["allow_many"] if allow_many is None else allow_many
        )
        resolved_how = self.defaults["how"] if how is None else how
        resolved_left_index_column = (
            self.defaults["left_index_column"]
            if left_index_column is None
            else left_index_column
        )
        resolved_right_index_column = (
            self.defaults["right_index_column"]
            if right_index_column is None
            else right_index_column
        )
        resolved_similarity_column = (
            self.defaults["similarity_column"]
            if similarity_column is None
            else similarity_column
        )
        resolved_suffixes = self.defaults["suffixes"] if suffixes is None else suffixes
        resolved_return_similarity_matrix = (
            self.defaults["return_similarity_matrix"]
            if return_similarity_matrix is None
            else return_similarity_matrix
        )

        if resolved_return_similarity_matrix:
            return jellyjoin(
                left,
                right,
                on=resolved_on,
                left_on=resolved_left_on,
                right_on=resolved_right_on,
                strategy=resolved_strategy,
                threshold=resolved_threshold,
                allow_many=resolved_allow_many,
                how=resolved_how,
                left_index_column=resolved_left_index_column,
                right_index_column=resolved_right_index_column,
                similarity_column=resolved_similarity_column,
                suffixes=resolved_suffixes,
                return_similarity_matrix=True,
            )

        return jellyjoin(
            left,
            right,
            on=resolved_on,
            left_on=resolved_left_on,
            right_on=resolved_right_on,
            strategy=resolved_strategy,
            threshold=resolved_threshold,
            allow_many=resolved_allow_many,
            how=resolved_how,
            left_index_column=resolved_left_index_column,
            right_index_column=resolved_right_index_column,
            similarity_column=resolved_similarity_column,
            suffixes=resolved_suffixes,
            return_similarity_matrix=False,
        )
