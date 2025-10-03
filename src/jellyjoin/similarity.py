from typing import Callable, Union
import jellyfish

__all__ = [
    "hamming_similarity",
    "levenshtein_similarity",
    "damerau_levenshtein_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
]


def hamming_similarity(x: str, y: str) -> float:
    """
    Compute the normalized Hamming similarity between two strings.

    The Hamming similarity is defined as:

    .. math::
        1 - \\frac{d_{H}(x, y)}{\\max(|x|, |y|)}

    where :math:`d_{H}` is the Hamming distance. This measure 
    assumes both strings are of equal length; if not, the shorter 
    one is conceptually padded to match.

    Parameters
    ----------
    x : str
        First input string.
    y : str
        Second input string.

    Returns
    -------
    float
        Similarity score between 0.0 and 1.0.
    """
    length = max(len(x), len(y))
    return 1.0 - jellyfish.hamming_distance(x, y) / length


def levenshtein_similarity(x: str, y: str) -> float:
    """
    Compute the normalized Levenshtein similarity between two strings.

    The Levenshtein similarity is defined as:

    .. math::
        1 - \\frac{d_{L}(x, y)}{\\max(|x|, |y|)}

    where :math:`d_{L}` is the Levenshtein distance (edit distance).

    Parameters
    ----------
    x : str
        First input string.
    y : str
        Second input string.

    Returns
    -------
    float
        Similarity score between 0.0 and 1.0.
    """
    length = max(len(x), len(y))
    return 1.0 - jellyfish.levenshtein_distance(x, y) / length


def damerau_levenshtein_similarity(x: str, y: str) -> float:
    """
    Compute the normalized Damerau-Levenshtein similarity between two strings.

    The Damerau-Levenshtein similarity is defined as:

    .. math::
        1 - \\frac{d_{DL}(x, y)}{\\max(|x|, |y|)}

    where :math:`d_{DL}` is the Damerau-Levenshtein distance 
    (edit distance with transpositions).

    Parameters
    ----------
    x : str
        First input string.
    y : str
        Second input string.

    Returns
    -------
    float
        Similarity score between 0.0 and 1.0.
    """
    length = max(len(x), len(y))
    return 1.0 - jellyfish.damerau_levenshtein_distance(x, y) / length


def jaro_similarity(x: str, y: str) -> float:
    """
    Compute the Jaro similarity between two strings.

    The Jaro similarity accounts for character transpositions and 
    common characters within a matching window. It is often used in 
    record linkage and duplicate detection.

    Parameters
    ----------
    x : str
        First input string.
    y : str
        Second input string.

    Returns
    -------
    float
        Similarity score between 0.0 and 1.0.
    """
    return jellyfish.jaro_similarity(x, y)


def jaro_winkler_similarity(x: str, y: str) -> float:
    """
    Compute the Jaro-Winkler similarity between two strings.

    The Jaro-Winkler similarity extends the Jaro similarity by 
    giving more weight to common prefixes, making it well-suited 
    for short strings such as names.

    Parameters
    ----------
    x : str
        First input string.
    y : str
        Second input string.

    Returns
    -------
    float
        Similarity score between 0.0 and 1.0.
    """
    return jellyfish.jaro_winkler_similarity(x, y)


# this global needs to be defined *after* the functions.
FUNCTION_MAP = {
    "hamming": hamming_similarity,
    "levenshtein": levenshtein_similarity,
    "damerau_levenshtein": damerau_levenshtein_similarity,
    "jaro": jaro_similarity,
    "jaro_winkler": jaro_winkler_similarity,
}
def get_similarity_function(
    function: Union[str, Callable[[str, str], float]]
) -> Callable[[str, str], float]:
    """
    Resolve a string identifier to a string similarity function.

    The input name is normalized by converting to lowercase, 
    stripping leading and trailing whitespace, and replacing 
    hyphens with underscores.

    Parameters
    ----------
    function : str or callable
        The name of the similarity function, or the similarity function itself.

    Returns
    -------
    callable
        The similarity function corresponding to the given name.

    Raises
    ------
    KeyError
        If the name does not correspond to a known similarity function.
    """
    if callable(function):
        return function

    key = function.strip().lower().replace("-", "_")
    return FUNCTION_MAP[key]
