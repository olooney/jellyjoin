from typing import Union, List, Optional, Any, Tuple, Literal, Callable, TypeAlias
from collections.abc import Collection
import numpy as np

# type descriptions
SimilarityStrategyCallable: TypeAlias = Callable[
    [Collection[str], Collection[str]], np.ndarray
]
SimilarityCallable: TypeAlias = Callable[[str, str], float]
PreprocessorCallable: TypeAlias = Callable[[str], str]
SimilarityIdentifier: TypeAlias = Union[None, str, SimilarityCallable]
