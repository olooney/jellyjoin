from collections.abc import Collection
from typing import Callable, TypeAlias, Union

import numpy as np

# type descriptions
StrategyCallable: TypeAlias = Callable[[Collection[str], Collection[str]], np.ndarray]
SimilarityCallable: TypeAlias = Callable[[str, str], float]
PreprocessorCallable: TypeAlias = Callable[[str], str]
SimilarityIdentifier: TypeAlias = Union[None, str, SimilarityCallable]
