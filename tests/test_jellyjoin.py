import os
import re
from functools import wraps

import dotenv
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import jellyjoin

dotenv.load_dotenv()

# not a fixture; used for parameterize
empties = [
    ([], ["X"]),  # left empty
    (["X"], []),  # right empty
    ([], []),  # both empty
]


def skip_if_openai_not_available(test_func):
    """Skip a test if OpenAI dependencies or credentials are missing."""

    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="no API key")
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("openai", reason="openai package not installed")
        return test_func(*args, **kwargs)

    return wrapper


def skip_if_nomic_not_available(test_func):
    """Skip test if the Nomic package is not installed."""

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("nomic", reason="nomic package not installed")
        return test_func(*args, **kwargs)

    return wrapper


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def left_words():
    return ["Cat", "Dog", "Piano"]


@pytest.fixture
def right_words():
    return ["CAT", "Dgo", "Whiskey"]


@pytest.fixture
def left_sections():
    return [
        "Introduction",
        "Mathematical Methods",
        "Empirical Validation",
        "Anticipating Criticisms",
        "Future Work",
    ]


@pytest.fixture
def right_sections():
    return [
        "Abstract",
        "Experimental Results",
        "Proposed Extensions",
        "Theoretical Modeling",
        "Limitations",
    ]


@pytest.fixture
def left_df():
    df = pd.DataFrame(
        {
            "API Path": [
                "user.email",
                "user.touch_count",
                "user.propensity_score",
                "user.ltv",
                "user.purchase_count",
                "account.status_code",
                "account.age",
                "account.total_purchase_count",
            ]
        }
    )
    df["Prefix"] = df["API Path"].str.split(".", n=1).str[0]
    return df


@pytest.fixture
def right_df():
    return pd.DataFrame(
        {
            "UI Field Name": [
                "Recent Touch Events",
                "Total Touch Events",
                "Account Age (Years)",
                "User Propensity Score",
                "Estimated Lifetime Value ($)",
                "Account Status",
                "Number of Purchases",
                "Freetext Notes",
            ],
            "Type": [
                "number",
                "number",
                "number",
                "number",
                "currency",
                "string",
                "number",
                "string",
            ],
        }
    )


@pytest.fixture
def pairwise_strategy_default():
    return jellyjoin.PairwiseStrategy()


@pytest.fixture
def pairwise_strategy_jw_lower():
    return jellyjoin.PairwiseStrategy(
        "jaro-winkler",
        preprocessor=lambda x: x.lower(),
    )


@pytest.fixture
def pairwise_strategy_levenshtein():
    return jellyjoin.PairwiseStrategy(jellyjoin.levenshtein_similarity)


@pytest.fixture(scope="session")
def openai_client():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Requires OpenAI key in environment")
    openai = pytest.importorskip("openai")
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def openai_strategy(openai_client):
    return jellyjoin.OpenAIEmbeddingStrategy(openai_client)


# -----------------------
# Tests
# -----------------------


def test_version():
    assert re.match(r"^\d+\.\d+\.\d+$", jellyjoin.__version__)
    assert jellyjoin.__version__ > "0.0.0"


@pytest.mark.parametrize("fn", jellyjoin.similarity.FUNCTION_MAP.values())
def test_similarity_functions(fn):
    assert fn("abcdefg", "abcdefg") == pytest.approx(1.0)
    assert fn("abcdefg", "hijklmn") == pytest.approx(0.0)
    assert 0.1 < fn("abcdefg", "acbdfgh") < 0.9


def test_pairwise_strategy_defaults(pairwise_strategy_default, left_words, right_words):
    matrix = pairwise_strategy_default(left_words, right_words)
    expected = np.array(
        [
            [0.33333333, 0.0, 0.0],
            [0.0, 0.66666667, 0.0],
            [0.0, 0.2, 0.14285714],
        ]
    )
    assert np.allclose(matrix, expected)


def test_pairwise_strategy(pairwise_strategy_jw_lower, left_words, right_words):
    matrix = pairwise_strategy_jw_lower(left_words, right_words)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.55555556, 0.0],
            [0.51111111, 0.0, 0.44761905],
        ]
    )
    assert np.allclose(matrix, expected)


def test_pairwise_strategy_with_custom_function(
    pairwise_strategy_levenshtein, left_words, right_words
):
    matrix = pairwise_strategy_levenshtein(left_words, right_words)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_words), len(right_words))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)


def test_pairwise_strategy_square(pairwise_strategy_default, left_sections):
    matrix = pairwise_strategy_default(left_sections, left_sections)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_sections), len(left_sections))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)
    assert np.all(np.isclose(matrix, matrix.T))
    assert np.all(np.isclose(np.diag(matrix), 1.0))


@skip_if_nomic_not_available
def test_nomic_strategy_defaults(left_words, right_words):
    nomic_strategy = jellyjoin.NomicEmbeddingStrategy()
    matrix = nomic_strategy(left_words, right_words)
    assert matrix.shape == (len(left_words), len(right_words))


@skip_if_nomic_not_available
def test_nomic_strategy_config(left_words, right_words):
    nomic_strategy = jellyjoin.NomicEmbeddingStrategy(
        embedding_model="nomic-embed-text-v1.5",
        preprocessor=lambda x: x.lower(),
        task_type="search_query",
        dimensionality=100,
        device="gpu",
        allow_download=True,
        dtype=np.float64,
    )
    matrix = nomic_strategy(left_words, right_words)
    assert matrix.shape == (len(left_words), len(right_words))
    assert matrix.dtype == np.float64


def test_triple_join():
    from jellyjoin.join import _triple_join

    left = pd.DataFrame(
        {"x": [1, 2, 3], "name": ["aa", "bb", "cc"], "Left": [True] * 3}
    )
    middle = pd.DataFrame(
        {"Left": [0, 1, 2], "Right": [2, 0, 1], "Similarity": [0.5, 0.6, 0.7]}
    )
    right = pd.DataFrame(
        {"y": [1, 2, 3], "name": ["AA", "BB", "CC"], "Right": [False] * 3}
    )

    result = _triple_join(
        left, middle, right, how="inner", suffixes=("_left", "_right")
    )

    expected_columns = [
        "Left",
        "Right",
        "Similarity",
        "x",
        "name_left",
        "Left_left",
        "y",
        "name_right",
        "Right_right",
    ]
    assert list(result.columns) == expected_columns
    assert result["name_left"].tolist() == ["aa", "bb", "cc"]
    assert result["name_right"].tolist() == ["CC", "AA", "BB"]


@pytest.mark.parametrize("left,right", empties)
def test_pairwise_jellyjoin_empty(left, right):
    df, matrix = jellyjoin.jellyjoin(
        left,
        right,
        strategy="jaro",
        return_similarity_matrix=True,
    )
    assert df.columns.tolist() == [
        "Left",
        "Right",
        "Similarity",
        "Left Value",
        "Right Value",
    ]
    assert len(df) == 0
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left), len(right))


@skip_if_openai_not_available
@pytest.mark.parametrize("left,right", empties)
def test_openai_empty(left, right):
    strategy = jellyjoin.get_similarity_strategy("openai")
    matrix = strategy(left, right)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left), len(right))


@skip_if_nomic_not_available
@pytest.mark.parametrize("left,right", empties)
def test_nomic_empty(left, right):
    strategy = jellyjoin.get_similarity_strategy("nomic")
    matrix = strategy(left, right)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left), len(right))


def test_jellyjoin_validation():
    with pytest.raises(ValueError, match=r"Pass exactly two suffixes\."):
        jellyjoin.jellyjoin([], [], suffixes=("only_one",))

    with pytest.raises(ValueError, match=r"Pass exactly two suffixes\."):
        jellyjoin.jellyjoin([], [], suffixes=("_left", "_right", "_extra"))

    with pytest.raises(ValueError, match=r"suffixes cannot be the same\."):
        jellyjoin.jellyjoin([], [], suffixes=("_left_foot", "_left_foot"))

    with pytest.raises(TypeError, match=r"suffixes\[0\] must be a string\."):
        jellyjoin.jellyjoin([], [], suffixes=(None, "_right"))

    with pytest.raises(TypeError, match=r"suffixes\[1\] must be a string\."):
        jellyjoin.jellyjoin([], [], suffixes=("_left", 5))

    with pytest.raises(TypeError, match=r"suffixes\[0\] cannot be an empty string\."):
        jellyjoin.jellyjoin([], [], suffixes=("", "_right"))

    with pytest.raises(TypeError, match=r"suffixes\[1\] cannot be an empty string\."):
        jellyjoin.jellyjoin([], [], suffixes=("_left", ""))

    with pytest.raises(TypeError, match=r"similarity_column must be a string\."):
        jellyjoin.jellyjoin([], [], similarity_column=123)

    with pytest.raises(
        ValueError, match=r"similarity_column cannot be an empty string\."
    ):
        jellyjoin.jellyjoin([], [], similarity_column="")

    with pytest.raises(
        ValueError, match=r'allow_many must be "left", "right", "both", or "neither"\.'
    ):
        jellyjoin.jellyjoin([], [], allow_many="some")

    with pytest.raises(
        ValueError, match=r'how argument must be "inner", "left", "right", or "outer"\.'
    ):
        jellyjoin.jellyjoin([], [], how="joiny")

    with pytest.raises(
        ValueError, match=r'Pass only "on" or "left_on" and "right_on", not both\.'
    ):
        jellyjoin.jellyjoin([], [], on="id", left_on="client_no")

    with pytest.raises(
        ValueError, match=r'Pass only "on" or "left_on" and "right_on", not both\.'
    ):
        jellyjoin.jellyjoin([], [], on="id", right_on="client_no")


def test_jellyjoin_options():
    left = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["aaa", "bbb", "ccc"],
            "left": [True] * 3,
        }
    )
    right = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["aab", "bb", "cac"],
            "right": [False] * 3,
        }
    )

    df = jellyjoin.jellyjoin(
        left,
        right,
        on="name",
        strategy=jellyjoin.PairwiseStrategy("jaro-winkler"),
        threshold=0.01,
        allow_many="left",
        how="outer",
        left_index_column="left_index",
        right_index_column=None,
        similarity_column="score",
        suffixes=("_2024", "_2025"),
    )

    expected = pd.DataFrame(
        {
            "left_index": [0, 1, 2],
            "score": [0.822222, 0.911111, 0.8],
            "id_2024": [1, 2, 3],
            "name_2024": ["aaa", "bbb", "ccc"],
            "left": [True, True, True],
            "id_2025": [1, 2, 3],
            "name_2025": ["aab", "bb", "cac"],
            "right": [False, False, False],
        }
    )

    # Ensure column order is exactly as expected
    assert list(df.columns) == list(expected.columns)

    # Compare values with float tolerance and matching index
    pdt.assert_frame_equal(
        df.reset_index(drop=True),
        expected,
        check_dtype=True,
        atol=1e-6,
        rtol=1e-6,
    )


def test_jellyjoin_drop_columns():
    df = jellyjoin.jellyjoin(
        ["x", "y"],
        ["y", "x"],
        left_index_column=None,
        right_index_column=None,
        similarity_column=None,
    )
    assert df.columns.tolist() == ["Left Value", "Right Value"]


def test_jellyjoin_with_lists(left_sections, right_sections):
    df = jellyjoin.jellyjoin(left_sections, right_sections)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(left_sections), len(right_sections))
    assert df["Similarity"].between(0.0, 1.0).all()


def test_jellyjoin_return_similarity_matrix(left_words, right_words):
    df, matrix = jellyjoin.jellyjoin(
        left_words,
        right_words,
        return_similarity_matrix=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(left_words), len(right_words))
    assert df["Similarity"].between(0.0, 1.0).all()

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_words), len(right_words))


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_jellyjoin_with_dataframes_all_hows(left_df, right_df, how):
    df = jellyjoin.jellyjoin(
        left_df,
        right_df,
        left_on="API Path",
        right_on="UI Field Name",
        threshold=0.4,
        how=how,
    )
    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].dropna().between(0.0, 1.0).all()


@pytest.mark.parametrize("allow_many", ["neither", "left", "right", "both"])
def test_jellyjoin_allow_many(left_df, right_df, allow_many):
    df = jellyjoin.jellyjoin(
        left_df,
        right_df,
        left_on="API Path",
        right_on="UI Field Name",
        threshold=0.4,
        allow_many=allow_many,
    )
    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].between(0.0, 1.0).all()


@skip_if_openai_not_available
def test_openai_strategy(openai_strategy, left_sections, right_sections):
    matrix = openai_strategy(left_sections, right_sections)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_sections), len(right_sections))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)


@skip_if_openai_not_available
def test_openai_strategy_small_batch(openai_client):
    LENGTH = 5
    strategy = jellyjoin.OpenAIEmbeddingStrategy(
        openai_client,
        batch_size=2,
    )
    left = ["test"] * LENGTH
    right = ["testing"]
    matrix = strategy(left, right)
    assert matrix.shape == (LENGTH, 1)


@skip_if_openai_not_available
def test_openai_strategy_truncate(openai_strategy):
    left = [
        "x" * 8191,
        "x" * 9001,
        "x" * 81910,
        " ".join(["eight"] * 8191),
        " ".join(["eight"] * 8192),
        " ".join(["eight"] * 9001),
    ]
    right = ["teen"]
    matrix = openai_strategy(left, right)
    assert matrix.shape == (6, 1)


@skip_if_openai_not_available
def test_openai_strategy_caching():
    strategy1 = jellyjoin.get_automatic_strategy()
    strategy2 = jellyjoin.get_automatic_strategy()
    assert strategy1 is strategy2


def test_get_similarity_function():
    get_similarity_function = jellyjoin.get_similarity_function

    # default
    out = get_similarity_function(None)
    assert out is jellyjoin.damerau_levenshtein_similarity

    # Callable passthrough (identity)
    g = lambda a, b: 1.0  # noqa: E731
    out = get_similarity_function(g)
    assert out is g

    # Exact names map correctly
    assert get_similarity_function("hamming") is jellyjoin.hamming_similarity
    assert get_similarity_function("levenshtein") is jellyjoin.levenshtein_similarity
    assert (
        get_similarity_function("damerau_levenshtein")
        is jellyjoin.damerau_levenshtein_similarity
    )
    assert get_similarity_function("jaro") is jellyjoin.jaro_similarity
    assert get_similarity_function("jaro_winkler") is jellyjoin.jaro_winkler_similarity

    # Normalization: case, whitespace, hyphen→underscore
    assert get_similarity_function("  JARO  ") is jellyjoin.jaro_similarity
    assert get_similarity_function("jaro-winkler") is jellyjoin.jaro_winkler_similarity
    assert get_similarity_function("LeVeNsHtEiN") is jellyjoin.levenshtein_similarity

    # raise for other
    with pytest.raises(KeyError):
        jellyjoin.get_similarity_function("whatever")


def test_get_similarity_strategy():
    # default to automatic strategy
    output = jellyjoin.get_similarity_strategy()
    assert isinstance(output, jellyjoin.SimilarityStrategy)

    output = jellyjoin.get_similarity_strategy(None)
    assert isinstance(output, jellyjoin.SimilarityStrategy)

    # pass through a Strategy subclass
    strategy = jellyjoin.PairwiseStrategy()
    output = jellyjoin.get_similarity_strategy(strategy)
    assert output is strategy

    # pass through a callable
    def custom_function(x, y):
        return 0.0

    output = jellyjoin.get_similarity_strategy(custom_function)
    assert output is custom_function

    # delegate to pairwise
    for strategy in [
        "jaro_winkler",
        "Jaro-Winkler",
        " JaRo-WiNkLeR ",
        "jaro_winkler_similarity",
    ]:
        output = jellyjoin.get_similarity_strategy(strategy)
        assert isinstance(output, jellyjoin.PairwiseStrategy)
        assert output.similarity_function is jellyjoin.jaro_winkler_similarity

    with pytest.raises(ValueError, match=r"^Strategy name 'whatever' must"):
        jellyjoin.get_similarity_strategy("whatever")

    # raise for anything else
    with pytest.raises(TypeError):
        jellyjoin.get_similarity_strategy(123)


@skip_if_openai_not_available
def test_get_similarity_strategy_openai(left_words, right_words):
    for strategy in ("openai", "OpenAI", " openai "):
        output = jellyjoin.get_similarity_strategy(strategy)
        assert isinstance(output, jellyjoin.OpenAIEmbeddingStrategy)

    df = jellyjoin.jellyjoin(left_words, right_words, strategy="openai")
    assert isinstance(df, pd.DataFrame)


@skip_if_nomic_not_available
def test_get_similarity_strategy_nomic(left_words, right_words):
    for strategy in ["nomic", "NoMiC", " nomic "]:
        output = jellyjoin.get_similarity_strategy(strategy)
        assert isinstance(output, jellyjoin.NomicEmbeddingStrategy)

    df = jellyjoin.jellyjoin(left_words, right_words, strategy="nomic")
    assert isinstance(df, pd.DataFrame)
