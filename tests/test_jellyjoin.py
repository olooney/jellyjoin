import os
import re
from functools import wraps

import dotenv
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import jellyjoin as jj

dotenv.load_dotenv()


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


def skip_if_ollama_not_available(test_func):
    """Skip test if the Ollama package is not installed."""

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("ollama", reason="ollama package not installed")
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
def pairwise_strategy():
    return jj.PairwiseStrategy()


@pytest.fixture(scope="session")
def openai_client():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Requires OpenAI key in environment")
    openai = pytest.importorskip("openai")
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def openai_strategy(openai_client):
    return jj.OpenAIEmbeddingStrategy(openai_client)


@pytest.fixture(
    params=[([], ["X"]), (["X"], []), ([], [])],
    ids=["left-empty", "right-empty", "both-empty"],
)
def empties(request):
    return request.param


# -----------------------
# Tests
# -----------------------


def test_version():
    assert re.match(r"^\d+\.\d+\.\d+$", jj.__version__)
    assert jj.__version__ > "0.0.0"


@pytest.mark.parametrize("fn", jj.similarity.FUNCTION_MAP.values())
def test_similarity_functions(fn):
    assert fn("abcdefg", "abcdefg") == pytest.approx(1.0)
    assert fn("abcdefg", "hijklmn") == pytest.approx(0.0)
    assert 0.1 < fn("abcdefg", "acbdfgh") < 0.9


def test_pairwise_strategy_default(pairwise_strategy, left_words, right_words):
    matrix = pairwise_strategy(left_words, right_words)
    expected = np.array(
        [
            [0.33333333, 0.0, 0.0],
            [0.0, 0.66666667, 0.0],
            [0.0, 0.2, 0.14285714],
        ]
    )
    assert np.allclose(matrix, expected)


def test_pairwise_strategy(left_words, right_words):
    strategy = jj.PairwiseStrategy(
        "jaro-winkler",
        preprocessor=lambda x: x.lower(),
    )
    matrix = strategy(left_words, right_words)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.55555556, 0.0],
            [0.51111111, 0.0, 0.44761905],
        ]
    )
    assert np.allclose(matrix, expected)


def test_pairwise_strategy_with_custom_function(left_words, right_words):
    strategy = jj.PairwiseStrategy(jj.levenshtein_similarity)
    matrix = strategy(left_words, right_words)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_words), len(right_words))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)


def test_pairwise_strategy_square(pairwise_strategy, left_sections):
    matrix = pairwise_strategy(left_sections, left_sections)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_sections), len(left_sections))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)
    assert np.all(np.isclose(matrix, matrix.T))
    assert np.all(np.isclose(np.diag(matrix), 1.0))


@skip_if_nomic_not_available
def test_nomic_strategy_defaults(left_words, right_words):
    nomic_strategy = jj.NomicEmbeddingStrategy()
    matrix = nomic_strategy(left_words, right_words)
    assert matrix.shape == (len(left_words), len(right_words))


@skip_if_nomic_not_available
def test_nomic_strategy_config(left_words, right_words):
    nomic_strategy = jj.NomicEmbeddingStrategy(
        embedding_model="nomic-embed-text-v1.5",
        preprocessor=lambda x: x.lower(),
        task_type="search_query",
        dimensionality=100,
        device="gpu",
        dtype=np.float64,
    )
    matrix = nomic_strategy(left_words, right_words)
    assert matrix.shape == (len(left_words), len(right_words))
    assert matrix.dtype == np.float64


def test_triple_join():
    from jellyjoin._join import _triple_join

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


def test_pairwise_jellyjoin_empty(empties):
    left, right = empties
    df, matrix = jj.jellyjoin(
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
def test_openai_empty(empties):
    left, right = empties
    strategy = jj.get_similarity_strategy("openai")
    matrix = strategy(left, right)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left), len(right))

    # test zero length vectors with .embed() directly
    embedding_vectors = strategy.embed([])
    assert embedding_vectors.ndim == 2
    assert len(embedding_vectors) == 0


@skip_if_nomic_not_available
def test_nomic_empty(empties):
    left, right = empties
    strategy = jj.get_similarity_strategy("nomic")
    matrix = strategy(left, right)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left), len(right))

    # test zero length vectors with .embed() directly
    embedding_vectors = strategy.embed([])
    assert embedding_vectors.ndim == 2
    assert len(embedding_vectors) == 0


@skip_if_ollama_not_available
def test_ollama_empty(empties):
    left, right = empties
    strategy = jj.get_similarity_strategy("ollama")
    matrix = strategy(left, right)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left), len(right))

    # test zero length vectors with .embed() directly
    embedding_vectors = strategy.embed([])
    assert embedding_vectors.ndim == 2
    assert len(embedding_vectors) == 0


@skip_if_openai_not_available
def test_openai_validation():
    with pytest.raises(
        TypeError,
        match=r"embedding_model must be the name of an OpenAI embedding model as a string\.",
    ):
        jj.OpenAIEmbeddingStrategy(embedding_model=123)

    # TODO additional validation


@skip_if_nomic_not_available
def test_nomic_validation():
    with pytest.raises(
        TypeError,
        match=r"embedding_model must be the name of a Nomic embedding model as a string\.",
    ):
        jj.NomicEmbeddingStrategy(embedding_model=123)


@skip_if_ollama_not_available
def test_ollama_validation():
    with pytest.raises(
        TypeError,
        match=r"client, if not None, should be of type 'ollama._client\.Client'",
    ):
        jj.OllamaEmbeddingStrategy(client=42)

    import ollama

    client = ollama.Client()
    with pytest.raises(
        ValueError,
        match=r"Do not pass both client and host arguments; host is only used to instantiate a new client internally\.",
    ):
        jj.OllamaEmbeddingStrategy(
            client=client,
            host="http://localhost:11434",
        )

    with pytest.raises(
        TypeError,
        match=r"embedding_model must be the name of an Ollama embedding model as a string\.",
    ):
        jj.OllamaEmbeddingStrategy(embedding_model=123)


def test_jellyjoin_validation():
    with pytest.raises(ValueError, match=r"Pass exactly two suffixes\."):
        jj.jellyjoin([], [], suffixes=("only_one",))

    with pytest.raises(ValueError, match=r"Pass exactly two suffixes\."):
        jj.jellyjoin([], [], suffixes=("_left", "_right", "_extra"))

    with pytest.raises(ValueError, match=r"suffixes cannot be the same\."):
        jj.jellyjoin([], [], suffixes=("_left_foot", "_left_foot"))

    with pytest.raises(TypeError, match=r"suffixes\[0\] must be a string\."):
        jj.jellyjoin([], [], suffixes=(None, "_right"))

    with pytest.raises(TypeError, match=r"suffixes\[1\] must be a string\."):
        jj.jellyjoin([], [], suffixes=("_left", 5))

    with pytest.raises(TypeError, match=r"suffixes\[0\] cannot be an empty string\."):
        jj.jellyjoin([], [], suffixes=("", "_right"))

    with pytest.raises(TypeError, match=r"suffixes\[1\] cannot be an empty string\."):
        jj.jellyjoin([], [], suffixes=("_left", ""))

    with pytest.raises(TypeError, match=r"similarity_column must be a string\."):
        jj.jellyjoin([], [], similarity_column=123)

    with pytest.raises(TypeError, match=r"left_index_column must be a string\."):
        jj.jellyjoin([], [], left_index_column=123)

    with pytest.raises(TypeError, match=r"right_index_column must be a string\."):
        jj.jellyjoin([], [], right_index_column=123)

    with pytest.raises(
        ValueError, match=r'allow_many must be "left", "right", "both", or "neither"\.'
    ):
        jj.jellyjoin([], [], allow_many="some")

    with pytest.raises(
        ValueError, match=r'how argument must be "inner", "left", "right", or "outer"\.'
    ):
        jj.jellyjoin([], [], how="joiny")

    with pytest.raises(
        ValueError,
        match=r"If the `on` argument is passed, `left_on` and `right_on` must not be passed\.",
    ):
        jj.jellyjoin([], [], on="id", left_on="client_no")

    with pytest.raises(
        ValueError,
        match=r"If the `on` argument is passed, `left_on` and `right_on` must not be passed\.",
    ):
        jj.jellyjoin([], [], on="id", right_on="client_no")

    with pytest.raises(
        TypeError,
        match=r"Arguments `on`, `left_on`, and `right_on` must be strings if supplied\.",
    ):
        jj.jellyjoin([], [], on=42)

    with pytest.raises(
        TypeError,
        match=r"Arguments `on`, `left_on`, and `right_on` must be strings if supplied\.",
    ):
        jj.jellyjoin([], [], left_on=42)

    with pytest.raises(
        TypeError,
        match=r"Arguments `on`, `left_on`, and `right_on` must be strings if supplied\.",
    ):
        jj.jellyjoin([], [], right_on=42)


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

    df = jj.jellyjoin(
        left,
        right,
        on="name",
        strategy=jj.PairwiseStrategy("jaro-winkler"),
        threshold=0.01,
        allow_many="left",
        how="outer",
        left_index_column="left_index",
        right_index_column=jj.DROP,
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
    df = jj.jellyjoin(
        ["x", "y"],
        ["y", "x"],
        left_index_column=jj.DROP,
        right_index_column=jj.DROP,
        similarity_column=jj.DROP,
    )
    assert df.columns.tolist() == ["Left Value", "Right Value"]

    df = jj.jellyjoin(
        ["x", "y"],
        ["y", "x"],
        left_index_column="",
        right_index_column="",
        similarity_column="",
    )
    assert df.columns.tolist() == ["Left Value", "Right Value"]


def test_jellyjoin_with_lists(left_sections, right_sections):
    df = jj.jellyjoin(left_sections, right_sections)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(left_sections), len(right_sections))
    assert df["Similarity"].between(0.0, 1.0).all()


def test_jellyjoin_return_similarity_matrix(left_words, right_words):
    def validate_df(df):
        assert isinstance(df, pd.DataFrame)
        assert len(df) == min(len(left_words), len(right_words))
        assert df["Similarity"].between(0.0, 1.0).all()

    def validate_matrix(matrix):
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (len(left_words), len(right_words))

    df, matrix = jj.jellyjoin(
        left_words,
        right_words,
        return_similarity_matrix=True,
    )
    validate_df(df)
    validate_matrix(matrix)

    jelly = jj.Jelly(return_similarity_matrix=True)
    df, matrix = jelly.join(left_words, right_words)
    validate_df(df)
    validate_matrix(matrix)

    jelly = jj.Jelly()
    df, matrix = jelly.join(left_words, right_words, return_similarity_matrix=True)
    validate_df(df)
    validate_matrix(matrix)

    jelly = jj.Jelly(return_similarity_matrix=False)
    df, matrix = jelly.join(left_words, right_words, return_similarity_matrix=True)
    validate_df(df)
    validate_matrix(matrix)

    jelly = jj.Jelly(return_similarity_matrix=True)
    df = jelly.join(left_words, right_words, return_similarity_matrix=False)
    validate_df(df)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_jellyjoin_with_dataframes_all_hows(left_df, right_df, how):
    df = jj.jellyjoin(
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
    df = jj.jellyjoin(
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
    strategy = jj.OpenAIEmbeddingStrategy(
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
    strategy1 = jj.get_automatic_strategy()
    strategy2 = jj.get_automatic_strategy()
    assert strategy1 is strategy2


def test_get_similarity_function():
    get_similarity_function = jj.get_similarity_function

    # default
    out = get_similarity_function(None)
    assert out is jj.damerau_levenshtein_similarity

    # Callable passthrough (identity)
    g = lambda a, b: 1.0  # noqa: E731
    out = get_similarity_function(g)
    assert out is g

    # Exact names map correctly
    assert get_similarity_function("hamming") is jj.hamming_similarity
    assert get_similarity_function("levenshtein") is jj.levenshtein_similarity
    assert (
        get_similarity_function("damerau_levenshtein")
        is jj.damerau_levenshtein_similarity
    )
    assert get_similarity_function("jaro") is jj.jaro_similarity
    assert get_similarity_function("jaro_winkler") is jj.jaro_winkler_similarity

    # Normalization: case, whitespace, hyphen→underscore
    assert get_similarity_function("  JARO  ") is jj.jaro_similarity
    assert get_similarity_function("jaro-winkler") is jj.jaro_winkler_similarity
    assert get_similarity_function("LeVeNsHtEiN") is jj.levenshtein_similarity

    # raise for other
    with pytest.raises(KeyError):
        jj.get_similarity_function("whatever")


def test_get_similarity_strategy():
    # default to automatic strategy
    output = jj.get_similarity_strategy()
    assert isinstance(output, jj.SimilarityStrategy)

    output = jj.get_similarity_strategy(None)
    assert isinstance(output, jj.SimilarityStrategy)

    # pass through a Strategy subclass
    strategy = jj.PairwiseStrategy()
    output = jj.get_similarity_strategy(strategy)
    assert output is strategy

    # pass through a callable
    def custom_function(x, y):
        return 0.0

    output = jj.get_similarity_strategy(custom_function)
    assert output is custom_function

    # delegate to pairwise
    for strategy in [
        "jaro_winkler",
        "Jaro-Winkler",
        " JaRo-WiNkLeR ",
        "jaro_winkler_similarity",
    ]:
        output = jj.get_similarity_strategy(strategy)
        assert isinstance(output, jj.PairwiseStrategy)
        assert output.similarity_function is jj.jaro_winkler_similarity

    with pytest.raises(ValueError, match=r"^Strategy name 'whatever' must"):
        jj.get_similarity_strategy("whatever")

    # raise for anything else
    with pytest.raises(TypeError):
        jj.get_similarity_strategy(123)


@skip_if_openai_not_available
def test_get_similarity_strategy_openai(left_words, right_words):
    for strategy in ("openai", "OpenAI", " openai "):
        output = jj.get_similarity_strategy(strategy)
        assert isinstance(output, jj.OpenAIEmbeddingStrategy)

    df = jj.jellyjoin(left_words, right_words, strategy="openai")
    assert isinstance(df, pd.DataFrame)


@skip_if_nomic_not_available
def test_get_similarity_strategy_nomic(left_words, right_words):
    for strategy in ["nomic", "NoMiC", " nomic "]:
        output = jj.get_similarity_strategy(strategy)
        assert isinstance(output, jj.NomicEmbeddingStrategy)

    df = jj.jellyjoin(left_words, right_words, strategy="nomic")
    assert isinstance(df, pd.DataFrame)


@skip_if_ollama_not_available
def test_get_similarity_strategy_ollama(left_words, right_words):
    for strategy in ["ollama", "Ollama", " ollama "]:
        output = jj.get_similarity_strategy(strategy)
        assert isinstance(output, jj.OllamaEmbeddingStrategy)

    df = jj.jellyjoin(left_words, right_words, strategy="ollama")
    assert isinstance(df, pd.DataFrame)


def test_jelly_class(left_words, right_words):
    # remember options
    with pytest.raises(ValueError):
        jj.Jelly(suffixes=("_only",))

    with pytest.raises(ValueError):
        jj.Jelly(how="not-a-valid-how")

    jelly = jj.Jelly(
        strategy="jaro",
        threshold=0.4,
        allow_many="both",
        how="left",
        similarity_column="Score",
        suffixes=("_x", "_y"),
        return_similarity_matrix=True,
    )
    df1, mat1 = jelly.join(left_words, right_words)

    assert isinstance(df1, pd.DataFrame)
    assert "Score" in df1.columns
    assert df1["Score"].between(0.0, 1.0).all()

    assert isinstance(mat1, np.ndarray)
    assert mat1.shape == (len(left_words), len(right_words))

    # overrides
    df2 = jelly.join(
        left_words,
        right_words,
        similarity_column="Sim",
        return_similarity_matrix=False,
        threshold=0.9999,
        how="inner",
    )

    assert isinstance(df2, pd.DataFrame)
    assert len(df2) == 0  # threshold is too high for match
    assert "Sim" in df2.columns
    assert "Score" not in df2.columns

    # column names
    left_df = pd.DataFrame({"name": ["alpha", "beta"], "v": [1, 2]})
    right_df = pd.DataFrame({"name": ["alpha", "gamma"], "v": [10, 30]})

    jelly = jj.Jelly(on="name", similarity_column="S", suffixes=("1", "2"))
    df3 = jelly.join(left_df, right_df, strategy="jaro")

    assert isinstance(df3, pd.DataFrame)
    assert "S" in df3.columns
    assert "v1" in df3.columns
    assert "v2" in df3.columns
    assert len(df3) >= 1


def test_jelly_default_on(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        threshold=0.4,
        allow_many="both",
        how="outer",
        on="API Path",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        right_on="UI Field Name",
    )

    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0


def test_jelly_left_right_on_in_constructor(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
    )

    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0
    assert df.columns.tolist() == [
        "Left",
        "Right",
        "Similarity",
        "API Path",
        "Prefix",
        "UI Field Name",
        "Type",
    ]


def test_jelly_left_right_on_in_join(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        left_on="API Path",
        right_on="UI Field Name",
    )

    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0
    assert df.columns.tolist() == [
        "Left",
        "Right",
        "Similarity",
        "API Path",
        "Prefix",
        "UI Field Name",
        "Type",
    ]


def test_jelly_both_on_default_and_right_on_override(left_df, right_df):
    # "on" supplies left_on; right_on override is provided at call time
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        on="API Path",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        right_on="UI Field Name",
    )

    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0
    assert df.columns.tolist() == [
        "Left",
        "Right",
        "Similarity",
        "API Path",
        "Prefix",
        "UI Field Name",
        "Type",
    ]


def test_jelly_rename_output_columns_in_constructor(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
        left_index_column="L",
        right_index_column="R",
        similarity_column="S",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
    )

    assert isinstance(df, pd.DataFrame)
    assert "S" in df.columns and "Similarity" not in df.columns
    assert "L" in df.columns and "Left" not in df.columns
    assert "R" in df.columns and "Right" not in df.columns
    assert df["S"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0


def test_jelly_rename_output_columns_in_join(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        left_index_column="L",
        right_index_column="R",
        similarity_column="S",
    )

    assert isinstance(df, pd.DataFrame)
    assert "S" in df.columns and "Similarity" not in df.columns
    assert "L" in df.columns and "Left" not in df.columns
    assert "R" in df.columns and "Right" not in df.columns
    assert df["S"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0


def test_jelly_drop_output_columns_in_constructor(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
        left_index_column=jj.DROP,
        right_index_column=jj.DROP,
        similarity_column=jj.DROP,
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
    )

    assert isinstance(df, pd.DataFrame)
    assert "Similarity" not in df.columns
    assert "Left" not in df.columns
    assert "Right" not in df.columns
    assert len(df) > 0
    for c in ["API Path", "UI Field Name"]:
        assert c in df.columns


def test_jelly_drop_output_columns_in_join(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        left_index_column=jj.DROP,
        right_index_column=jj.DROP,
        similarity_column=jj.DROP,
    )

    assert isinstance(df, pd.DataFrame)
    assert "Similarity" not in df.columns
    assert "Left" not in df.columns
    assert "Right" not in df.columns
    assert len(df) > 0
    for c in ["API Path", "UI Field Name"]:
        assert c in df.columns


def test_jelly_constructor_rename_overridden_by_join_drop(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
        left_index_column="L",
        right_index_column="R",
        similarity_column="S",
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        left_index_column=jj.DROP,
        right_index_column=jj.DROP,
        similarity_column=jj.DROP,
    )

    assert isinstance(df, pd.DataFrame)
    assert "S" not in df.columns and "Similarity" not in df.columns
    assert "L" not in df.columns and "Left" not in df.columns
    assert "R" not in df.columns and "Right" not in df.columns
    assert len(df) > 0


def test_jelly_constructor_drop_overridden_by_join_rename(left_df, right_df):
    jelly_many_to_many = jj.Jelly(
        strategy=jj.PairwiseStrategy(),
        threshold=0.4,
        allow_many="both",
        how="outer",
        left_on="API Path",
        right_on="UI Field Name",
        left_index_column=jj.DROP,
        right_index_column=jj.DROP,
        similarity_column=jj.DROP,
    )
    df = jelly_many_to_many.join(
        left_df,
        right_df,
        left_index_column="L",
        right_index_column="R",
        similarity_column="S",
    )

    assert isinstance(df, pd.DataFrame)
    assert "S" in df.columns and "Similarity" not in df.columns
    assert "L" in df.columns and "Left" not in df.columns
    assert "R" in df.columns and "Right" not in df.columns
    assert df["S"].dropna().between(0.0, 1.0).all()
    assert len(df) > 0


def test_jelly_on():
    left = pd.DataFrame(
        {
            "name": ["alice", "bob"],
            "id": ["1", "2"],
        }
    )
    right = pd.DataFrame(
        {
            "name": ["Alice", "Bobby"],
            "id": ["20", "10"],
        }
    )

    df_func = jj.jellyjoin(
        left,
        right,
        on="name",
        strategy="jaro_winkler",
        threshold=0.0,
    )
    assert isinstance(df_func, pd.DataFrame)
    assert df_func["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df_func) > 0

    # Jelly defaults
    jelly = jj.Jelly(
        on="name",
        strategy="jaro_winkler",
        threshold=0.0,
    )
    df_class = jelly.join(left, right)

    assert isinstance(df_class, pd.DataFrame)
    assert df_class["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df_class) > 0

    # .join method override
    df_class = jelly.join(left, right, on="id")

    assert isinstance(df_class, pd.DataFrame)
    assert df_class["Similarity"].dropna().between(0.0, 1.0).all()
    assert len(df_class) > 0

    # verify that "on" was override and we are now joining on "id" instead.
    assert df_class.loc[0, "name_left"] == "alice"
    assert df_class.loc[0, "name_right"] == "Bobby"
    assert df_class.loc[1, "name_left"] == "bob"
    assert df_class.loc[1, "name_right"] == "Alice"
