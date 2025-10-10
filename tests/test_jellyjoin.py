import os
import re

import dotenv
import numpy as np
import pandas as pd
import pytest

import jellyjoin
from jellyjoin import levenshtein_similarity

# -----------------------
# Fixtures
# -----------------------

dotenv.load_dotenv()


@pytest.fixture()
def left_words():
    return ["Cat", "Dog", "Piano"]


@pytest.fixture()
def right_words():
    return ["CAT", "Dgo", "Whiskey"]


@pytest.fixture()
def left_sections():
    return [
        "Introduction",
        "Mathematical Methods",
        "Empirical Validation",
        "Anticipating Criticisms",
        "Future Work",
    ]


@pytest.fixture()
def right_sections():
    return [
        "Abstract",
        "Experimental Results",
        "Proposed Extensions",
        "Theoretical Modeling",
        "Limitations",
    ]


@pytest.fixture()
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


@pytest.fixture()
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
    return jellyjoin.PairwiseSimilarityStrategy()


@pytest.fixture
def pairwise_strategy_jw_lower():
    return jellyjoin.PairwiseSimilarityStrategy(
        "jaro-winkler",
        preprocessor=lambda x: x.lower(),
    )


@pytest.fixture
def pairwise_strategy_levenshtein():
    return jellyjoin.PairwiseSimilarityStrategy(levenshtein_similarity)


@pytest.fixture(scope="session")
def openai_client():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Requires OpenAI key in environment")
    openai = pytest.importorskip("openai")
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture
def openai_strategy(openai_client):
    return jellyjoin.OpenAIEmbeddingSimilarityStrategy(openai_client)


# -----------------------
# Tests
# -----------------------


def test_version():
    assert re.match(r"^\d+\.\d+\.\d+$", jellyjoin.__version__)
    assert jellyjoin.__version__ > "0.0.0"


def test_pairwise_similarity_strategy_defaults(
    pairwise_strategy_default, left_words, right_words
):
    matrix = pairwise_strategy_default(left_words, right_words)
    expected = np.array(
        [
            [0.33333333, 0.0, 0.0],
            [0.0, 0.66666667, 0.0],
            [0.0, 0.2, 0.14285714],
        ]
    )
    assert np.allclose(matrix, expected)


def test_pairwise_similarity_strategy(
    pairwise_strategy_jw_lower, left_words, right_words
):
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


@pytest.mark.parametrize(
    "left,right",
    [
        ([], ["X"]),  # left empty
        (["X"], []),  # right empty
        ([], []),  # both empty
    ],
)
def test_jellyjoin_empty(left, right):
    df = jellyjoin.jellyjoin(left, right)
    assert df.columns.tolist() == [
        "Left",
        "Right",
        "Similarity",
        "Left Value",
        "Right Value",
    ]
    assert len(df) == 0


def test_jellyjoin_with_lists(left_sections, right_sections):
    df = jellyjoin.jellyjoin(left_sections, right_sections)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(left_sections), len(right_sections))
    assert df["Similarity"].between(0.0, 1.0).all()


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
        threshold=0.6,
        allow_many=allow_many,
    )
    assert isinstance(df, pd.DataFrame)
    assert df["Similarity"].between(0.0, 1.0).all()


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="no API key")
def test_openai_strategy(openai_strategy, left_sections, right_sections):
    matrix = openai_strategy(left_sections, right_sections)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_sections), len(right_sections))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)


# too expensive to run all the time...
@pytest.mark.skipif(True, reason="no API key")
def test_openai_strategy_batch(openai_strategy):
    LENGTH = 5000
    left = ["test"] * LENGTH
    right = ["testing"]
    matrix = openai_strategy(left, right)
    assert matrix.shape == (LENGTH, 1)


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="no API key")
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
