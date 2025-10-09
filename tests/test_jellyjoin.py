import os
import re

import numpy as np
import pandas as pd
import pytest
import dotenv

import jellyjoin
from jellyjoin.similarity import levenshtein_similarity

dotenv.load_dotenv()

left_words = ["Cat", "Dog", "Piano"]
right_words = ["CAT", "Dgo", "Whiskey"]

left_sections = [
    "Introduction",
    "Mathematical Methods",
    "Empirical Validation",
    "Anticipating Criticisms",
    "Future Work",
]
right_sections = [
    "Abstract",
    "Experimental Results",
    "Proposed Extensions",
    "Theoretical Modeling",
    "Limitations",
]

left_df = pd.DataFrame(
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
left_df["Prefix"] = left_df["API Path"].str.split(".", n=1).str[0]


right_df = pd.DataFrame(
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


def test_version():
    assert re.match(r"^\d+\.\d+\.\d+$", jellyjoin.__version__)
    assert jellyjoin.__version__ > "0.0.0"


def test_pairwise_similarity_strategy_defaults():
    strategy = jellyjoin.PairwiseSimilarityStrategy()
    matrix = strategy(left_words, right_words)

    expected = np.array(
        [
            [0.33333333, 0.0, 0.0],
            [0.0, 0.66666667, 0.0],
            [0.0, 0.2, 0.14285714],
        ]
    )
    assert np.allclose(matrix, expected)


def test_pairwise_similarity_strategy():
    strategy = jellyjoin.PairwiseSimilarityStrategy(
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


def test_pairwise_strategy_with_custom_function():
    strategy = jellyjoin.PairwiseSimilarityStrategy(levenshtein_similarity)
    matrix = strategy(left_words, right_words)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_words), len(right_words))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)


def test_pairwise_strategy_square():
    strategy = jellyjoin.PairwiseSimilarityStrategy()
    matrix = strategy(left_sections, left_sections)
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
    examples = [
        ([], ["X"]),
        (["X"], []),
        ([], []),
    ]
    for left, right in examples:
        df = jellyjoin.jellyjoin(left, right)
        assert df.columns.tolist() == [
            "Left",
            "Right",
            "Similarity",
            "Left Value",
            "Right Value",
        ]
        assert len(df) == 0


def test_jellyjoin_with_lists():
    df = jellyjoin.jellyjoin(left_sections, right_sections)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(left_sections), len(right_sections))
    assert df["Similarity"].between(0.0, 1.0).all()


def test_jellyjoin_with_dataframes_all_hows():
    for how in ["inner", "left", "right", "outer"]:
        df = jellyjoin.jellyjoin(
            left_df,
            right_df,
            left_on="API Path",
            right_on="UI Field Name",
            threshold=0.4,
            how=how,
        )
        assert isinstance(df, pd.DataFrame)
        if "similarity" in df.columns:
            assert df["similarity"].between(0.0, 1.0).all()


@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ, reason="Requires OpenAI key in environment"
)
def test_openai_strategy_if_available():
    if not hasattr(jellyjoin, "OpenAIEmbeddingSimilarityStrategy"):
        pytest.skip("OpenAIEmbeddingSimilarityStrategy not implemented")
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    strategy = jellyjoin.OpenAIEmbeddingSimilarityStrategy(client)
    matrix = strategy(left_sections, right_sections)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(left_sections), len(right_sections))
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)
