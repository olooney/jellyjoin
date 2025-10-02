import numpy as np
import jellyjoin

def test_pairwise_similarity_strategy_defaults():
    strategy = jellyjoin.PairwiseSimilarity()
    similarity_matrix = strategy(
        ["Cat", "Dog", "Piano"],
        ["CAT", "Dgo", "Whiskey"],
    )

    expected = np.array([
        [0.33333333, 0.        , 0.        ],
        [0.        , 0.66666667, 0.        ],
        [0.        , 0.2       , 0.14285714],
    ])

    assert np.allclose(similarity_matrix, expected)


def test_pairwise_similarity_strategy():
    similarity_strategy = jellyjoin.PairwiseSimilarity(
        "jaro-winkler",
        preprocessor=lambda x: x.lower(),
    )
    similarity_matrix = similarity_strategy(
        ["Cat", "Dog", "Piano"], 
        ["CAT", "Dgo", "Whiskey"],
    )
    expected = np.array([
        [1.        , 0.        , 0.        ],
        [0.        , 0.55555556, 0.        ],
        [0.51111111, 0.        , 0.44761905],
    ])
    assert np.allclose(similarity_matrix, expected)
    
