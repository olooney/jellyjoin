# Similarity Strategies in Jellyjoin

Start by importing:

```python
import jellyjoin.jj as jj
```

Most strategies require an optional package install and/or additional setup.
You can install all the optional packages with:

```bash
pip install jellyjoin[extra]
```

However, this may pull in more than you need, and additional setup may still
be necessary.

See each strategy for details.

# Out-of-the-Box Similarity Strategies

## PairwiseStrategy

Uses traditional string similarity metrics like Jaro-Winkler or Damerau-Levenshtein.

### Installation
No extra dependencies required.

### Simple Example
```python
from jellyjoin import PairwiseStrategy

strategy = PairwiseStrategy()
scores = strategy(["apple", "banana"], ["apple", "orange"])
```

### Custom Example
```python
import jellyjoin as jj

from jellyjoin.similarity import (
    jaro_winkler_similarity,
    damerau_levenshtein_similarity,
)

def lower(x):
    return x.lower()

def email_similarity(x: str, y: str) -> float:
    def local(email: str) -> str:
        return email.split("@", 1)[0]

    jw_local = jaro_winkler_similarity(local(x), local(y))
    dl_full = damerau_levenshtein_similarity(x, y)
    return max(jw_local, dl_full)

strategy = jj.PairwiseStrategy(
    similarity_function=email_similarity,
    preprocessor=lower,
)
scores = strategy(["some.guy@gmail.com"], ["someone@guy.com"])
```

## OpenAIEmbeddingStrategy

Uses OpenAI embeddings to compute vector-based similarities.

### Installation
```bash
pip install openai tiktoken
```

### Simple Example

If you have the `OPENAI_API_KEY` environment variable set up, you can
simply run:

```python
from jellyjoin import OpenAIEmbeddingStrategy

strategy = OpenAIEmbeddingStrategy()
scores = strategy(["apple", "banana"], ["apple", "orange"])
```

### Azure OpenAI Example

You'll need some extra dependencies:

```bash
pip install azure-identity openai tiktoken
```

Then you can instantiate the `AzureOpenAI` client in this usual way (if you're
using Auzre, this should already be familiar to you):

```python
from jellyjoin import OpenAIEmbeddingStrategy
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

credential = DefaultAzureCredential()
azure_client = AzureOpenAI(
    azure_endpoint="https://<your-resource-name>.openai.azure.com/",
    credential=credential
)
```

Then simply pass the azure client into the strategy:

```python
strategy = OpenAIEmbeddingStrategy(client=azure_client)
scores = strategy(["apple"], ["orange"])
```


### Custom Example

You can use the class with other embedding models or even other OpenAI
comptible servers with some configuration. However, you will need to look up
the details such as `max_tokens` to ensure truncation works correctly for
other embedding models.

```python
from jellyjoin import OpenAIEmbeddingStrategy
import openai

client = openai.OpenAI(
    base_url="<your-openai-compatible-server>",
    api_key="your-api-key"
)
strategy = OpenAIEmbeddingStrategy(
    embedding_model="text-embedding-ada-002",
    batch_size=2048,
    max_tokens=8191,
    encoding="p50k_base",
)
scores = strategy(["apple"], ["orange"])
```


---

## Azure OpenAI

You can also connect using Azure OpenAI with `azure.identity.DefaultAzureCredentials`.

### Installation

---

## NomicEmbeddingStrategy

Uses local embeddings via Nomic's GPT4All backend. Note that the first time
you call Nomic, it will down several hundred megabytes of weights and save
them to the `~/.cache` directory.

### Installation
```bash
pip install nomic
```

### Simple Example
```python
from jellyjoin import NomicEmbeddingStrategy

strategy = NomicEmbeddingStrategy()
scores = strategy(["apple", "banana"], ["apple", "orange"])
```

Nomic will try to detect a GPU and use it if possible, but you can specify the
device if you want. Nomic embedding models can be returned with reduced
dimensionality for increased speed, and you can specify several "task_types" to
specialize it for the kind of text input you're using. See the [nomic][NMC]
docs for details.

[NMC]: https://www.nomic.ai/blog/posts/nomic-embed-text-v1

### Example (custom task and dimensionality)
```python
from jellyjoin import NomicEmbeddingStrategy

strategy = NomicEmbeddingStrategy(task_type="classification", dimensionality=256, device="cpu")
scores = strategy(["apple"], ["orange"])
```


## OllamaEmbeddingStrategy

Uses Ollama's local embedding API. Requires the Ollama server to be running and
reachable, so you must start the Ollama server in a separate process before
using this strategy.

### Installation and Setup

Install the Python client package:

```bash
pip install ollama
```

You'll need to ensure ollama is running, for example by running this in another terminal:

```
ollama serve
```

Then you can pull the embedding model you want, e.g.:

```
ollama pull nomic-embed-text:v1.5
```

### Simple Example
```python
from jellyjoin import OllamaEmbeddingStrategy

strategy = OllamaEmbeddingStrategy()
scores = strategy(["apple", "banana"], ["apple", "orange"])
```

The simple example assumes the model is running `http://localhost:11434`, which
is Ollama's default. You can also point it somewhere else if you want:

### Example (custom model and host)
```python
from jellyjoin import OllamaEmbeddingStrategy

strategy = OllamaEmbeddingStrategy(
    embedding_model="mxbai-embed-large:latest",
    host="http://<external-server>:11434",
)
scores = strategy(["apple"], ["orange"])
```

---

# Fully Custom Similarity Strategies

You can also create your own custom similarity strategy by subclassing either `jellyjoin.typing.SimilarityStrategy`
(if you want to start from scratch) or `jellyjoin.strategy.EmbeddingStrategy` (if you just want to change the underlying
embedding model call and use the same matrix multiplication trick as the built-in strategies.

## 1. Subclassing SimilarityStrategy

Use this when you want full control of pairwise comparisons without embeddings.

Implement `__call__(left_texts, right_texts)` and return an N x M NumPy array
containing similarity scores in [0, 1].

```python
import numpy as np
from collections.abc import Collection
from jellyjoin.typing import SimilarityStrategy
from numpy.random import random

class RandomStrategy(SimilarityStrategy):
    def __init__(self):
        pass

    def __call__(self, left_texts: Collection[str], right_texts: Collection[str]) -> np.ndarray:
        n, m = len(left_texts), len(right_texts)
        out = np.zeros((n, m), dtype=float)

        def score(a: str, b: str) -> float:
            return random()

        for i, lx in enumerate(left_texts):
            for j, ry in enumerate(right_texts):
                out[i, j] = score(lx, ry)

        return out
```

## 2. Subclassing EmbeddingStrategy

Use this when you just want to integrate your own embedding model.
Implement the `.embed(self, texts)` method and return a 2D array of shape `(len(texts), D)`.
The base class handles the rest.

```python
import numpy as np
from collections.abc import Collection
from jellyjoin.strategy import EmbeddingStrategy

class MyEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self, preprocessor=lambda x: x, dtype=np.float32):
        super().__init__(preprocessor=preprocessor)
        self.dtype = dtype

    def embed(self, texts: Collection[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, 0), dtype=self.dtype)

        # call your embedding model and stack the results into a single numpy matrix.

        return matrix
```

It might be a good idea to look at the code for the [built-in strategies][JJGHS] to
get a good idea of how it should work.

[JJGHS]: https://github.com/olooney/jellyjoin/blob/main/src/jellyjoin/strategy.py
