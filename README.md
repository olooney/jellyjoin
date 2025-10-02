# Jellyjoin Python Package

"Jellyjoin: the softest of joins."

Join dataframes or lists based on semantic similarity.

Author: Oran Looney
License: MIT
Year: 2025

---

## Installation

```bash
pip install jellyjoin
```

---

## Usage

```python
import jellyjoin

jelly_df = jellyjoin.jellyjoin(
    left_df,
    right_df,
    left_column="Column Name", 
    right_column="Other Column Name", 
    threshold=0.7,
    similarity_strategy=jellyjoin.PairwiseSimilarity(),
)

print(jelly_df)
```


---

## Development

To set up a development environment:

```bash
git clone https://github.com/<your-username>/jellyjoin.git
cd jellyjoin
pip install -e .[dev]
```

Run tests:

```bash
pytest
```
