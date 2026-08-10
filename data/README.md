# Local data

Place the original FER2013 dataset at `data/raw/fer2013.csv`. The file must
contain the `emotion`, `pixels`, and `Usage` columns, including the `Training`,
`PublicTest`, and `PrivateTest` partitions.

The dataset is not distributed with the project. Git ignores every file in
this directory except this README. If data preparation stages are added in the
future, use:

- `data/external/` for auxiliary sources
- `data/interim/` for intermediate results
- `data/processed/` for consumption-ready data

Never store samples, private indexes, or CSV copies under `src/`, `tests/`, or
`docs/`.
