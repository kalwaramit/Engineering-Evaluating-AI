# Engineering-Evaluating-AI

A Python ML pipeline for support-ticket classification using TF-IDF features and tree-based classifiers.

## What this project does

- Loads ticket datasets from CSV files.
- Cleans and de-duplicates ticket text.
- Builds TF-IDF embeddings from ticket summary + interaction content.
- Trains/evaluates classification models for:
  - `Type2` (`y2`)
  - `Type2+Type3` (`y2 + y3`)
  - `Type2+Type3+Type4` (`y2 + y3 + y4`)
- Prints `classification_report` metrics for each target.

## Project structure

- `main.py` - Entry point (data load, preprocess, embedding, modelling).
- `preprocess.py` - Data loading, de-duplication, noise removal, optional translation.
- `embeddings.py` - TF-IDF embedding generation.
- `modelling/data_model.py` - Train/test split preparation and label chain creation.
- `modelling/modelling.py` - Model training/evaluation loop.
- `model/` - Model implementations (`randomforest.py`, `adaboost.py`, `hist_gb.py`, `random_trees_ensembling.py`, `SDG.py`).
- `Config.py` - Central column/config constants.
- `data/` - Input CSV files (`AppGallery.csv`, `Purchasing.csv`).

## Requirements

- Python 3.10+ (tested with Python 3.13)
- pip

Install dependencies:

```bash
pip install numpy pandas scikit-learn transformers stanza
```

## Data expectations

Input CSV files should include these columns (or equivalent renamed columns):

- `Ticket Summary`
- `Interaction content`
- `Type 1`, `Type 2`, `Type 3`, `Type 4`
- `Ticket id`

During load:

- `Type 1..4` are renamed to `y1..y4`
- target `y` is set from `y2`

## Run

From project root:

```bash
python main.py
```

If using the project virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

## Notes on output

- You may see low precision/recall for rare classes in chained targets (`Type2+Type3`, `Type2+Type3+Type4`).
- `UndefinedMetricWarning` can occur when some classes have no predicted samples; this is common for highly imbalanced classes.

## Optional translation

`preprocess.py` includes `translate_to_en(...)` based on `stanza` + `transformers` (`facebook/m2m100_418M`).

- It is currently not active in `main.py` (commented line).
- Enabling it may download large models and increase runtime significantly.

## Reproducibility

Random seeds are set in multiple modules (`seed = 0`) for more stable runs.
