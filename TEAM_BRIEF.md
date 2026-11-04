# Team brief

How we build it. `README.md` says *why*; this file fixes the names, keys and rules every module must share. Changes go through a pull request.

## Settings that must match README.md and config.yaml

| Key | Value |
|---|---|
| Target column | `Revenue` (0 = no purchase, 1 = purchase after cleaning) |
| Metric | `pr_auc` (average precision on the held-out test set) |
| Threshold | `0.65` |
| Direction | `higher_is_better` |
| Split | stratified, `test_size = 0.2`, `random_state = 42` |

## Columns

- **Numeric (10):** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
- **Categorical (7):** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend` (number codes are categories, stored as text after cleaning)

## Module contract

Each function takes the config values it needs as arguments; only `main.py` reads `config.yaml`.

| Module | Function | Input → output | Raises |
|---|---|---|---|
| `utils.py` | `load_config(path)` | YAML file → dict | missing file, not a mapping, missing section |
| | `setup_logging(log_file, level)`, `save_dataframe`, `save_json`, `ensure_parent_dir` | | |
| `load_data.py` | `load_raw_data(path)` | CSV → raw DataFrame, unchanged | missing file, not `.csv`, empty, no rows |
| `validate.py` | `validate_dataframe(df, data_config)` | raw DataFrame → same DataFrame, **never fixed** | empty, missing columns, missing values, non-numeric or negative numbers, unknown month or visitor type, non-binary or one-class target |
| `clean_data.py` | `clean_dataframe(df_raw, target_column, categorical_columns, value_aliases)` | raw → clean copy; input untouched, duplicates dropped, index reset to 0..n-1 | empty, column not found |
| `features.py` | `build_feature_recipe(numeric_columns, categorical_columns)` | → **unfitted** `ColumnTransformer` | no columns, overlapping lists |
| `train.py` | `split_data(df, target_column, test_size, random_state)` | → `x_train, x_test, y_train, y_test`, indices kept | target missing, bad test size |
| | `train_model(x_train, y_train, recipe, model_params)` | → fitted `Pipeline` (**the only place that fits**) | empty, size mismatch, bad parameter |
| | `save_model(model, path)` | → `.joblib` file | |
| `evaluate.py` | `evaluate_model(model, x_test, y_test, metric_name)` | → float | unknown metric, empty, size mismatch |
| | `check_quality_gate(score, threshold, direction)` | → bool, logs MET / NOT MET | unknown direction |
| `infer.py` | `load_model(path)` | `.joblib` → `Pipeline` | missing file |
| | `predict(model, x_new, decision_threshold, output_columns)` | → probability + 0/1 label, same index as input | empty, threshold outside (0, 1) |
| `main.py` | `main(config_path)` | runs every step, writes `clean.csv`, `model.joblib`, `predictions.csv`, `metrics.json` | |

## Rules

1. **No hard-coded values:** paths, column names, parameters and thresholds live in `config.yaml`.
2. **Logging, not print:** one `logger = logging.getLogger(__name__)` per module; lazy format `logger.info("%d rows", n)`, never f-strings in logger calls.
3. **Fail fast:** bad data raises `ValueError` with a clear message as early as possible; every `raise` has a test.
4. **Nothing learned before the split:** only `train.py` calls `fit`.
5. **Tests per module:** unit tests plus one test named `test_integration_*` that connects the module with an upstream neighbour. Tests use the synthetic data in `tests/conftest.py`, never the real CSV.
6. **Git:** work only on `feature/<module>` branches from an up-to-date `dev`; PR into `dev`, one teammate approves; never commit to `dev` or `main` directly. No data, models, reports, logs or `.env` in Git.
7. **Before every PR:** `pytest --cov=src --cov-branch`, `flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests`, `black .`, `isort .`.

## Owners

| Member | Branch | Files |
|---|---|---|
| A (admin) | `feature/team-setup`, `feature/main` | config and repo files, `utils.py`, `main.py` + tests |
| B | `feature/load-validate` | `load_data.py`, `validate.py` + tests |
| C | `feature/clean-data` | `clean_data.py` + tests |
| D | `feature/features-train` | `features.py`, `train.py` + tests |
| E | `feature/evaluate-infer` | `evaluate.py`, `infer.py` + tests |

Merge order into `dev`: A setup → B → C → D → E → A main. Each PR's tests pass once the PRs before it are merged.
