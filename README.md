# Shopper Intent Radar

ML pipeline that scores each website session by its chance of ending in a purchase, so the e-commerce team can target offers at the sessions where they matter.

## 1. Business case

### 1.1 Context
- **Client:** an online retailer whose one-year session log was published as the *Online Shoppers Purchasing Intention* dataset (Sakar et al., 2019, UCI Machine Learning Repository). The retailer is anonymised in the source.
- **Business unit:** E-commerce Marketing team, owner of on-site promotions (discount pop-ups, live chat).
- **Fact:** 12,330 sessions in one year; only **15.5 % (1,908) ended in a purchase**, so 84.5 % of visits leave without buying.
- **Maturity:** web analytics (Google Analytics-style page metrics) are collected, but promotions are triggered by fixed rules and no ML runs in production. Target for this project: MLOps level 0 → 1 (a tested, reproducible pipeline run by one command).

### 1.2 Problem and baseline
- **Pain point:** promotions are shown by a simple rule, so discounts are wasted on visitors who would buy anyway or who will never buy, and most promotion budget does not change behaviour.
- **Current process (baseline):** target every session that visited a page with a positive `PageValues` (a page that has led to purchases before).
- **Baseline value:** this rule flags 22 % of sessions with **precision 0.57** and recall 0.81, giving **PR AUC = 0.49** on the held-out test set (source: rule applied to the cleaned dataset, 20 % stratified test split, seed 42, same split as the pipeline).

### 1.3 Goal and metrics
- **Business KPI:** share of buying sessions found within the 20 % of sessions the team can afford to target. Target: **≥ 70 %** (a random 20 % finds 20 %).
- **ML metric (quality gate):** `pr_auc` (average precision) on the held-out test set.
- **Threshold:** `pr_auc >= 0.65`, **higher is better**, clearly above the 0.49 rule baseline.
- **Why this metric:** buyers are the minority class (15.5 %) and the team acts on a ranked list, so PR AUC measures ranking quality on the class that matters better than accuracy (a model that predicts "no purchase" for everyone already scores 84.5 % accuracy).
- **Limits:** a run below 0.65 is reported as *not met* and its predictions are not used for targeting.

### 1.4 Solution and ML framing
- **Users and decision:** the marketing team decides *which sessions see a promotion*, using the purchase probability.
- **ML framing:** binary classification. Target `Revenue` (True = the session ended in a purchase). Output: a purchase probability per session.
- **Mode:** in this stage the pipeline runs in **batch** (`python -m src.main` scores a session file and writes `predictions.csv`) to prove the model is reliable. Later stages will serve it in real time through an API.
- **Fallback:** if the pipeline fails or the gate is not met, the team keeps the `PageValues > 0` rule.

### 1.5 Data
- **Source:** `online_shoppers_intention.csv`, exported from the retailer's web analytics. **Owner:** web analytics / data team.
- **Size:** 12,330 rows (one per session), 17 features + target: page counts and durations by page type (`Administrative`, `Informational`, `ProductRelated`), `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.
- **Labels:** `Revenue` is recorded automatically when a session completes a transaction; refreshed with each new export.
- **Known issues:** 125 duplicated rows; `Month` has only 10 of 12 months (no January or April) and mixes formats (`June` is spelled out while the others use three letters, such as `Jul`); numeric codes (`Browser`, `Region`, …) are categories, not quantities. The pipeline validates and handles these explicitly.
- **Privacy:** no user ID, name, IP or payment data; sessions cannot be linked to a person.
- **Available at prediction time:** all features describe the session's own browsing, which exists before the purchase decision. *Caveat:* the export stores end-of-session totals, so in real-time use the same features must be computed on the session so far (see Risk 1).

### 1.6 Value, cost and risks
- **Gain over non-AI:** the pipeline's random forest finds **77 %** of buying sessions in the top 20 % of scores (precision 0.61), against 20 % for a random list of the same size, and reaches **PR AUC 0.73** against 0.49 for the rule. *Assumption:* if better targeting moves 1 % of the 10,297 non-buying sessions per year into purchases at an average order of EUR 60, that is about 103 extra orders, roughly **EUR 6,200 per year per 12k sessions**, scaling with traffic.
- **Rough cost:** a few days of team effort; trains and scores on a laptop in seconds, so infrastructure cost is close to zero at this stage.
- **Risk 1, training–serving skew** (model trained on full-session totals, used on partial sessions). *Mitigation:* the batch stage only scores completed-session exports; before real-time release, retrain on features computed at the moment of scoring.
- **Risk 2, seasonality and drift** (traffic peaks in May and November; months missing from training). *Mitigation:* `validate.py` rejects unexpected month and visitor-type values, and the quality gate blocks any run below the threshold.

**Must match `TEAM_BRIEF.md` and `config.yaml`:** metric `pr_auc`, threshold `0.65`, direction `higher_is_better`, target `Revenue`.

## 2. Setup

Requires [conda](https://docs.conda.io/) (Miniconda or Anaconda).

```bash
git clone <this-repo-url>
cd <repo-folder>
conda env create -f environment.yml
conda activate shopper_intent
```

## 3. Get the data

The data is not stored in Git. Download *Online Shoppers Purchasing Intention Dataset* from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) and save the CSV as:

```
data/raw/online_shoppers_intention.csv
```

## 4. Run the pipeline

```bash
python -m src.main
```

One command loads, validates, cleans, splits, trains, evaluates and predicts. It writes:

| Output | Content |
|---|---|
| `data/processed/clean.csv` | cleaned data |
| `models/model.joblib` | fitted pipeline (feature recipe + random forest) |
| `reports/predictions.csv` | purchase probability and 0/1 label per test session |
| `reports/metrics.json` | metric, score, threshold and whether the quality gate is met |
| `logs/pipeline.log` | audit trail of the run |

The last log line reports the quality gate, for example `Pipeline finished: pr_auc MET`. Every setting (paths, columns, model parameters, threshold) is in `config.yaml`.

## 5. Test and check code quality

```bash
pytest --cov=src --cov-branch                      # all tests, branch coverage
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests
pylint src
radon cc -s -a src
black --check . && isort --check-only .
```

Tests use a small synthetic dataset (`tests/conftest.py`), so they run without the real CSV. `tests/test_main.py` runs `main()` end to end.

## 6. Repository structure

```
.
├── config.yaml          # every setting, loaded by main.py
├── environment.yml      # conda environment, Python pinned
├── TEAM_BRIEF.md        # module contract, shared names and rules
├── src/
│   ├── load_data.py     # read raw CSV
│   ├── validate.py      # contract checks, fail fast, never fix
│   ├── clean_data.py    # duplicates, spellings, types
│   ├── features.py      # unfitted feature recipe
│   ├── train.py         # split, fit, save (only place that fits)
│   ├── evaluate.py      # metric and quality gate
│   ├── infer.py         # load model, predict
│   ├── utils.py         # config, logging, file helpers
│   └── main.py          # orchestrates every step
├── tests/               # unit, integration and pipeline tests
└── data/ models/ reports/ logs/   # created by the pipeline, ignored by Git
```

## 7. Team and module owners

| Member | Modules |
|---|---|
| A (admin) | repo setup, `config.yaml`, `utils.py`, `main.py` |
| B | `load_data.py`, `validate.py` |
| C | `clean_data.py` |
| D | `features.py`, `train.py` |
| E | `evaluate.py`, `infer.py` |
