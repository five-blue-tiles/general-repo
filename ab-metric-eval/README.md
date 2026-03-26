# ab-metric-eval

Statistical analysis tools for A/B experiment metric evaluation.

`ab-metric-eval` provides two general-purpose functions for analysing experiment results from any pandas DataFrame — no Spark or platform-specific dependencies required.

## Features

- **Experiment summaries** — group-level enrollment counts, crossover and exclusion rates, and a daily enrollment timeline chart
- **Hypothesis testing** — Welch's t-test (continuous) and proportions z-test (binary), with configurable significance level
- **CUPED variance reduction** — optional pre-exposure covariate adjustment to increase statistical power
- **Winsorization** — symmetric or asymmetric outlier clipping for continuous metrics
- **Pre-exposure bias testing** — diagnostic check for randomisation imbalance before drawing conclusions
- **Flexible imputation** — zero-fill or drop-missing strategies for pre/post pivoting

## Installation

```bash
pip install ab-metric-eval
```

## Quick start

```python
import pandas as pd
from ab_metric_eval import generate_experiment_summary, analyze_experiment_metric

# df is a pandas DataFrame with one or two rows per unit
# (pre-exposure and post-exposure), containing at minimum:
#   experiment_group, analysis_unit_id, analysis_exclusion_flag,
#   crossover_flag, first_exposure_utc_date

# 1. Summarise enrollment and group balance
summary_df, enrollment_df, fig = generate_experiment_summary(df)

# 2. Run a hypothesis test on a continuous metric
summary, results, fig, winsor_summary = analyze_experiment_metric(
    df=df,
    metric_column="revenue",
    pre_exposure_eval_column="pre_revenue",   # optional, enables CUPED + bias test
    metric_type="continuous",
    winsorize_percentile=0.99,
    apply_cuped=True,
    alpha=0.05,
)
```

## API reference

### `generate_experiment_summary(df)`

Generates group-level summary statistics and a daily enrollment timeline.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Per-unit experiment data. Required columns: `experiment_group`, `name`, `analysis_unit_id`, `analysis_exclusion_flag`, `crossover_flag`, `first_exposure_utc_date`. Optional: `metric_time_window`. |

**Returns** `(summary_df, enrollment_df, fig)` — group summary, daily enrollment counts, matplotlib figure.

---

### `analyze_experiment_metric(df, metric_column, ...)`

Runs a full statistical analysis on a single metric, including optional pre-exposure bias testing and CUPED adjustment.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | Per-unit data (pre- and/or post-exposure rows). |
| `metric_column` | `str` | — | Post-exposure metric column to test. |
| `pre_exposure_eval_column` | `str \| None` | `None` | Pre-exposure metric column. Required for bias test and CUPED. |
| `metric_type` | `str` | `'continuous'` | `'continuous'` or `'binary'`. |
| `winsorize_percentile` | `float \| tuple \| None` | `None` | Winsorization bounds. A single float (e.g. `0.99`) clips symmetrically at the (1%, 99%) percentiles. A tuple `(lower, upper)` allows asymmetric clipping; use `None` for either end to skip that tail. |
| `alpha` | `float` | `0.05` | Significance level for the hypothesis test. |
| `apply_cuped` | `bool` | `False` | Apply CUPED adjustment (requires `pre_exposure_eval_column`; not supported for binary metrics). |
| `imputation_strategy` | `str` | `'zero'` | How to handle units missing pre- or post-exposure rows after pivoting: `'zero'` imputes with 0, `'drop'` removes the unit. |
| `include_exclusion_flag` | `bool` | `False` | If `False`, units where `analysis_exclusion_flag` is `True` are removed. |
| `include_crossover_flag` | `bool` | `False` | If `False`, units where `crossover_flag` is `True` are removed. |
| `bias_test_alpha` | `float \| None` | `None` | Significance threshold for the pre-exposure balance diagnostic. When `None`, a graduated reporting scheme is used instead of a binary pass/fail. |

**Returns** `(summary_df, test_results_df, fig, winsorization_summary_df)`.

---

### Data classes

`ExperimentConfig` and `MetricDefinition` are convenience dataclasses for grouping experiment parameters and metric definitions respectively.

```python
from ab_metric_eval import ExperimentConfig, MetricDefinition

config = ExperimentConfig(
    expt_name="my-experiment",
    enrollment_date_start="2025-01-01",
    enrollment_date_end_incl="2025-01-31",
    analysis_date_start="2025-02-01",
    analysis_date_end_incl="2025-02-28",
    unit_type="userId",
)

metric = MetricDefinition(
    name="revenue",
    sql_expression="SUM(revenue)",
    description="Total revenue per user",
)
```

## Requirements

- Python 3.8+
- numpy >= 1.19
- pandas >= 1.0
- scipy >= 1.5
- statsmodels >= 0.12
- matplotlib >= 3.3
- seaborn >= 0.11

## License

MIT
