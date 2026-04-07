"""
Statistical analysis functions for experiment evaluation.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportion_confint

logger = logging.getLogger(__name__)


def generate_experiment_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Generate summary statistics and enrollment timeline for experiment groups.
    
    Handles both single-record-per-unit and pre/post-exposure record structures.
    For pre/post structures, deduplicates to unit level before summarizing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing experiment data with required columns:
        - experiment_group
        - name (experiment name)
        - analysis_unit_id
        - analysis_exclusion_flag
        - crossover_flag
        - first_exposure_utc_date
        - metric_time_window (optional: 'pre-exposure' or 'post-exposure')
    
    Returns:
    --------
    tuple: (summary_df, enrollment_df, fig) - Summary stats, enrollment data, and matplotlib figure
    """
    
    # Validate required column
    if 'experiment_group' not in df.columns:
        raise ValueError("DataFrame must contain 'experiment_group' column")
    
    # Deduplicate to unit level if metric_time_window column exists
    if 'metric_time_window' in df.columns:
        # [CHANGE #5] Prefer post-exposure rows when deduplicating, with defensive
        # validation that exposure-level columns are consistent within each unit.
        # Previously used keep='first' which was non-deterministic due to arbitrary
        # DataFrame ordering from Spark-to-Pandas conversion.
        invariant_cols = ['experiment_group', 'analysis_exclusion_flag', 'crossover_flag']
        for col in invariant_cols:
            if col in df.columns:
                nunique = df.groupby('analysis_unit_id')[col].nunique()
                if (nunique > 1).any():
                    n_varying = (nunique > 1).sum()
                    logger.warning(
                        f"Column '{col}' varies within {n_varying} unit(s) — "
                        f"post-exposure value will be used"
                    )
        
        # Sort so post-exposure comes last, then keep last record per unit
        time_window_order = {'pre-exposure': 0, 'post-exposure': 1}
        df_sorted = df.copy()
        df_sorted['_tw_sort'] = df_sorted['metric_time_window'].map(time_window_order).fillna(-1)
        df_sorted = df_sorted.sort_values('_tw_sort')
        df_deduplicated = df_sorted.drop_duplicates(subset=['analysis_unit_id'], keep='last')
        df_deduplicated = df_deduplicated.drop(columns=['_tw_sort'])
        
        print(f"Data structure: Pre/Post exposure records detected")
        print(f"Original records: {len(df)}, Deduplicated to units: {len(df_deduplicated)}")
        df_working = df_deduplicated
    else:
        df_working = df
        print(f"Data structure: Single record per unit")
    
    # Summary Statistics by Experiment Group
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY EXPERIMENT GROUP")
    print("=" * 80)
    
    summary = df_working.groupby('experiment_group').agg(
        experiment_name=('name', 'first'),
        total_units=('analysis_unit_id', 'nunique'),
        pct_exclusion=('analysis_exclusion_flag', lambda x: (x.sum() / len(x) * 100).round(2)),
        pct_crossover=('crossover_flag', lambda x: (x.sum() / len(x) * 100).round(2))
    ).reset_index()
    
    print("\n")
    print(summary.to_string(index=False))
    print("\n")
    
    # Enrollment Timeline Chart
    print("=" * 80)
    print("ENROLLMENT TIMELINE")
    print("=" * 80)
    
    # Prepare data for chart: count units by date and experiment group
    enrollment_data = df_working.groupby(['first_exposure_utc_date', 'experiment_group']).agg(
        units_enrolled=('analysis_unit_id', 'nunique')
    ).reset_index()
    
    # Create the line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=enrollment_data,
        x='first_exposure_utc_date',
        y='units_enrolled',
        hue='experiment_group',
        marker='o',
        ax=ax
    )
    
    ax.set_xlabel('Enrollment Date', fontsize=12)
    ax.set_ylabel('Number of Units Enrolled', fontsize=12)
    ax.set_title('Daily Enrollment by Experiment Group', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    ax.legend(title='Experiment Group')
    plt.tight_layout()
    
    return summary, enrollment_data, fig


def analyze_experiment_metric(df, metric_column, pre_exposure_eval_column=None, 
                             metric_type='continuous', winsorize_percentile=None, 
                             alpha=0.05, include_exclusion_flag=False, 
                             include_crossover_flag=False, apply_cuped=False,
                             imputation_strategy='zero',
                             bias_test_alpha=None):
    """
    Analyze and visualize the distribution of a metric across experiment groups with hypothesis testing.
    Includes optional pre-experimental bias testing and CUPED adjustment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The per-unit dataframe (with 2 records per unit: pre and post-exposure, or 1 record if pre-exposure not applicable)
    metric_column : str
        Post-exposure metric column name
    pre_exposure_eval_column : str, optional
        Pre-exposure metric column name (for bias test and CUPED). If None, all pre-exposure checks are skipped.
        Default: None
    metric_type : str
        Type of metric: 'continuous' or 'binary'
    winsorize_percentile : float or tuple of (float, float), optional
        Winsorization bounds. A single float (e.g., 0.99) is treated as symmetric:
        clips at (1 - value, value) percentiles, i.e. (0.01, 0.99). A tuple
        (lower, upper) allows asymmetric clipping. Use None for either bound to
        skip that tail, e.g. (None, 0.99) for upper-only clipping.
        [CHANGE #1] Previously only clipped the upper tail, which introduced
        asymmetric bias for metrics with outliers on both ends.
    alpha : float
        Significance level for hypothesis test (default: 0.05)
    include_exclusion_flag : bool
        If False (default), excludes records where analysis_exclusion_flag = True
    include_crossover_flag : bool
        If False (default), excludes records where crossover_flag = True
    apply_cuped : bool
        If True, applies CUPED adjustment using pre-exposure metric (requires pre-exposure variation)
        NOTE: CUPED is not supported for binary metrics and will raise an error.
        NOTE: CUPED requires pre_exposure_eval_column to be specified.
    imputation_strategy : str
        Strategy for handling missing metric values after pre/post pivoting.
        'zero' (default): impute NaN with 0 — appropriate for count/sum metrics
        'drop': drop units with missing values — appropriate for metrics where 0
            is a meaningful value (e.g., satisfaction scores)
        [CHANGE #6] Previously hardcoded to zero-imputation with no option to
        override. Users should choose based on metric semantics.
    bias_test_alpha : float, optional
        Significance thresholds for the pre-exposure bias diagnostic. If None
        (default), uses a graduated reporting scheme instead of a binary
        threshold: p < 0.01 = "strong evidence of imbalance", p < 0.10 = "mild
        evidence", p >= 0.10 = "no evidence".
        [CHANGE #11] Previously reused the main test alpha (typically 0.05) for
        the bias diagnostic, which has a ~5% false alarm rate on properly
        randomized experiments. The graduated scheme avoids binary overreaction.
    
    Returns:
    --------
    tuple: (summary_df, test_results_df, fig, winsorization_summary_df)
    """
    
    # Validate inputs
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
    if metric_type not in ['continuous', 'binary']:
        raise ValueError("metric_type must be 'continuous' or 'binary'")
    if metric_column not in df.columns:
        raise ValueError(f"Column '{metric_column}' not found in dataframe")
    
    # [CHANGE #6] Validate imputation_strategy
    if imputation_strategy not in ['zero', 'drop']:
        raise ValueError("imputation_strategy must be 'zero' or 'drop'")
    
    # [CHANGE #1] Normalize winsorize_percentile to a (lower, upper) tuple
    winsorize_bounds = None
    if winsorize_percentile is not None:
        if isinstance(winsorize_percentile, (int, float)):
            # Single float = symmetric: e.g., 0.99 -> (0.01, 0.99)
            winsorize_bounds = (1 - winsorize_percentile, winsorize_percentile)
        elif isinstance(winsorize_percentile, tuple) and len(winsorize_percentile) == 2:
            winsorize_bounds = winsorize_percentile
        else:
            raise ValueError(
                "winsorize_percentile must be a float (symmetric) or a "
                "tuple of (lower, upper) percentiles. Use None for either "
                "bound to skip that tail."
            )
        # Validate bounds
        for i, bound in enumerate(winsorize_bounds):
            if bound is not None and (bound < 0 or bound > 1):
                raise ValueError(f"Winsorization bound {bound} must be between 0 and 1")
        if (winsorize_bounds[0] is not None and winsorize_bounds[1] is not None 
                and winsorize_bounds[0] >= winsorize_bounds[1]):
            raise ValueError("Lower winsorization bound must be less than upper bound")
    
    # [CHANGE #11] Validate bias_test_alpha if provided
    if bias_test_alpha is not None and (bias_test_alpha <= 0 or bias_test_alpha >= 1):
        raise ValueError("bias_test_alpha must be between 0 and 1")
    
    # CUPED validation
    if apply_cuped and pre_exposure_eval_column is None:
        raise ValueError(
            "CUPED adjustment requires pre_exposure_eval_column to be specified. "
            "Set pre_exposure_eval_column or set apply_cuped=False."
        )
    
    if metric_type == 'binary' and apply_cuped:
        raise ValueError(
            "CUPED adjustment is not supported for binary metrics. "
            "CUPED assumes a linear relationship between pre and post metrics, which is "
            "inappropriate for binary outcomes. Set apply_cuped=False for binary metrics."
        )
    
    # Check if pre-exposure column is provided and exists
    if pre_exposure_eval_column is not None and pre_exposure_eval_column not in df.columns:
        raise ValueError(f"Column '{pre_exposure_eval_column}' not found in dataframe")
    
    # Create a copy to avoid modifying the original dataframe
    df_working = df.copy()
    
    # Check if pre-exposure records exist in the data
    has_pre_exposure_records = False
    if 'metric_time_window' in df_working.columns:
        pre_exposure_count = (df_working['metric_time_window'] == 'pre-exposure').sum()
        has_pre_exposure_records = pre_exposure_count > 0
        print(f"Detected {pre_exposure_count} pre-exposure records in dataset")
    
    # Determine if we should process pre-exposure data
    process_pre_exposure = (pre_exposure_eval_column is not None) and has_pre_exposure_records
    
    if not process_pre_exposure:
        if pre_exposure_eval_column is None:
            print("\nPre-exposure analysis SKIPPED: pre_exposure_eval_column not specified")
        elif not has_pre_exposure_records:
            print("\nPre-exposure analysis SKIPPED: No pre-exposure records detected in dataset")
        
        # Filter for post-exposure records only
        if 'metric_time_window' in df_working.columns:
            print("Filtering for post-exposure records only...")
            df_working = df_working[df_working['metric_time_window'] == 'post-exposure'].copy()
            print(f"Records after filtering: {len(df_working)}")
        
        post_col = metric_column
        pre_col = None
    else:
        print("\nPre-exposure analysis ENABLED: Processing pre and post-exposure data")
        
        # Pivot pre/post data if metric_time_window column exists
        if 'metric_time_window' in df_working.columns:
            print("Pivoting pre/post-exposure records to single row per unit...")
            
            # Identify grouping columns (all except metric columns and metric_time_window)
            cols_to_exclude = ['metric_time_window', metric_column, pre_exposure_eval_column]
            cols_to_exclude = [col for col in cols_to_exclude if col is not None]
            grouping_cols = [col for col in df_working.columns if col not in cols_to_exclude]
            
            # Pivot metrics
            pivot_cols = {}
            metrics_to_pivot = [metric_column]
            if pre_exposure_eval_column is not None:
                metrics_to_pivot.append(pre_exposure_eval_column)
            
            for metric_col in metrics_to_pivot:
                pivot_df = df_working.pivot_table(
                    index='analysis_unit_id',
                    columns='metric_time_window',
                    values=metric_col,
                    aggfunc='first'
                )
                # Create proper column names with underscore
                pivot_df.columns = [f"{metric_col}_{col.replace('-', '_')}" for col in pivot_df.columns]
                pivot_cols[metric_col] = pivot_df
            
            # Get grouping columns (take first record per unit)
            group_df = df_working.drop_duplicates(subset=['analysis_unit_id'], keep='first')[grouping_cols]
            
            # Merge everything together
            df_working = group_df.copy()
            for metric_col, pivot_df in pivot_cols.items():
                df_working = df_working.merge(pivot_df, on='analysis_unit_id', how='left')
            
            # [CHANGE #6] Handle missing values based on imputation_strategy parameter.
            # Previously hardcoded to zero-imputation. Now configurable:
            #   'zero': appropriate for count/sum metrics where no events = zero
            #   'drop': appropriate for metrics where 0 is meaningful (e.g., scores)
            post_col = f"{metric_column}_post_exposure"
            cols_to_impute = [post_col]
            
            if pre_exposure_eval_column is not None:
                pre_col = f"{pre_exposure_eval_column}_pre_exposure"
                cols_to_impute.append(pre_col)
            else:
                pre_col = None
            
            print(f"\nMissing Data Handling (strategy='{imputation_strategy}'):")
            n_before = len(df_working)
            for col in cols_to_impute:
                if col in df_working.columns:
                    n_missing = df_working[col].isna().sum()
                    if n_missing > 0:
                        pct_missing = n_missing / len(df_working) * 100
                        if imputation_strategy == 'zero':
                            df_working[col] = df_working[col].fillna(0)
                            print(f"  {col}: {n_missing} missing values imputed to 0 ({pct_missing:.1f}%)")
                        elif imputation_strategy == 'drop':
                            print(f"  {col}: {n_missing} units with missing values will be dropped ({pct_missing:.1f}%)")
            
            if imputation_strategy == 'drop':
                drop_cols = [c for c in cols_to_impute if c in df_working.columns]
                df_working = df_working.dropna(subset=drop_cols).copy()
                n_dropped = n_before - len(df_working)
                if n_dropped > 0:
                    print(f"  Total units dropped: {n_dropped} ({n_dropped / n_before * 100:.1f}%)")
                    logger.warning(
                        f"Dropped {n_dropped} units due to missing values with "
                        f"imputation_strategy='drop'. This may bias results if "
                        f"missingness is correlated with treatment."
                    )
            
            print(f"\nSuccessfully pivoted to {len(df_working)} unique units")
        else:
            pre_col = pre_exposure_eval_column
            post_col = metric_column
    
    # Apply filtering
    if not include_exclusion_flag:
        if 'analysis_exclusion_flag' in df_working.columns:
            df_working = df_working[df_working['analysis_exclusion_flag'] != True]
    
    if not include_crossover_flag:
        if 'crossover_flag' in df_working.columns:
            df_working = df_working[df_working['crossover_flag'] != True]
    
    # Log filtering info
    original_count = len(df)
    filtered_count = len(df_working)
    print(f"\nRecords before filtering: {original_count}")
    print(f"Records after filtering: {filtered_count}")
    print(f"Records excluded: {original_count - filtered_count}")
    print(f"Include analysis_exclusion_flag: {include_exclusion_flag}")
    print(f"Include crossover_flag: {include_crossover_flag}")
    print()
    
    # ===== WINSORIZATION =====
    # [CHANGE #7] Initialize counters defensively to avoid unbound variable if
    # only one branch executes. Previously n_before_pre could be referenced
    # without assignment if process_pre_exposure was False.
    n_clipped_lower_pre = 0
    n_clipped_upper_pre = 0
    n_clipped_lower_post = 0
    n_clipped_upper_post = 0
    
    winsorization_summary = {
        'applied': False,
        'bounds': None,
        'pre_exposure': {},
        'post_exposure': {},
        'total_winsorized': 0
    }
    
    # [CHANGE #1] Winsorization is now symmetric by default. Previously only
    # clipped the upper tail, which introduced asymmetric bias for metrics with
    # outliers on both ends. winsorize_bounds was normalized from the user's
    # input during validation above.
    if winsorize_bounds is not None:
        lower_bound, upper_bound = winsorize_bounds
        bounds_desc = f"[{lower_bound or 'None'}, {upper_bound or 'None'}]"
        print("=" * 80)
        print(f"WINSORIZATION (bounds: {bounds_desc})")
        print("=" * 80)
        
        def _winsorize_column(data, lower_pct, upper_pct, col_label):
            """Apply symmetric winsorization and return (clipped_data, summary_dict, n_lower, n_upper)."""
            clip_kwargs = {}
            n_lower = 0
            n_upper = 0
            summary = {}
            
            if lower_pct is not None:
                lower_val = data.quantile(lower_pct)
                n_lower = (data < lower_val).sum()
                clip_kwargs['lower'] = lower_val
                summary['lower_threshold'] = lower_val
                summary['lower_clipped'] = int(n_lower)
            
            if upper_pct is not None:
                upper_val = data.quantile(upper_pct)
                n_upper = (data > upper_val).sum()
                clip_kwargs['upper'] = upper_val
                summary['upper_threshold'] = upper_val
                summary['upper_clipped'] = int(n_upper)
            
            clipped = data.clip(**clip_kwargs)
            total = n_lower + n_upper
            summary['total_clipped'] = int(total)
            summary['percentage_clipped'] = total / len(data) * 100
            
            print(f"\n{col_label}:")
            if lower_pct is not None:
                print(f"  Lower threshold ({lower_pct:.2%}ile): {lower_val:.4f} — {n_lower} points clipped")
            if upper_pct is not None:
                print(f"  Upper threshold ({upper_pct:.2%}ile): {upper_val:.4f} — {n_upper} points clipped")
            print(f"  Total clipped: {total} ({total / len(data) * 100:.2f}%)")
            
            return clipped, summary, n_lower, n_upper
        
        # Winsorize pre-exposure if applicable
        if process_pre_exposure and pre_col is not None and pre_col in df_working.columns:
            df_working[pre_col], pre_summary, n_clipped_lower_pre, n_clipped_upper_pre = \
                _winsorize_column(df_working[pre_col], lower_bound, upper_bound, f"Pre-exposure ({pre_col})")
            winsorization_summary['pre_exposure'] = pre_summary
        
        # Winsorize post-exposure
        if post_col in df_working.columns:
            df_working[post_col], post_summary, n_clipped_lower_post, n_clipped_upper_post = \
                _winsorize_column(df_working[post_col], lower_bound, upper_bound, f"Post-exposure ({post_col})")
            winsorization_summary['post_exposure'] = post_summary
        
        total_clipped = (n_clipped_lower_pre + n_clipped_upper_pre + 
                         n_clipped_lower_post + n_clipped_upper_post)
        winsorization_summary['applied'] = True
        winsorization_summary['bounds'] = winsorize_bounds
        winsorization_summary['total_winsorized'] = total_clipped
        
        print(f"\nTotal values winsorized: {total_clipped}")
        print()
        
        title_suffix = f" (Winsorized at {bounds_desc})"
    else:
        title_suffix = ""
    
    # Validate metric type
    # [CHANGE #8] Drop NaN before checking unique values for binary validation.
    # Previously, NaN in the column caused a confusing validation error because
    # np.nan not in [0, 1] evaluates to True (NaN comparisons are always False).
    unique_values = df_working[post_col].dropna().unique()
    if metric_type == 'binary':
        if not all(val in [0, 1] for val in unique_values):
            raise ValueError(
                f"metric_type='binary' requires values to be 0 or 1 only. "
                f"Found values: {sorted(set(unique_values))}"
            )
        n_null_binary = df_working[post_col].isna().sum()
        if n_null_binary > 0:
            logger.warning(
                f"{n_null_binary} NaN values in binary metric '{post_col}' — "
                f"these will be excluded from analysis"
            )
            df_working = df_working[df_working[post_col].notna()].copy()
    
    # Separate data by group
    if process_pre_exposure and pre_col is not None:
        control_pre = df_working[df_working['experiment_group'] == 'Control'][pre_col]
        test_pre = df_working[df_working['experiment_group'] == 'Test'][pre_col]
    
    control_post = df_working[df_working['experiment_group'] == 'Control'][post_col]
    test_post = df_working[df_working['experiment_group'] == 'Test'][post_col]
    
    # ===== PRE-EXPERIMENTAL BIAS TEST =====
    bias_test_results = []
    # [CHANGE #11] bias_detected is now a graduated label instead of a boolean.
    # Previously used the main test alpha as a binary threshold, which caused
    # ~5% false alarm rate on properly randomized experiments.
    bias_detected = False  # still used as boolean for backward compat in viz
    bias_label = "Not tested"
    pre_p_value = np.nan
    has_pre_variation = False
    
    def _classify_bias(p_val, custom_alpha=None):
        """Classify pre-exposure imbalance with graduated thresholds or custom alpha."""
        if custom_alpha is not None:
            # User explicitly set a threshold — use binary classification
            detected = p_val < custom_alpha
            if detected:
                label = f"Imbalance detected (p < {custom_alpha})"
            else:
                label = f"No imbalance (p >= {custom_alpha})"
            return detected, label
        else:
            # Default: graduated reporting — no single binary threshold
            if p_val < 0.01:
                return True, "Strong evidence of imbalance — investigate randomization"
            elif p_val < 0.10:
                return False, "Mild evidence of imbalance — CUPED adjustment recommended"
            else:
                return False, "No evidence of imbalance"
    
    if process_pre_exposure and pre_col is not None:
        print("=" * 80)
        print("PRE-EXPERIMENTAL BIAS TEST")
        print("=" * 80)
        
        # Check pre-exposure variation
        pre_variance = df_working[pre_col].var()
        has_pre_variation = pre_variance > 0
        
        if not has_pre_variation:
            print(f"WARNING: No variation in pre-exposure metric (variance = {pre_variance:.6f})")
            print(f"Pre-experimental bias test skipped - all values are identical.")
            print(f"CUPED adjustment cannot be applied.")
            bias_detected = False
            bias_label = "No variation (test skipped)"
            pre_p_value = np.nan
        elif metric_type == 'continuous':
            pre_t_stat, pre_p_value = ttest_ind(test_pre.dropna(), control_pre.dropna(), equal_var=False)
            bias_detected, bias_label = _classify_bias(pre_p_value, bias_test_alpha)
            
            bias_test_results.append({
                'test_name': 'Pre-Exposure Bias Test',
                'test_type': "Welch's t-Test",
                'test_statistic': pre_t_stat.round(4),
                'p_value': pre_p_value.round(4),
                'bias_detected': bias_detected,
                'bias_label': bias_label
            })
            
            print(f"Test Type: Welch's t-Test (unequal variances)")
            print(f"Test Statistic: {pre_t_stat:.4f}")
            print(f"P-value: {pre_p_value:.4f}")
            print(f"Assessment: {bias_label}")
            
        elif metric_type == 'binary':
            control_pos = (control_pre == 1).sum()
            control_neg = (control_pre == 0).sum()
            test_pos = (test_pre == 1).sum()
            test_neg = (test_pre == 0).sum()
            contingency_table = np.array([[control_pos, control_neg], [test_pos, test_neg]])
            odds_ratio, pre_p_value = stats.fisher_exact(contingency_table)
            bias_detected, bias_label = _classify_bias(pre_p_value, bias_test_alpha)
            
            bias_test_results.append({
                'test_name': 'Pre-Exposure Bias Test',
                'test_type': "Fisher's Exact Test",
                'odds_ratio': round(float(odds_ratio), 4),
                'p_value': round(float(pre_p_value), 4),
                'bias_detected': bias_detected,
                'bias_label': bias_label
            })
            
            print(f"Test Type: Fisher's Exact Test")
            print(f"Odds Ratio: {odds_ratio:.4f}")
            print(f"P-value: {pre_p_value:.4f}")
            print(f"Assessment: {bias_label}")
    
    # ===== CUPED ADJUSTMENT =====
    cuped_actually_applied = False
    theta = None
    covariance_info = None
    
    if apply_cuped and process_pre_exposure and pre_col is not None:
        if not has_pre_variation:
            print("\n" + "=" * 80)
            print("CUPED ADJUSTMENT - SKIPPED")
            print("=" * 80)
            print("WARNING: CUPED cannot be applied because pre-exposure metric has no variation.")
            print("Proceeding with raw post-exposure analysis.")
            post_col_to_test = post_col
        else:
            print("\n" + "=" * 80)
            print("CUPED ADJUSTMENT")
            print("=" * 80)
            
            # Calculate pooled covariance and variance
            all_pre = df_working[pre_col].dropna()
            all_post = df_working[post_col].dropna()
            pre_mean = all_pre.mean()
            
            # Align data for covariance calculation
            valid_mask = df_working[pre_col].notna() & df_working[post_col].notna()
            valid_pre = df_working.loc[valid_mask, pre_col]
            valid_post = df_working.loc[valid_mask, post_col]
            
            # CRITICAL FIX #1: Use ddof=1 for sample covariance (consistent with sample variance)
            cov_matrix = np.cov(valid_pre, valid_post, ddof=1)
            pre_variance_sample = np.var(valid_pre, ddof=1)
            theta = cov_matrix[0, 1] / pre_variance_sample
            
            covariance_info = {
                'covariance_pre_post': cov_matrix[0, 1],
                'variance_pre': pre_variance_sample,
                'correlation_pre_post': cov_matrix[0, 1] / (np.std(valid_pre, ddof=1) * np.std(valid_post, ddof=1)),
                'n_observations_for_covariance': len(valid_pre)
            }
            
            print(f"Covariance (pre, post): {cov_matrix[0, 1]:.6f}")
            print(f"Variance (pre): {pre_variance_sample:.6f}")
            print(f"Correlation (pre, post): {covariance_info['correlation_pre_post']:.6f}")
            print(f"N observations used: {len(valid_pre)}")
            print(f"CUPED Adjustment Factor (θ): {theta:.6f}")
            
            # Create adjusted post-exposure metric
            df_working['post_adjusted'] = df_working[post_col] - theta * (df_working[pre_col] - pre_mean)
            post_col_to_test = 'post_adjusted'
            cuped_actually_applied = True
    else:
        post_col_to_test = post_col
    
    # Update data references after CUPED
    control_post_analysis = df_working[df_working['experiment_group'] == 'Control'][post_col_to_test]
    test_post_analysis = df_working[df_working['experiment_group'] == 'Test'][post_col_to_test]
    
    # ===== POST-EXPERIMENTAL HYPOTHESIS TEST =====
    print("\n" + "=" * 80)
    print("POST-EXPOSURE HYPOTHESIS TEST")
    print("=" * 80)
    
    test_results = []
    
    if metric_type == 'continuous':
        # Post-exposure test (Welch's t-test)
        t_stat, p_value = ttest_ind(test_post_analysis.dropna(), control_post_analysis.dropna(), equal_var=False)
        
        # Calculate means and standard errors (of the analysis column — raw or CUPED-adjusted)
        control_mean_analysis = control_post_analysis.mean()
        test_mean_analysis = test_post_analysis.mean()
        control_se = control_post_analysis.sem()
        test_se = test_post_analysis.sem()
        
        # [CHANGE #2] Always compute raw group means for interpretable effect size reporting.
        # When CUPED is applied, the test is run on the adjusted metric (more power), but
        # the raw means and raw difference are what stakeholders should use for effect size
        # interpretation. The adjusted metric has a different scale.
        if cuped_actually_applied:
            raw_control = df_working[df_working['experiment_group'] == 'Control'][post_col]
            raw_test = df_working[df_working['experiment_group'] == 'Test'][post_col]
            raw_control_mean = raw_control.mean()
            raw_test_mean = raw_test.mean()
            raw_mean_diff = raw_test_mean - raw_control_mean
            # Report raw effect size with CUPED-adjusted SE for the CI
            # (this is the standard industry approach — raw point estimate, adjusted CI)
            mean_diff = raw_mean_diff
            control_mean = raw_control_mean
            test_mean = raw_test_mean
        else:
            raw_control_mean = None
            raw_test_mean = None
            raw_mean_diff = None
            mean_diff = test_mean_analysis - control_mean_analysis
            control_mean = control_mean_analysis
            test_mean = test_mean_analysis
        
        se_diff = np.sqrt(control_se**2 + test_se**2)
        
        # CRITICAL FIX #3: Use Welch-Satterthwaite df for difference CI only
        n1, n2 = len(test_post_analysis), len(control_post_analysis)
        v1, v2 = test_post_analysis.var(ddof=1), control_post_analysis.var(ddof=1)
        df_welch = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        ci_t = stats.t.ppf(1 - alpha/2, df=df_welch)
        
        ci_lower = mean_diff - ci_t * se_diff
        ci_upper = mean_diff + ci_t * se_diff
        
        # [CHANGE #3] Percentage difference CI — add guard for small control means.
        # The delta method approximation used here assumes the control mean is estimated
        # precisely. When |control_mean| < 2 * control_se, the denominator is poorly
        # estimated and the relative CI is unreliable.
        pct_ci_unreliable = False
        if control_mean != 0:
            if abs(control_mean) < 2 * control_se:
                pct_ci_unreliable = True
                logger.warning(
                    f"Control mean ({control_mean:.4f}) is small relative to its SE "
                    f"({control_se:.4f}). Percentage difference CI may be unreliable."
                )
                print(f"  WARNING: Control mean is small relative to SE — percentage CI is unreliable")
            pct_diff = (mean_diff / control_mean) * 100
            se_pct = (se_diff / control_mean) * 100
            ci_pct_lower = pct_diff - ci_t * se_pct
            ci_pct_upper = pct_diff + ci_t * se_pct
        else:
            pct_diff = np.nan
            se_pct = np.nan
            ci_pct_lower = np.nan
            ci_pct_upper = np.nan
        
        # Significance indicator
        if p_value < alpha:
            sig_indicator = "*" if p_value >= 0.01 else ("**" if p_value >= 0.001 else "***")
        else:
            sig_indicator = "ns"
        
        test_results.append({
            'analysis_type': 'Raw' if not cuped_actually_applied else 'CUPED-Adjusted',
            'test_type': "Welch's t-Test",
            'metric_column': metric_column,
            'control_mean': control_mean.round(4),
            'test_mean': test_mean.round(4),
            'mean_diff': mean_diff.round(4),
            'mean_diff_ci_lower': ci_lower.round(4),
            'mean_diff_ci_upper': ci_upper.round(4),
            'pct_diff': pct_diff.round(2) if not np.isnan(pct_diff) else None,
            'pct_diff_ci_lower': ci_pct_lower.round(2) if not np.isnan(ci_pct_lower) else None,
            'pct_diff_ci_upper': ci_pct_upper.round(2) if not np.isnan(ci_pct_upper) else None,
            # [CHANGE #3] Flag when percentage CI is unreliable due to small control mean
            'pct_ci_unreliable': pct_ci_unreliable if control_mean != 0 else True,
            'test_statistic': t_stat.round(4),
            'p_value': p_value.round(4),
            'alpha': alpha,
            'significance': sig_indicator,
            'cuped_applied': cuped_actually_applied,
            'cuped_theta': theta.round(6) if theta is not None else None,
            # [CHANGE #2] Include raw means when CUPED is applied so stakeholders
            # can interpret effect size on the original metric scale
            'raw_control_mean': raw_control_mean.round(4) if raw_control_mean is not None else None,
            'raw_test_mean': raw_test_mean.round(4) if raw_test_mean is not None else None,
            'raw_mean_diff': raw_mean_diff.round(4) if raw_mean_diff is not None else None,
            # [CHANGE #11] bias_label replaces binary bias_detected for graduated reporting
            'pre_bias_detected': bias_detected,
            'pre_bias_label': bias_label,
            'pre_has_variation': has_pre_variation,
            'covariance_pre_post': covariance_info['covariance_pre_post'] if covariance_info else None,
            'correlation_pre_post': covariance_info['correlation_pre_post'] if covariance_info else None
        })
        
    elif metric_type == 'binary':
        control_pos = (control_post_analysis == 1).sum()
        control_neg = (control_post_analysis == 0).sum()
        test_pos = (test_post_analysis == 1).sum()
        test_neg = (test_post_analysis == 0).sum()
        n_control = control_pos + control_neg
        n_test = test_pos + test_neg
        contingency_table = np.array([[control_pos, control_neg], [test_pos, test_neg]])
        
        # [CHANGE #4] Use chi-squared for large samples, Fisher's exact for small.
        # Fisher's is overly conservative for large samples (typical in A/B tests).
        # Fall back to Fisher's when any expected cell count < 5.
        expected = np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) / contingency_table.sum()
        use_fisher = (expected < 5).any()
        
        if use_fisher:
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
            test_type_used = "Fisher's Exact Test (small sample)"
        else:
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table, correction=True)
            odds_ratio = (control_pos * test_neg) / (control_neg * test_pos) if (control_neg * test_pos) > 0 else np.inf
            test_type_used = "Chi-Squared Test (Yates correction)"
        
        control_prop = control_pos / n_control
        test_prop = test_pos / n_test
        
        # Wilson score confidence intervals for individual proportions
        control_ci_lower, control_ci_upper = proportion_confint(control_pos, n_control, alpha=alpha, method='wilson')
        test_ci_lower, test_ci_upper = proportion_confint(test_pos, n_test, alpha=alpha, method='wilson')
        
        prop_diff = test_prop - control_prop
        
        # [CHANGE #4] Newcombe-Wilson CI for difference in proportions (Method 10,
        # Newcombe 1998). Replaces the Wald interval which was inconsistent with
        # Fisher's exact test and has poor coverage for proportions near 0 or 1.
        # This builds the CI for the difference directly from the Wilson intervals
        # of each proportion — already computed above.
        ci_lower = prop_diff - np.sqrt(
            (test_prop - test_ci_lower)**2 + (control_ci_upper - control_prop)**2
        )
        ci_upper = prop_diff + np.sqrt(
            (test_ci_upper - test_prop)**2 + (control_prop - control_ci_lower)**2
        )
        
        if control_prop != 0:
            pct_diff = (prop_diff / control_prop) * 100
            # Use Newcombe CI endpoints for percentage CI
            ci_pct_lower = (ci_lower / control_prop) * 100
            ci_pct_upper = (ci_upper / control_prop) * 100
        else:
            pct_diff = np.nan
            ci_pct_lower = np.nan
            ci_pct_upper = np.nan
        
        if p_value < alpha:
            sig_indicator = "*" if p_value >= 0.01 else ("**" if p_value >= 0.001 else "***")
        else:
            sig_indicator = "ns"
        
        test_results.append({
            'analysis_type': 'Raw' if not cuped_actually_applied else 'CUPED-Adjusted',
            'test_type': test_type_used,
            'metric_column': metric_column,
            'control_proportion': control_prop.round(4),
            'test_proportion': test_prop.round(4),
            'proportion_diff': prop_diff.round(4),
            'prop_diff_ci_lower': ci_lower.round(4),
            'prop_diff_ci_upper': ci_upper.round(4),
            'ci_method': 'Newcombe-Wilson',  # [CHANGE #4] document CI method used
            'pct_diff': pct_diff.round(2) if not np.isnan(pct_diff) else None,
            'pct_diff_ci_lower': ci_pct_lower.round(2) if not np.isnan(ci_pct_lower) else None,
            'pct_diff_ci_upper': ci_pct_upper.round(2) if not np.isnan(ci_pct_upper) else None,
            'odds_ratio': odds_ratio.round(4) if not np.isinf(odds_ratio) else None,
            'p_value': p_value.round(4),
            'alpha': alpha,
            'significance': sig_indicator,
            'cuped_applied': cuped_actually_applied,
            'cuped_theta': theta.round(6) if theta is not None else None,
            # [CHANGE #11] Include graduated bias label
            'pre_bias_detected': bias_detected,
            'pre_bias_label': bias_label,
            'pre_has_variation': has_pre_variation
        })
    
    print()
    
    # ===== VISUALIZATIONS =====
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    palette = {'Control': '#1f77b4', 'Test': '#ff7f0e'}
    
    if cuped_actually_applied:
        # 3 rows x 2 cols: CUPED-Adjusted / Raw Post / Pre
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Row 0: CUPED-Adjusted Post-Exposure
        # KDE Plot - CUPED-Adjusted
        for group in df_working['experiment_group'].unique():
            group_data = df_working[df_working['experiment_group'] == group]['post_adjusted']
            sns.kdeplot(
                data=group_data,
                label=group,
                ax=axes[0, 0],
                color=palette.get(group, None),
                linewidth=2.5,
                fill=True,
                alpha=0.3
            )
        
        axes[0, 0].set_xlabel('Metric Value', fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title(f'Post-Exposure KDE Distribution (CUPED-Adjusted){title_suffix}', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(alpha=0.3)
        p_text_cuped = f"p = {test_results[0]['p_value']:.4f} {test_results[0]['significance']}"
        axes[0, 0].text(0.98, 0.98, p_text_cuped, transform=axes[0, 0].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Violin Plot - CUPED-Adjusted
        sns.violinplot(
            data=df_working,
            x='experiment_group',
            y='post_adjusted',
            ax=axes[0, 1],
            hue='experiment_group',
            palette=palette,
            legend=False
        )
        axes[0, 1].set_xlabel('Experiment Group', fontsize=11)
        axes[0, 1].set_ylabel('Metric Value', fontsize=11)
        axes[0, 1].set_title(f'Post-Exposure Distribution (CUPED-Adjusted){title_suffix}', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='y')
        axes[0, 1].text(0.98, 0.98, p_text_cuped, transform=axes[0, 1].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Row 1: Raw Post-Exposure
        # KDE Plot - Raw Post
        for group in df_working['experiment_group'].unique():
            group_data = df_working[df_working['experiment_group'] == group][post_col]
            sns.kdeplot(
                data=group_data,
                label=group,
                ax=axes[1, 0],
                color=palette.get(group, None),
                linewidth=2.5,
                fill=True,
                alpha=0.3
            )
        
        axes[1, 0].set_xlabel('Metric Value', fontsize=11)
        axes[1, 0].set_ylabel('Density', fontsize=11)
        axes[1, 0].set_title(f'Post-Exposure KDE Distribution (Raw){title_suffix}', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(alpha=0.3)
        p_text_raw = f"Reference (raw data)"
        axes[1, 0].text(0.98, 0.98, p_text_raw, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Violin Plot - Raw Post
        sns.violinplot(
            data=df_working,
            x='experiment_group',
            y=post_col,
            ax=axes[1, 1],
            hue='experiment_group',
            palette=palette,
            legend=False
        )
        axes[1, 1].set_xlabel('Experiment Group', fontsize=11)
        axes[1, 1].set_ylabel('Metric Value', fontsize=11)
        axes[1, 1].set_title(f'Post-Exposure Distribution (Raw){title_suffix}', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        axes[1, 1].text(0.98, 0.98, p_text_raw, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 2: Pre-Exposure
        if process_pre_exposure and pre_col is not None:
            # KDE Plot - Pre
            for group in df_working['experiment_group'].unique():
                group_data = df_working[df_working['experiment_group'] == group][pre_col]
                sns.kdeplot(
                    data=group_data,
                    label=group,
                    ax=axes[2, 0],
                    color=palette.get(group, None),
                    linewidth=2.5,
                    fill=True,
                    alpha=0.3
                )
            
            axes[2, 0].set_xlabel('Metric Value', fontsize=11)
            axes[2, 0].set_ylabel('Density', fontsize=11)
            axes[2, 0].set_title(f'Pre-Exposure KDE Distribution{title_suffix}', fontsize=12, fontweight='bold')
            axes[2, 0].legend(fontsize=10)
            axes[2, 0].grid(alpha=0.3)
            if has_pre_variation:
                # [CHANGE #11] Use graduated bias_label instead of binary indicator
                p_text_pre = f"p = {pre_p_value:.4f}\n{bias_label}"
                bg_color = 'lightcoral' if bias_detected else 'lightgreen'
            else:
                p_text_pre = "No variation (test skipped)"
                bg_color = 'lightgray'
            axes[2, 0].text(0.98, 0.98, p_text_pre, transform=axes[2, 0].transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
            
            # Violin Plot - Pre
            sns.violinplot(
                data=df_working,
                x='experiment_group',
                y=pre_col,
                ax=axes[2, 1],
                hue='experiment_group',
                palette=palette,
                legend=False
            )
            axes[2, 1].set_xlabel('Experiment Group', fontsize=11)
            axes[2, 1].set_ylabel('Metric Value', fontsize=11)
            axes[2, 1].set_title(f'Pre-Exposure Distribution{title_suffix}', fontsize=12, fontweight='bold')
            axes[2, 1].grid(alpha=0.3, axis='y')
            axes[2, 1].text(0.98, 0.98, p_text_pre, transform=axes[2, 1].transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
        
    elif process_pre_exposure and pre_col is not None:
        # 2 rows x 2 cols: Raw Post / Pre (no CUPED)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Row 0: Raw Post-Exposure
        # KDE Plot - Post
        for group in df_working['experiment_group'].unique():
            group_data = df_working[df_working['experiment_group'] == group][post_col]
            sns.kdeplot(
                data=group_data,
                label=group,
                ax=axes[0, 0],
                color=palette.get(group, None),
                linewidth=2.5,
                fill=True,
                alpha=0.3
            )
        
        axes[0, 0].set_xlabel('Metric Value', fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title(f'Post-Exposure KDE Distribution{title_suffix}', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(alpha=0.3)
        p_text_post = f"p = {test_results[0]['p_value']:.4f} {test_results[0]['significance']}"
        axes[0, 0].text(0.98, 0.98, p_text_post, transform=axes[0, 0].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Violin Plot - Post
        sns.violinplot(
            data=df_working,
            x='experiment_group',
            y=post_col,
            ax=axes[0, 1],
            hue='experiment_group',
            palette=palette,
            legend=False
        )
        axes[0, 1].set_xlabel('Experiment Group', fontsize=11)
        axes[0, 1].set_ylabel('Metric Value', fontsize=11)
        axes[0, 1].set_title(f'Post-Exposure Distribution{title_suffix}', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='y')
        axes[0, 1].text(0.98, 0.98, p_text_post, transform=axes[0, 1].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 1: Pre-Exposure
        # KDE Plot - Pre
        for group in df_working['experiment_group'].unique():
            group_data = df_working[df_working['experiment_group'] == group][pre_col]
            sns.kdeplot(
                data=group_data,
                label=group,
                ax=axes[1, 0],
                color=palette.get(group, None),
                linewidth=2.5,
                fill=True,
                alpha=0.3
            )
        
        axes[1, 0].set_xlabel('Metric Value', fontsize=11)
        axes[1, 0].set_ylabel('Density', fontsize=11)
        axes[1, 0].set_title(f'Pre-Exposure KDE Distribution{title_suffix}', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(alpha=0.3)
        if has_pre_variation:
            # [CHANGE #11] Use graduated bias_label instead of binary indicator
            p_text_pre = f"p = {pre_p_value:.4f}\n{bias_label}"
            bg_color = 'lightcoral' if bias_detected else 'lightgreen'
        else:
            p_text_pre = "No variation (test skipped)"
            bg_color = 'lightgray'
        axes[1, 0].text(0.98, 0.98, p_text_pre, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
        
        # Violin Plot - Pre
        sns.violinplot(
            data=df_working,
            x='experiment_group',
            y=pre_col,
            ax=axes[1, 1],
            hue='experiment_group',
            palette=palette,
            legend=False
        )
        axes[1, 1].set_xlabel('Experiment Group', fontsize=11)
        axes[1, 1].set_ylabel('Metric Value', fontsize=11)
        axes[1, 1].set_title(f'Pre-Exposure Distribution{title_suffix}', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        axes[1, 1].text(0.98, 0.98, p_text_pre, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.5))
        
    else:
        # 1 row x 2 cols: Post-exposure only
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # KDE Plot - Post
        for group in df_working['experiment_group'].unique():
            group_data = df_working[df_working['experiment_group'] == group][post_col]
            sns.kdeplot(
                data=group_data,
                label=group,
                ax=axes[0],
                color=palette.get(group, None),
                linewidth=2.5,
                fill=True,
                alpha=0.3
            )
        
        axes[0].set_xlabel('Metric Value', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title(f'Post-Exposure KDE Distribution{title_suffix}', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        p_text_post = f"p = {test_results[0]['p_value']:.4f} {test_results[0]['significance']}"
        axes[0].text(0.98, 0.98, p_text_post, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Violin Plot - Post
        sns.violinplot(
            data=df_working,
            x='experiment_group',
            y=post_col,
            ax=axes[1],
            hue='experiment_group',
            palette=palette,
            legend=False
        )
        axes[1].set_xlabel('Experiment Group', fontsize=11)
        axes[1].set_ylabel('Metric Value', fontsize=11)
        axes[1].set_title(f'Post-Exposure Distribution{title_suffix}', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].text(0.98, 0.98, p_text_post, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # ===== SUMMARY STATISTICS =====
    summary_stats = []
    for group in sorted(df_working['experiment_group'].unique()):
        group_post = df_working[df_working['experiment_group'] == group][post_col]
        
        stats_dict = {
            'experiment_group': group,
            'post_mean': group_post.mean().round(2),
            'post_sd': group_post.std().round(2),
            'post_median': group_post.quantile(0.50).round(2),
            'post_min': group_post.min().round(2),
            'post_p10': group_post.quantile(0.10).round(2),
            'post_p20': group_post.quantile(0.20).round(2),
            'post_p30': group_post.quantile(0.30).round(2),
            'post_p40': group_post.quantile(0.40).round(2),
            'post_p50_median': group_post.quantile(0.50).round(2),
            'post_p60': group_post.quantile(0.60).round(2),
            'post_p70': group_post.quantile(0.70).round(2),
            'post_p80': group_post.quantile(0.80).round(2),
            'post_p90': group_post.quantile(0.90).round(2),
            'post_max': group_post.max().round(2),
            'post_pct_zeros': ((group_post == 0).sum() / len(group_post) * 100).round(2)
        }
        
        # Add pre-exposure stats if available
        if process_pre_exposure and pre_col is not None:
            group_pre = df_working[df_working['experiment_group'] == group][pre_col]
            stats_dict.update({
                'pre_mean': group_pre.mean().round(2),
                'pre_sd': group_pre.std().round(2),
                'pre_median': group_pre.quantile(0.50).round(2),
            })
        
        summary_stats.append(stats_dict)
    
    summary_df = pd.DataFrame(summary_stats)
    test_results_df = pd.DataFrame(test_results)
    
    # Create winsorization summary dataframe
    winsorization_df = pd.DataFrame([winsorization_summary])
    
    return summary_df, test_results_df, fig, winsorization_df
