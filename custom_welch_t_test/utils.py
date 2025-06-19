"""
Utility functions for data validation and processing.
"""

import pandas as pd
import numpy as np
from .exceptions import InvalidColumnError, InvalidDataError

def validate_inputs(dataframe, independent_var_col, dependent_var_col, control_test_values):
    """
    Validates input parameters for the t-test.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    independent_var_col (str): Column name of the independent variable
    dependent_var_col (str): Column name of the dependent variable
    control_test_values (list): Array with control and test values
    
    Raises:
    InvalidColumnError: When columns don't exist
    InvalidDataError: When data validation fails
    """
    # Check if dataframe is valid
    if not isinstance(dataframe, pd.DataFrame):
        raise InvalidDataError("Input must be a pandas DataFrame")
    
    if dataframe.empty:
        raise InvalidDataError("DataFrame is empty")
    
    # Check if columns exist
    if independent_var_col not in dataframe.columns:
        raise InvalidColumnError(f"Column '{independent_var_col}' not found in dataframe")
    
    if dependent_var_col not in dataframe.columns:
        raise InvalidColumnError(f"Column '{dependent_var_col}' not found in dataframe")
    
    # Check number of distinct values in independent variable
    distinct_values = dataframe[independent_var_col].nunique()
    if distinct_values > 2:
        raise InvalidDataError(
            f"Independent variable column has {distinct_values} distinct values. "
            "Only 2 are allowed for t-test."
        )
    
    # Check if control_test_values has exactly 2 elements
    if not isinstance(control_test_values, (list, tuple, np.ndarray)):
        raise InvalidDataError("control_test_values must be a list, tuple, or array")
    
    if len(control_test_values) != 2:
        raise InvalidDataError(
            "control_test_values must contain exactly 2 values [control_value, test_value]"
        )

def extract_groups(dataframe, independent_var_col, dependent_var_col, control_test_values):
    """
    Extracts control and test groups from the dataframe.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    independent_var_col (str): Column name of the independent variable
    dependent_var_col (str): Column name of the dependent variable
    control_test_values (list): Array with control and test values
    
    Returns:
    tuple: (control_vec, test_vec, control_value, test_value)
    
    Raises:
    InvalidDataError: When groups cannot be extracted or are empty
    """
    control_value, test_value = control_test_values
    
    # Filter data for control and test groups
    control_data = dataframe[dataframe[independent_var_col] == control_value][dependent_var_col]
    test_data = dataframe[dataframe[independent_var_col] == test_value][dependent_var_col]
    
    # Check if both groups have data
    if len(control_data) == 0:
        raise InvalidDataError(f"No data found for control value '{control_value}'")
    
    if len(test_data) == 0:
        raise InvalidDataError(f"No data found for test value '{test_value}'")
    
    # Convert to numpy arrays and remove NaN values
    control_vec = control_data.dropna().values
    test_vec = test_data.dropna().values
    
    # Check for valid data after removing NaN values
    if len(control_vec) == 0:
        raise InvalidDataError("No valid (non-NaN) data in control group")
    
    if len(test_vec) == 0:
        raise InvalidDataError("No valid (non-NaN) data in test group")
    
    # Check for minimum sample size
    if len(control_vec) < 2:
        raise InvalidDataError("Control group must have at least 2 observations")
    
    if len(test_vec) < 2:
        raise InvalidDataError("Test group must have at least 2 observations")
    
    return control_vec, test_vec, control_value, test_value

def format_results(results):
    """
    Formats t-test results for display.
    
    Parameters:
    results (dict): T-test results dictionary
    
    Returns:
    str: Formatted results string
    """
    if not isinstance(results, dict):
        return "Invalid results format"
    
    output = []
    output.append("Welch's 2-sided t-test Results:")
    output.append("=" * 35)
    output.append(f"t-statistic: {results.get('t_statistic', 'N/A'):.4f}")
    output.append(f"p-value: {results.get('p_value', 'N/A'):.4f}")
    output.append(f"Degrees of freedom: {results.get('degrees_of_freedom', 'N/A'):.2f}")
    output.append(f"Mean difference: {results.get('mean_difference', 'N/A'):.4f}")
    output.append("")
    output.append("Group Statistics:")
    output.append(f"Control ({results.get('control_value', 'N/A')}):")
    output.append(f"  Mean: {results.get('mean_control', 'N/A'):.4f}")
    output.append(f"  Std: {results.get('std_control', 'N/A'):.4f}")
    output.append(f"  N: {results.get('n_control', 'N/A')}")
    output.append(f"Test ({results.get('test_value', 'N/A')}):")
    output.append(f"  Mean: {results.get('mean_test', 'N/A'):.4f}")
    output.append(f"  Std: {results.get('std_test', 'N/A'):.4f}")
    output.append(f"  N: {results.get('n_test', 'N/A')}")
    
    return "\n".join(output)