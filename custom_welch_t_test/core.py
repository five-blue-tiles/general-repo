"""
Core functionality for Welch's t-test analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from .exceptions import WelchTTestError, InvalidDataError, InvalidColumnError
from .utils import validate_inputs, extract_groups

def welch_t_test(dataframe, independent_var_col, dependent_var_col, control_test_values):
    """
    Performs Welch's 2-sided t-test on data from a dataframe.
    
    Parameters:
    dataframe (pd.DataFrame): Input dataframe
    independent_var_col (str): Column name of the independent variable
    dependent_var_col (str): Column name of the dependent variable
    control_test_values (list): Array with two values [control_value, test_value] 
                               to identify control and test groups
    
    Returns:
    dict: Dictionary containing test results
    
    Raises:
    InvalidColumnError: When specified columns don't exist in dataframe
    InvalidDataError: When data validation fails
    WelchTTestError: When t-test calculation fails
    """
    try:
        # Validate inputs
        validate_inputs(dataframe, independent_var_col, dependent_var_col, control_test_values)
        
        # Extract control and test groups
        control_vec, test_vec, control_value, test_value = extract_groups(
            dataframe, independent_var_col, dependent_var_col, control_test_values
        )
        
        # Perform Welch's t-test (assumes unequal variances)
        t_statistic, p_value = ttest_ind(control_vec, test_vec, equal_var=False)
        
        # Calculate additional statistics
        mean_control = np.mean(control_vec)
        mean_test = np.mean(test_vec)
        std_control = np.std(control_vec, ddof=1)
        std_test = np.std(test_vec, ddof=1)
        n_control = len(control_vec)
        n_test = len(test_vec)
        
        # Calculate degrees of freedom for Welch's t-test
        s1_sq = std_control**2
        s2_sq = std_test**2
        df = (s1_sq/n_control + s2_sq/n_test)**2 / (
            (s1_sq/n_control)**2/(n_control-1) + (s2_sq/n_test)**2/(n_test-1)
        )
        
        results = {
            't_statistic': t_statistic,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'mean_control': mean_control,
            'mean_test': mean_test,
            'std_control': std_control,
            'std_test': std_test,
            'n_control': n_control,
            'n_test': n_test,
            'mean_difference': mean_test - mean_control,
            'control_value': control_value,
            'test_value': test_value
        }
        
        return results
        
    except (InvalidColumnError, InvalidDataError) as e:
        raise e
    except Exception as e:
        raise WelchTTestError(f"T-test calculation failed: {str(e)}") from e