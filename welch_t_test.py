import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

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
    dict: Dictionary containing test results, or error message
    """
    try:
        # Check if columns exist
        if independent_var_col not in dataframe.columns:
            return {"error": f"Column '{independent_var_col}' not found in dataframe"}
        if dependent_var_col not in dataframe.columns:
            return {"error": f"Column '{dependent_var_col}' not found in dataframe"}
        
        # Check number of distinct values in independent variable
        distinct_values = dataframe[independent_var_col].nunique()
        if distinct_values > 2:
            return {"error": f"Independent variable column has {distinct_values} distinct values. Only 2 are allowed for t-test."}
        
        # Check if control_test_values has exactly 2 elements
        if len(control_test_values) != 2:
            return {"error": "control_test_values must contain exactly 2 values [control_value, test_value]"}
        
        control_value, test_value = control_test_values
        
        # Filter data for control and test groups
        control_data = dataframe[dataframe[independent_var_col] == control_value][dependent_var_col]
        test_data = dataframe[dataframe[independent_var_col] == test_value][dependent_var_col]
        
        # Check if both groups have data
        if len(control_data) == 0:
            return {"error": f"No data found for control value '{control_value}'"}
        if len(test_data) == 0:
            return {"error": f"No data found for test value '{test_value}'"}
        
        # Convert to numpy arrays and remove NaN values
        control_vec = control_data.dropna().values
        test_vec = test_data.dropna().values
        
        if len(control_vec) == 0:
            return {"error": "No valid (non-NaN) data in control group"}
        if len(test_vec) == 0:
            return {"error": "No valid (non-NaN) data in test group"}
        
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
        df = (s1_sq/n_control + s2_sq/n_test)**2 / ((s1_sq/n_control)**2/(n_control-1) + (s2_sq/n_test)**2/(n_test-1))
        
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
        
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Create sample dataframe
    data = {
        'group': ['control', 'control', 'control', 'control', 'test', 'test', 'test', 'test'],
        'value': [12.5, 13.2, 11.8, 14.1, 15.2, 14.8, 16.1, 15.7]
    }
    df = pd.DataFrame(data)
    
    # Perform test
    results = welch_t_test(df, 'group', 'value', ['control', 'test'])
    
    # Print results
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("Welch's 2-sided t-test Results:")
        print(f"t-statistic: {results['t_statistic']:.4f}")
        print(f"p-value: {results['p_value']:.4f}")
        print(f"Degrees of freedom: {results['degrees_of_freedom']:.2f}")
        print(f"Mean difference: {results['mean_difference']:.4f}")
        print(f"Control mean: {results['mean_control']:.4f} (n={results['n_control']})")
        print(f"Test mean: {results['mean_test']:.4f} (n={results['n_test']})")