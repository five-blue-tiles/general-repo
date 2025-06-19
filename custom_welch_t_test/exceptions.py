"""
Custom exceptions for the Welch's t-test library.
"""

class WelchTTestError(Exception):
    """
    Base exception class for Welch's t-test library.
    """
    def __init__(self, message="An error occurred in Welch's t-test calculation"):
        self.message = message
        super().__init__(self.message)

class InvalidDataError(WelchTTestError):
    """
    Exception raised when input data is invalid or insufficient.
    
    This includes cases where:
    - DataFrame is empty or invalid
    - Groups have insufficient data
    - Independent variable has more than 2 distinct values
    - control_test_values format is incorrect
    """
    def __init__(self, message="Invalid data provided for t-test"):
        super().__init__(message)

class InvalidColumnError(WelchTTestError):
    """
    Exception raised when specified columns don't exist in the DataFrame.
    """
    def __init__(self, message="Column not found in DataFrame"):
        super().__init__(message)

class InsufficientDataError(InvalidDataError):
    """
    Exception raised when there's insufficient data for statistical analysis.
    
    This includes cases where:
    - Sample size is too small
    - All values are NaN
    - Groups are empty after filtering
    """
    def __init__(self, message="Insufficient data for statistical analysis"):
        super().__init__(message)

class StatisticalError(WelchTTestError):
    """
    Exception raised when statistical calculations fail.
    
    This includes cases where:
    - Division by zero in variance calculations
    - Invalid statistical computations
    - Numerical instability
    """
    def __init__(self, message="Statistical calculation error"):
        super().__init__(message)