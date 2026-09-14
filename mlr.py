"""This module contains functions to perform multiple linear regression for HW 1 in AE 498 Computational Systems Engineering.

"""

import numpy as np
import scipy


def model_fit(X, y):
    """Function to fit a multiple linear regression model to a given dataset using ordinary least squares estimation.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.
    y : array_like
        Output/response data (n elements), where each element is an observation.

    Returns
    -------
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y_predicted : array_like
        Predicted responses for input data (n elements).

    """

    # YOUR CODE HERE
    X = np.array(X)
    y = np.array(y)

    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    y_predicted = X @ coefficients

    return coefficients, y_predicted


def anova(y, y_predicted, p):
    """Function to perform an ANOVA F-test for a multiple linear regression model.

    Parameters
    ----------
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).
    p : int
        Number of predictors.

    Returns
    -------
    p_value : float
        P-value for model F-statistic.
    f_statistic : float
        F-statistic for model F-test.

    """

    # YOUR CODE HERE
    y = np.asarray(y)
    y_predicted = np.asarray(y_predicted)

    n = len(y)

    # Mean of observed responses
    y_mean = np.mean(y)

    # Regression sum of squares
    ssr = np.sum((y_predicted - y_mean) ** 2)

    # Error sum of squares
    sse = np.sum((y - y_predicted) ** 2)

    # Degrees of freedom
    df_regression = p
    df_error = n - p - 1

    # Mean squares
    msr = ssr / df_regression
    mse = sse / df_error

    # F-statistic
    f_statistic = msr / mse

    # Right-tail p-value for F-test
    p_value = scipy.stats.f.sf(f_statistic, df_regression, df_error)

    return p_value, f_statistic

def coefficient_tests(X, coefficients, y, y_predicted):
    """Function to perform hypothesis t-tests for coefficients of a multiple linear regression model.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).

    Returns
    -------
    p_values : list_like
        P-values for coefficient t-statistics.
    t_statistics : list_like
        T-statistics for coefficient t-tests.

    """

    # YOUR CODE HERE
    X = np.asarray(X)
    coefficients = np.asarray(coefficients)
    y = np.asarray(y)
    y_predicted = np.asarray(y_predicted)

    n, k = X.shape

    # k is number of coefficients, including intercept
    # error degrees of freedom = n - k
    df_error = n - k

    # Error sum of squares
    sse = np.sum((y - y_predicted) ** 2)

    # Mean squared error
    mse = sse / df_error

    # Inverse of X^T X
    xtx_inv = np.linalg.inv(X.T @ X)

    # Standard errors of coefficients
    standard_errors = np.sqrt(mse * np.diag(xtx_inv))

    # t-statistics
    t_statistics = coefficients / standard_errors

    # Two-sided p-values
    p_values = 2 * scipy.stats.t.sf(np.abs(t_statistics), df_error)

    return p_values, t_statistics

def normalize_data(X):
    """Function to normalize input data to [-1, 1].

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. The first column should be all ones.

    Returns
    -------
    X_normalized : array_like
        Input data (n x (p + 1) dimensions) normalized to the range [-1, 1].

    """

    n, n_columns = np.shape(X)

    # normalize each column
    X_normalized = np.ones((n, n_columns))
    for column in range(n_columns):
        x_column = X[:, column]
        x_max = max(x_column)
        x_min = min(x_column)
        if x_max != x_min:
            x_column_normalized = [
                (x - (x_max + x_min) / 2) / ((x_max - x_min) / 2) for x in x_column
            ]
            X_normalized[:, column] = x_column_normalized

    return X_normalized
