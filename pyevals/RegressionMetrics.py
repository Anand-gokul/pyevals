from pyevals.utils import *
import pandas as pd
from sklearn.metrics import f1_score
from pyevals.exceptions import *


def rsquared_score(actual, predicted):
    """
    :Params:
    --------
    actual: Input your y_test
    predicted: Input your y_pred
    :Returns:
    ---------
    returns your root mean square error
    Examples
    --------
    >>> actual = [3, 0.5, 2, 7]
    >>> predicted = [2.5, 0.3, 2, 8]
    >>> rsquared_score(actual, predicted)
    0.93530997
    """
    # Declaring the exceptions.

    if len(actual) == 0:
        raise PyEvalsValueError('There are currently no values in actual.')
    if len(predicted) == 0:
        raise PyEvalsValueError('There are currently no values in predicted.')
    # for i in range(len(actual)):
    #     if (actual[i] < 0).any() or (predicted[i] < 0)():
    #         raise PyEvalsValueError('There cannot be any negative values to calculate r2_score')

    actual = check_type(actual)
    predicted = check_type(predicted)
    type_actual, actual, predicted = input_regression(actual, predicted)
    equal_lengths(actual, predicted)
    actual, predicted = np.array(actual), np.array(predicted)
    sse = ((actual - predicted) ** 2).sum(dtype=float)
    sse_nonzero = sse != 0  # To check for only non-zero values as input
    sst = ((actual - np.average(actual)) ** 2).sum(dtype=float)
    sst_nonzero = sst != 0  # To check for only non-zero values as input
    nonzero_values = sse_nonzero and sst_nonzero
    output = 1 - (sse[nonzero_values] / sst[nonzero_values])

    # Condition if either the values are equal to zero, it returns 1.0 else returns the actual score.
    if not np.any(sse_nonzero):
        if not np.any(sst_nonzero):
            return 1.0
    return output


def rmse(actual, predicted):
    """
    :Params:
    --------
    actual: Input your y_test
    predicted: Input your y_pred
    :Returns:
    ---------
    returns your root mean square error
    Examples
    --------
    >>>actual = [1, 0.5, 2, 7]
    >>>predicted = [2, 0.8, 1, 6]
    >>>print(rmse(actual, predicted))
    0.8789197915623472
    """
    # Exceptions
    if len(actual) == 0:
        raise PyEvalsValueError('There are currently no values in actual.')
    if len(predicted) == 0:
        raise PyEvalsValueError('There are currently no values in predicted.')
    actual = check_type(actual)
    predicted = check_type(predicted)
    actual, predicted = np.array(actual), np.array(predicted)
    type_actual, actual, predicted = input_regression(actual, predicted)
    equal_lengths(actual, predicted)
    output = np.average((simple_error(actual, predicted)) ** 2)
    return np.sqrt(output)


def mean_absolute_percentage_error(actual, predicted):
    """
    :Params:
    --------
    actual: Input your y_test
    predicted: Input your y_pred
    :Returns:
    ---------
    returns your root mean square error
    :Examples
    --------
    >>>actual = [[0.5, 1], [-1, 1], [7, -6]]
    >>>predicted = [[0, 2], [-1, 2], [8, -5]]
    55.15873015873016
    """

    if len(actual) == 0 or len(predicted) == 0:
        raise PyEvalsValueError('There are currently no values in your actual or predicted')

    actual = check_type(actual)
    predicted = check_type(predicted)
    type_actual, actual, predicted = input_regression(actual, predicted)
    equal_lengths(actual, predicted)
    actual, predicted = np.array(actual), np.array(predicted)
    output = np.mean(np.abs(simple_error(actual, predicted) / actual) * 100)
    return output


def adj_rsquared_score(X_train, actual, predicted, rsquared_score):
    """
    :Params:
    --------
    X_train = Your independant variables train data
    actual = Your y_test
    predicted = your y_pred
    rsquared_Score = R2 score.
    :Returns:
    ---------
    Return the Adjusted R squared score/ value
    :Examples:
    ----------
    >>>actual = [3, 0.5, 2, 7]
    >>>predicted = [2.5, 0.3, 2, 8]
    >>>print(adj_rsquared_score(actual, predicted))
    0.94436658
    """

    if len(actual) == 0:
        raise PyEvalsValueError('There are currently no values in actual.')
    if len(predicted) == 0:
        raise PyEvalsValueError('There are currently no values in predicted.')
    for i in range(len(actual)):
        if (actual[i] < 0) or (predicted[i] < 0):
            raise PyEvalsValueError('There cannot be any negative values to calculate r2_score')

    actual = check_type(actual)
    predicted = check_type(predicted)
    type_actual, actual, predicted = input_regression(actual, predicted)
    equal_lengths(actual, predicted)
    actual, predicted = np.array(actual), np.array(predicted)
    df = pd.DataFrame(X_train)
    n = len(X_train)
    p = len(df.columns)
    output = (1 - ((1 - rsquared_score(actual, predicted)) * (n - 1) / (n - p - 1)))
    return output


def f1(actual, predicted):
    """
    :Params:
    --------
    actual = Your y_test
    predicted = your y_pred
    n = Number of samples/ sample size
    p = Number of independant variables
    :Returns:
    ---------
    Return the Adjusted R squared score/ value
    :Examples:
    ----------
    >>>actual = [3, 0.5, 2, 7]
    >>>predicted = [2.5, 0.3, 2, 8]
    >>>print(f1(actual, predicted))
    0.94436658
    """

    if len(actual) == 0:
        raise PyEvalsValueError('There are currently no values in actual.')
    if len(predicted) == 0:
        raise PyEvalsValueError('There are currently no values in predicted.')
    for i in range(len(actual)):
        if (actual[i] < 0) or (predicted[i] < 0):
            raise PyEvalsValueError('There cannot be any negative values to calculate r2_score')

    actual = check_type(actual)
    predicted = check_type(predicted)
    type_actual, actual, predicted = input_regression(actual, predicted)
    equal_lengths(actual, predicted)
    actual, predicted = np.array(actual), np.array(predicted)
    output = f1_score(actual, predicted)
    return output
