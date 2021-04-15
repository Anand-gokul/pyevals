import numpy as np
import numbers
from scipy.sparse import csr_matrix
from pyevals.exceptions import *


def simple_error(actual: np.ndarray, predicted: np.ndarray):
    """

    :param actual:
    :param predicted:
    :return:
    """
    return actual - predicted


def check_type(array, dtype="numeric"):
    original_dtype = getattr(array, "dtype", None)
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    if dtype_numeric is True:
        if original_dtype is not None and original_dtype.kind == "O":
            dtype = np.float64
    else:
        dtype = None

    return array


def check_complex(array):
    if hasattr(array, "dtype") and array.dtype is not None and hasattr(array.dtype, "kind") and array.dtype.kind == 'c':
        return True


def samples(input_array_list):
    """
    Params:
    -------
    input_array_list: The input array/ list.
    Returns:
    --------
        Returns the length of the array or list after converting the list to array and also after checking the shape.
    """
    if not hasattr(input_array_list, '__len__') and not hasattr(input_array_list, 'shape'):
        if hasattr(input_array_list, '__array__'):
            input_array_list = np.asarray(input_array_list)
        else:
            raise PyEvalsTypeError('Expected a sequence or an array like input %s' % type(input_array_list))
    if hasattr(input_array_list, 'shape') and input_array_list.shape is not None:
        if len(input_array_list.shape) == 0:
            raise PyEvalsTypeError('The input array %r cannot be considered'
                                  % input_array_list)
        if isinstance(input_array_list.shape[0], numbers.Integral):
            return input_array_list.shape[0]
    try:
        return len(input_array_list)
    except TypeError as ex:
        raise PyEvalsTypeError('Expected a sequence or an array like input %s' % type(input_array_list)) from ex


def equal_lengths(*arrays):
    length = [samples(val) for val in arrays if val is not None]
    unique = np.unique(length)
    if len(unique) > 1:
        raise PyEvalsValueError("The lengths of input variables are not equal %r" % [int(val) for val in length])


def input_regression(actual, predicted, dtype='numeric'):
    """

    :param actual:
    :param predicted:
    :param dtype:
    :return:
    """
    equal_lengths(actual, predicted)
    actual = check_type(actual, dtype=dtype)
    predicted = check_type(predicted, dtype=dtype)
    actual, predicted = np.array(actual), np.array(predicted)

    if actual.ndim == 1:
        actual = actual.reshape((-1, 1))
    if predicted.ndim == 1:
        predicted = predicted.reshape((-1, 1))

    if actual.shape[1] != predicted.shape[1]:
        raise PyEvalsValueError("Actual {0} and predicted {1} values have different shapes.".
                                format(actual.shape, predicted.shape))
    outputs = actual.shape[1]
    type_actual = 'continuous' if outputs == 1 else 'continuous-multioutput'

    return type_actual, actual, predicted


def check_targets(actual, predicted):
    """

    :param actual:
    :param predicted:
    :return:
    """
    equal_lengths(actual, predicted)
    type_true = check_type(actual)
    type_pred = check_type(predicted)
    y_type = {type_true, type_pred}
    if y_type == {"binary", "multiclass"}:
        y_type = {"multiclass"}

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if y_type not in ["binary", "multiclass", "multilabel-indicator"]:
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        if y_type == "binary":
            try:
                unique_values = np.union1d(actual, predicted)
            except TypeError as e:
                # We expect actual and predicted to be of the same data type.
                # If `actual` was provided to the classifier as strings,
                # `predicted` given by the classifier will also be encoded with
                # strings. So we raise a meaningful error
                raise TypeError(
                    f"Labels in actual and predicted should be of the same type. "
                    f"Got actual={np.unique(actual)} and "
                    f"predicted={np.unique(predicted)}. Make sure that the "
                    f"predictions provided by the classifier coincides with "
                    f"the true labels."
                ) from e
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        actual = csr_matrix(actual)
        predicted = csr_matrix(predicted)
        y_type = 'multilabel-indicator'

    return y_type, actual, predicted
