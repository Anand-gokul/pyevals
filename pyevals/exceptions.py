#!usr/bin/env python3.8
# Exceptions for the project.

class PyEvalsRootError(Exception):
    """
        Main exception for all the forth coming errors.
    """
    pass


class PyEvalsValueError(PyEvalsRootError, ValueError):
    """
        A bad value or no value has been passed into the function
    """
    pass


class PyEvalsTypeError(PyEvalsRootError, TypeError):
    """
        Raise this error when there is an unexpected type or none type passed as input.
    """
    pass


class PyEvalsKeyError(PyEvalsRootError, KeyError):
    """
        Raise this error when there is an none or any value found that the user tried to initiate.
    """


class PyEvalsInterruptError(PyEvalsRootError, KeyboardInterrupt):
    """
        Raise this exception when user interrupts.
    """
