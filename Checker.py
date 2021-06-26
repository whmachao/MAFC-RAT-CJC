import utilities.Constants as Constants
import numpy as np

def check_BDT_SPLIT_RATIO_LIST():
    variable = Constants.BDT_SPLIT_RATIO_LIST
    print(type(variable))
    if not isinstance(variable, np.ndarray):
        raise ValueError(str(variable) + ' is not a LIST!')
    if len(set(variable)) != len(variable):
        raise ValueError(str(variable) + ' has duplicated elements!')
    if min(variable) <= 0.0:
        raise ValueError(str(variable) + ' has negative element!')
    if max(variable) >= 1.0:
        raise ValueError(str(variable) + ' has element greater than 1.0!')
    return variable