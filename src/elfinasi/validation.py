"""
Various validation functions here.
"""
import dateutil.parser
from datetime import datetime
from typing import Tuple, List, Union

import pandas as pd
import numpy as np


_time_type = Union[datetime, str]
_time_range_type = List[_time_type]


def validate_time_range(time_range: _time_range_type) -> List[datetime]:
    """
    Validates tries to parse the time_range into datetime objects.
    """
    if time_range is None:
        return None

    assert isinstance(
        time_range, (list, tuple, np.ndarray, pd.DatetimeIndex)
    ), "time_range must be a list, tuple, or np.ndarray."
    assert len(time_range) == 2, "time_range must be a list or a tuple with start and end times."

    time_range_parsed = []

    for t in time_range:
        if isinstance(t, str):
            time_range_parsed.append(dateutil.parser.parse(t))
        elif isinstance(t, (int, float)):
            raise ValueError(f'Unknown time format, {t}')
        elif isinstance(t, (np.datetime64)):
            time_range_parsed.append(dateutil.parser.parse(str(t)))
        else:
            time_range_parsed.append(t)

    for t in time_range_parsed:
        _validate_year(t)

    time_range_parsed.sort()
    return time_range_parsed


def _validate_year(time):
    """
    Provides a sanity check that the year is after 2000 and before the current year + 1.
    """
    year = time.year
    if year < 2019:
        raise ValueError(f'The passed year={year} must be greater than 2018.')
    elif year > 2022:
        raise ValueError(f'The passed year={year} must be less 2023.')
    return