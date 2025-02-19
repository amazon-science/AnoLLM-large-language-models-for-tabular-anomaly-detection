import typing as tp
import numpy as np
import pandas as pd

def _array_to_dataframe(
    data: tp.Union[pd.DataFrame, np.ndarray], columns=None
) -> pd.DataFrame:
    """Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(
        data, np.ndarray
    ), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert (
        columns
    ), "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(
        data[0]
    ), "%d column names are given, but array has %d columns!" % (
        len(columns),
        len(data[0]),
    )

    return pd.DataFrame(data=data, columns=columns)

