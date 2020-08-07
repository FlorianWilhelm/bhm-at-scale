from typing import List, Iterator, Optional

import numpy as np
import pandas as pd


def gen_splits(size: int, lower: int, upper: int) -> Optional[Iterator[int]]:
    """Generate random numbers from lower to upper that sum up to <= size

    Args:
        size:
        lower:
        upper:

    Returns:

    """
    if size < lower:
        return
    elif size < upper:
        yield size
    else:
        split = np.random.randint(lower, upper)
        yield split
        yield from gen_splits(size - split, lower, upper)


def gen_partitions(size: int, lower: int, upper: int):
    splits = gen_splits(size, lower, upper)
    partitions = np.hstack([np.repeat(i, s) for i, s in enumerate(splits)])
    if (delta := size - partitions.shape[0]) > 0:
        assert delta < lower, 'Splitting violation!'
        return np.hstack([partitions, np.repeat(-1, delta)])
    else:
        return partitions


def encode(df: pd.DataFrame,
           cols: List[str],
           drop_first: bool = True) -> pd.DataFrame:
    """Do a dummy encoding for the columsn specified

    Args:
        df: DataFrame
        cols: List of columns to perform dummy encoding on
        drop_first: parameter for dummy encoding
    """
    dfs = []
    for col in df.columns:
        ds = df[col]
        if col not in cols:
            dfs.append(ds.to_frame())
        else:
            dfs.append(pd.get_dummies(ds, prefix=col, drop_first=drop_first))
    return pd.concat(dfs, axis=1)


def make_cube(df: pd.DataFrame, idx_cols: List[str]) -> np.ndarray:
    """Make an array cube from a Dataframe

    Args:
        df: Dataframe
        idx_cols: columns defining the dimensions of the cube

    Returns:
        multi-dimensional array
    """
    assert len(set(idx_cols) & set(df.columns)) == len(idx_cols), 'idx_cols must be subset of columns'

    df = df.set_index(keys=idx_cols)  # don't overwrite a parameter, thus copy!
    idx_dims = [level.max() + 1 for level in df.index.levels]
    idx_dims.append(len(df.columns))

    cube = np.empty(idx_dims)
    cube.fill(np.nan)
    cube[tuple(np.array(df.index.to_list()).T)] = df.values

    return cube
