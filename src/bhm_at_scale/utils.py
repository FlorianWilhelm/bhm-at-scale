from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
from jax import ops
import numpy as np
import pandas as pd


def summary(samples: Dict[str, jnp.DeviceArray], poisson: bool) -> Dict[str, Dict[str, jnp.DeviceArray]]:
    """Generate a nice summary given the samples from `get_samples` of the Poisson model

    Args:
        samples: samples as acquired from `predictive`.get_samples
        poisson: exponentiate the values if Poisson was used

    Returns:
        dict: summary over the samples
    """
    site_stats = {}
    for k, v in samples.items():
        if poisson:  # Poisson model, thus we exponentiate!
            v = jnp.exp(v)
        else:
            v = v.astype(jnp.float32)  # for percentile to work
        v = ops.index_update(v, v < 0., 0.)  # avoid -1.
        site_stats[k] = {
            "mean": jnp.mean(v, 0),
            "std": jnp.std(v, 0),
            "5%": jnp.percentile(v, 5., axis=0),
            "25%": jnp.percentile(v, 25., axis=0),
            "75%": jnp.percentile(v, 75., axis=0),
            "95%": jnp.percentile(v, 95., axis=0),
        }
    return site_stats


def make_intervals(arr: np.ndarray, offset: int = 0) -> List[Tuple[int, int]]:
    """Creates a list of (start, stop)-indices from a boolean array

    Example:
        list(make_intervals([True, True, False, False, True, True, True]))
        > [(0, 2), (4, 7)]

    So the first True interval is from index 0 to 2 and the second from 4 to 7.

    Args:
        arr: numpy array of boolean
        offset: default argument for recursion, keep to 0

    Returns:
        list of interval tuples
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    arr = arr.astype(np.int)
    if arr.size > 0:
        start_idx = np.nonzero(arr)[0]
        if start_idx.size > 0:
            start_idx = start_idx[0]
        else:
            return
        span = np.nonzero(~arr[start_idx:].astype(np.bool))[0]
        if span.size > 0:
            span = span[0]
        else:
            span = arr[start_idx:].size
        end_idx = start_idx + span
        yield offset + start_idx, offset + end_idx
        offset += start_idx + span
        yield from make_intervals(arr[end_idx:], offset)


def stats_to_df(stats: Dict[str, Dict[str, jnp.DeviceArray]], col_names: List[str]) -> pd.DataFrame:
    """Transforms the stats of `summary` above to a proper dataframe

    Args:
        stats: stats from the result of `summary`
        col_names (list): column names for the results

    Returns:
        nice dataframe
    """
    dfs = []
    for site_name, site in stats.items():
        for stat_name, values in site.items():
            dims = len(values.shape)
            values = jnp.atleast_2d(values)
            stat_df = pd.DataFrame(values, columns=col_names)
            idx = jnp.arange(values.shape[0]) if dims == 2 else -1
            stat_df.insert(0, column='metric', value=stat_name)
            stat_df.insert(0, column='site', value=site_name)
            stat_df.insert(0, column='StoreId', value=idx)
            dfs.append(stat_df)
    return pd.concat(dfs)


def preds_to_df(preds: Dict[str, Dict[str, jnp.DeviceArray]]) -> pd.DataFrame:
    """Transforms the predictions after a `summary` call above to a proper dataframe

    Args:
        preds: predictions from the result of `summary`

    Returns:
        nice dataframe
    """
    obs = preds['obs']
    dfs = []
    for stat_name, values in obs.items():
        dims = len(values.shape)
        values = jnp.atleast_2d(values)
        stat_df = pd.DataFrame(values, columns=[f"ts_{t}" for t in np.arange(values.shape[-1])])
        idx = jnp.arange(values.shape[0]) if dims == 2 else -1
        stat_df.insert(0, column='metric', value=stat_name)
        stat_df.insert(0, column='StoreId', value=idx)
        dfs.append(stat_df)
    return pd.concat(dfs)


def reorder_cols(df: pd.DataFrame,
                 *,
                 first: Union[str, List[str]],
                 last: Union[str, List[str]]) -> pd.DataFrame:
    """Reorder the columns having `first` first and `last` last

    Args:
        df: Dataframe
        first: columns that should be first
        last: columns that should be last
    """
    if isinstance(first, (str, int)):
        first = [first]
    if isinstance(last, (str, int)):
        last = [last]
    other_cols = [col for col in df.columns if col not in first + last]
    return df.reindex(columns=first + other_cols + last)
