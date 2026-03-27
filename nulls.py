import pandas as pd
import numpy as np
from tqdm import tqdm
import gzip
import pickle
from joblib import Parallel, delayed

from neuromaps.nulls import alexander_bloch
from nispace.nulls import nulls_moran


def spin_nulls(data, parc, parc_space="fsLR", parc_density="32k",
               n_nulls=1000, seed=None, dtype=np.float32, n_jobs=8):
    """
    Spin null maps using neuromaps' alexander_bloch, with mirror-symmetric rotations
    across hemispheres (handled internally by neuromaps via _gen_rotation).

    data   : (n_maps, n_lh + n_rh) array, LH parcels first
    returns: dict {map_label: (n_nulls, n_parcels)}
    """
    if isinstance(data, pd.DataFrame):
        names = data.index
    else:
        names = [f"map_{i}" for i in range(len(data))]
    data = np.atleast_2d(np.array(data, dtype=dtype))

    def null_fun(data_1d):
        return alexander_bloch(
            data_1d,
            atlas=parc_space,
            density=parc_density,
            parcellation=parc,
            n_perm=n_nulls,
            seed=seed,
        ).astype(dtype).T   # (n_nulls, n_parcels)
    null_maps = Parallel(n_jobs=n_jobs)(
        delayed(null_fun)(data[i]) 
        for i in tqdm(range(data.shape[0]))
    )
    null_maps = {names[i]: null_maps[i] for i in range(data.shape[0])}
    return null_maps

def moran_nulls(data, dist_mat, n_nulls=1000, seed=None):
    """
    Moran spectral randomization null maps using NiSpace, if given a largely symmetric distance
    matrix (symmetrized cortical parcellation or most subcortical parcellations), this will
    produce largely symmetric nulls.

    data   : (n_maps, n_lh + n_rh) array, LH parcels first
    returns: dict {map_label: (n_nulls, n_parcels)}
    """
    if isinstance(data, pd.DataFrame):
        names = data.index
    else:
        names = [f"map_{i}" for i in range(len(data))]
    data = np.atleast_2d(np.array(data))
    dist_mat = np.array(dist_mat)
    
    null_maps = {}
    for i in range(data.shape[0]):
        null_maps[names[i]] = nulls_moran(
            data[i],
            dist_mat,
            n_nulls=n_nulls,
            seed=seed
        )   # (n_nulls, n_parcels)
    return null_maps