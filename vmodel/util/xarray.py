import numpy as np
import xarray as xr

from vmodel import geometry as vgeom
from vmodel import metrics as vmetr
from vmodel import visibility as vvisi


def metric_ufunc(da: xr.DataArray, ufunc, dims):
    dims = list(dims)
    name = ufunc.__name__
    return xr.apply_ufunc(ufunc, da, input_core_dims=[dims], vectorize=True,
                          output_dtypes=[np.float],
                          dask='parallelized').rename(name)


def nndist(da: xr.DataArray) -> xr.DataArray:

    def compute_nndist(x):
        dmat = vmetr.distance_matrix(x)
        np.fill_diagonal(dmat, float('inf'))  # otherwise min(dmat) = 0 for dist. to self
        nndist = dmat.min(axis=0)
        return nndist

    input_dims = [['agent', 'space']]
    output_dims = [['agent']]
    return xr.apply_ufunc(compute_nndist, da, input_core_dims=input_dims,
                          output_core_dims=output_dims, vectorize=True,
                          dask='parallelized').rename('nndist')


def convex_hull(da: xr.DataArray) -> xr.DataArray:
    result = xr.apply_ufunc(vgeom.convex_hull, da, input_core_dims=[['agent', 'space']],
                            output_core_dims=[['agent']], vectorize=True)
    result = result.transpose(..., 'agent', 'time')
    result.name = 'hull'
    return result


def norm(da: xr.DataArray, dims, ord=None) -> xr.DataArray:
    kwargs = {'ord': ord, 'axis': -1}
    name = f'{da.name}_norm'
    return xr.apply_ufunc(np.linalg.norm, da, input_core_dims=[[dims]], kwargs=kwargs,
                          dask='parallelized').rename(name)


def connectivity(da: xr.DataArray) -> xr.DataArray:
    result = xr.apply_ufunc(vmetr.connectivity, da,
                            input_core_dims=[['agent', 'agent2']], vectorize=True)
    result.name = 'connectivity'
    return result


def distance_matrix(da: xr.DataArray, fill_diag=True) -> xr.DataArray:
    from scipy.spatial import distance_matrix

    dm = lambda x: distance_matrix(x, x)

    dmat = xr.apply_ufunc(dm, da, input_core_dims=[['agent', 'space']],
                          output_core_dims=[['agent', 'agent2']], vectorize=True)
    dmat.name = 'distance'
    if fill_diag:
        dmat = dmat.where(dmat != 0, float('inf'))
    return dmat


def visibility(da: xr.DataArray) -> xr.DataArray:

    def compute_visibility(pos, rad=float(da.radius)):
        return vvisi.visibility_graph(pos, rad)

    input_dims = [['agent', 'space']]
    output_dims = [['agent', 'agent2']]
    output_sizes = {'agent2': len(da.agent)}
    return xr.apply_ufunc(compute_visibility, da,
                          input_core_dims=input_dims,
                          output_core_dims=output_dims,
                          output_dtypes=[np.bool],
                          dask_gufunc_kwargs={'output_sizes': output_sizes},
                          vectorize=True, dask='parallelized')
