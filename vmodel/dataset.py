import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from vmodel.util.util import clean_attrs


def generate_filename(args):

    # Construct output file name
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    fnamedict = {
        'agents': args.num_agents,
        'runs': args.num_runs,
        'times': args.num_timesteps,
        'dist': args.ref_distance,
        'perc': args.perception_radius,
        'topo': args.max_agents,
        'rngstd': args.range_std,
    }
    formatexts = {'netcdf': 'nc', 'pickle': 'pkl'}
    args_str = '_'.join(f'{k}_{v}' for k, v in fnamedict.items())
    return f'{time_str}_{args_str}.states.{formatexts[args.format]}'


def create_dataset(datas, args):

    ds = xr.Dataset()

    # Clean up attrs dict to be compatible with YAML and NETCDF
    ds.attrs = clean_attrs(vars(args))

    time = np.array(datas[0].time)
    pos = np.array([d.pos for d in datas])
    vel = np.array([d.vel for d in datas])

    coord_run = np.arange(args.num_runs, dtype=int) + 1
    coord_time = pd.to_timedelta(time, unit='s')
    coord_agent = np.arange(args.num_agents, dtype=int) + 1
    coord_space = np.array(['x', 'y'])

    coords_rtas = {
        'run': coord_run,
        'time': coord_time,
        'agent': coord_agent,
        'space': coord_space
    }

    dapos = xr.DataArray(pos, dims=coords_rtas.keys(), coords=coords_rtas)
    dapos.attrs['units'] = 'meters'
    dapos.attrs['long_name'] = 'position'
    ds['position'] = dapos

    davel = xr.DataArray(vel, dims=coords_rtas.keys(), coords=coords_rtas)
    davel.attrs['units'] = 'meters/second'
    davel.attrs['long_name'] = 'velocity'
    ds['velocity'] = davel

    ds = ds.transpose('run', 'agent', 'space', 'time')

    # Return only state (position and velocity)
    if args.no_save_precomputed:
        return ds

    coords_rtaa = {
        'run': coord_run,
        'time': coord_time,
        'agent': coord_agent,
        'agent2': coord_agent
    }

    vis = np.array([d.vis for d in datas])
    davis = xr.DataArray(vis, dims=coords_rtaa.keys(), coords=coords_rtaa)
    davis.attrs['units'] = 'boolean'
    davis.attrs['long_name'] = 'visibility'
    ds['visibility'] = davis

    # Tranpose to match data generated from Gazebo
    ds = ds.transpose('run', 'agent', 'agent2', 'space', 'time')

    return ds


def save_dataset(ds, fname, args):

    if args.format == 'pickle':
        with open(fname, 'wb') as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.format == 'netcdf':
        comp = dict(zlib=True, complevel=5)
        encoding = None if args.no_compress else {v: comp for v in ds.data_vars}
        ds.to_netcdf(fname, encoding=encoding)

    with open(f'{fname}.yaml', 'w') as f:
        yaml.dump(ds.attrs, f)
