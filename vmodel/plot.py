import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from vmodel.geometry import tangent_points_to_circle
from vmodel.util import sort_dict
import vmodel.util.xarray as xrutil
from scipy.spatial import ConvexHull

REDUCE_NAMES = {
    'min': 'Mininum',
    'mean': 'Mean',
}


def plot_num_agents_vs_metric(ax, dsdict: dict, metric, skip_timesteps=100, **kwargs):
    dsdict = sort_dict(dsdict)
    levels = list(dsdict.keys())
    ys, yerrs = [], []
    for level, ds in dsdict.items():
        timeslice = slice(skip_timesteps, None)
        da = getattr(ds, metric).isel(time=timeslice)
        mean = da.mean(['time'])
        ys.append(mean.mean('run')); yerrs.append(mean.std('run'))
    ax.errorbar(levels, ys, yerr=yerrs, capsize=3, **kwargs)
    # ax.set(xlabel='Number of agents (log)', ylabel=f'{metric}')
    # ax.set(xscale='log')
    # ax.grid()
    # ax.legend()


def plot_num_agents_vs_nn_distance(ax, dsdict: dict, skip_timesteps=100, reduce='min',
                                   **kwargs):
    """Plot nearest neighbor distance level statistics
    Args:
        dsdict: num_agents -> ds
        skip_timesteps: Number of timesteps to skip
    """
    dsdict = sort_dict(dsdict)
    levels = list(dsdict.keys())
    ys, yerrs = [], []
    for level, ds in dsdict.items():
        timeslice = slice(skip_timesteps, None)
        mindist = ds.distance.isel(time=timeslice).min(['agent', 'time'])
        dist = getattr(mindist, reduce)('agent2')
        y, yerr = dist.mean('run'), dist.std('run')
        ys.append(y); yerrs.append(yerr)
    ax.errorbar(levels, ys, yerr=yerrs, capsize=3, fmt='-o', **kwargs)
    # ax.set(xlabel='Number of agents', ylabel=f'{name} nearest neighbor distance [m]')
    # ax.set(xscale='log')
    # ax.set(xticks=levels, xticklabels=levels)
    # ax.set(ylim=(0, None))
    # ax.grid()
    # ax.legend()


def plot_num_agents_vs_nn_distance_boxplot(ax, ds_dict, level_name='Number of agents',
                                           skip_timesteps=100):
    """Plot distance level statistics
    Args:
        exps (dict): keys: levels, values (ndarray) exp x time x agent x state
    """
    levels = list(ds_dict.keys())
    xs = []
    for _, ds in ds_dict.items():
        radius_collision = 2 * ds.attrs['radius']
        timeslice = slice(skip_timesteps, None)
        dist = ds.distance.isel(time=timeslice).min(['agent', 'agent2', 'time'])
        xs.append(dist)
    ax.boxplot(xs)
    xs, ys = np.arange(len(xs)) + 1, np.ones(len(levels)) * radius_collision
    ax.plot(xs, ys, color='tab:red', linestyle='--', label='Collision radius')
    ax.set(xticklabels=levels)
    ax.set(xlabel=level_name, ylabel='Minimum distance [m]')
    ax.set(ylim=(0, None))
    ax.legend()


def plot_visibility(ax, ds, focal_agent=1):
    """Plot visibility graph
    Args:
        ds (dataset): agent x space
    TODO: add perception radius to plot
    """
    radius_agent = ds.attrs['radius']

    pos_self = ds.position.sel(agent=focal_agent).data
    xz, yz = pos_self

    for a in ds.agent.data:
        visible = ds.visibility.sel(agent=focal_agent, agent2=a).data

        alpha = 1.0 if visible else 0.1
        color = 'tab:red' if a == focal_agent else 'tab:blue'
        zorder = 99 if a == focal_agent else 1

        pos = ds.position.sel(agent=a).data
        x, y = pos
        ax.plot(x, y)  # only for automatic dim!
        circle = plt.Circle((x, y), radius=radius_agent, color=color, alpha=alpha,
                            zorder=zorder)
        ax.add_patch(circle)
        align = 'center'
        ax.text(x, y, f'{a}', ha=align, va=align, size='x-large', alpha=alpha,
                zorder=zorder)

        points_or_none = tangent_points_to_circle(pos - pos_self, radius_agent)
        if points_or_none is None:
            continue
        points_or_none += pos_self
        (x1, y1), (x2, y2) = points_or_none
        ax.plot([xz, x1], [yz, y1], color=color, alpha=alpha)
        ax.plot([xz, x2], [yz, y2], color=color, alpha=alpha)

    ax.set(aspect='equal')
    ax.set(xlabel='x [m]', ylabel='y [m]')


def plot_visibility_statistics(ax, ds_dict, plot_fit=True, skip_timesteps=100):
    """Plot statistics about visibility
    Args:
        ds_dict (dict): keys (levels), values (dataset)
    """

    levels = list(ds_dict.keys())
    width = (levels[1] - levels[0]) / 2
    align, capsize, color = 'center', 5, 'C0'
    xs, ys, yerrs = [], [], []

    for level, ds in ds_dict.items():
        # Get attrs
        visibility = ds.visibility  # based on occlusion model
        # visibility = ds.distance < perception_radius  # based on perception radius
        # visibility = ds.visibility & (ds.distance < perception_radius)  # based on both
        timeslice = slice(skip_timesteps, None)
        y = visibility.sum('agent').isel(time=timeslice).mean(['time', 'agent2'])
        x, y, yerr = level, y.mean('run').data, y.std('run').data
        ax.bar(x, y, yerr=yerr, align=align, width=width, capsize=capsize, color=color)
        xs.append(x); ys.append(y), yerrs.append(yerr)

    if plot_fit:
        from scipy.optimize import curve_fit
        f = lambda x, a, b: a * np.log(x) + b
        popt, pcov = curve_fit(f, xs, ys)
        color, linestyle, alpha = 'tab:red', '--', 0.5
        label = 'Logarithmic fit'
        ax.plot(xs, f(xs, *popt), label=label, color=color, linestyle=linestyle, alpha=alpha)

    ax.set(xlabel='Number of agents', ylabel='Average number of visible neighbors')
    ax.set(xticks=levels, xticklabels=levels)
    ax.legend()


def plot_distance_metric_statistics(ax, ds, start=None):
    """Plot distance metrix boxplots
    Args:
        ds (dataset): experiment x timestep x agent x agent
    """
    mindist = ds.distance.min('agent2')  # experiment x timestep x agent
    mindist = mindist.isel(time=slice(start, None))

    align, capsize = 'center', 5

    # Plot min min distance mean
    min_mindist = mindist.min(['agent', 'time'])  # experiment
    x, y, yerr = 0, min_mindist.mean('run'), min_mindist.std('run')
    ax.bar(x, y, yerr=yerr, align=align, capsize=capsize)

    # Plot mean mean distance mean
    mean_mindist = mindist.mean(['agent', 'time'])
    x, y, yerr = 1, mean_mindist.mean('run'), mean_mindist.std('run')
    ax.bar(x, y, yerr=yerr, align=align, capsize=capsize)

    # Metadata
    ax.set(xticks=[0, 1], xticklabels=['minimum', 'mean'])
    ax.set(ylabel='inter-agent distance [m]', xlabel='distance over multiple runs')


def plot_nn_connectivity(ax, ds):
    """Plot nearest neighbor connectivity"""
    radius_agent = ds.attrs['radius']
    offset = 0.3 * radius_agent
    # Plot positions with label for agent ID
    color = 'C0'
    for agent in ds.agent.data:
        # Plot agents
        x, y = ds.position.sel(agent=agent)  # x, y
        ax.add_patch(plt.Circle((x, y), radius=radius_agent, color=color, zorder=99))
        # line = ax.plot(x, y, marker='o', color=color)
        ax.text(x - offset, y - offset, s=f'{agent}', zorder=100)

        # Plot minimum distances between agents
        agent_min = ds.distance.sel(agent=agent).argmin().data + 1  # minimum index to other agent
        x_min, y_min = ds.position.sel(agent=agent_min)
        xs, ys = np.array([x, x_min]), np.array([y, y_min])
        alpha = 0.5
        ax.plot(xs, ys, color=color, alpha=alpha)
        min_dist = ds.distance.sel(agent=agent).min().data
        ax.text(xs.mean() - offset, ys.mean() + offset, s=f'{min_dist:.2f}', alpha=alpha)
        # print(f'Agent {agent} has nearest neighbor {agent_min} (distance: {min_dist:.2f} m)')
    ax.set(xlabel='x [m]', ylabel='y [m]')
    ax.set(aspect='equal')


def plot_trajectories(ax, ds):
    """Plot trajectories of positions over time
    Args:
        ds (dataset): timestep x agent x state
    """
    for agent in ds.agent.data:
        xs, ys = ds.position.sel(agent=agent)
        line = ax.plot(xs, ys)
        color = line[0].get_color()
        ax.plot(xs.isel(time=0), ys.isel(time=0), color=color, marker='s')
        ax.plot(xs.isel(time=-1), ys.isel(time=-1), color=color, marker='o')
    ax.set(xlabel='x [m]', ylabel='y [m]')
    ax.set(aspect='equal')


def plot_convex_hull(ax, ds):
    """Plot convex hull of agents
        Args:
            ds (dataset): nagents x nstate
    """
    pos = ds.position.data
    color = 'C0'
    hull = ConvexHull(pos)
    periphery = hull.vertices
    core = [i for i in range(len(pos)) if i not in periphery]
    xs, ys = pos[core, 0], pos[core, 1]
    ax.scatter(xs, ys, color=color, alpha=0.5)
    xs, ys = pos[periphery, 0], pos[periphery, 1]
    ax.scatter(xs, ys, color='tab:red')
    for s in hull.simplices:
        xs, ys = pos[s, 0], pos[s, 1]
        ax.plot(xs, ys, color='tab:red', alpha=0.4)
    ax.set(xlabel='x [m]', ylabel='y [m]')
    ax.set(aspect='equal')


def plot_nn_distance_timeseries(ax, ds):
    """Plot distance metrics as timeseries
    Args:
        dmat (array): timestep x agent x state
    """
    # Get attrs
    radius_agent = ds.attrs['radius']
    radius_collision = 2 * radius_agent
    radius_safety = 2 * radius_collision

    # Compute mininum distances
    ts = np.array(ds.time, dtype=float) / 1e9
    min_dist_ts = ds.distance.min('agent2')

    # Plot mean minimum distance
    ys, yerr = min_dist_ts.mean('agent').data, min_dist_ts.std('agent')
    ax.plot(ts, ys, label=f'mean minimum distance (avg: {ys.mean():.2f} m)')
    ax.fill_between(ts, ys - yerr, ys + yerr, alpha=0.4)

    # Plot minimum distance
    ys = min_dist_ts.min('agent').data
    ax.plot(ts, ys, label=f'minimum distance (avg: {ys.mean():.2f} m)')

    # Plot safety and collision radius
    ls = '--'
    ys = np.ones_like(ts, dtype=float) * radius_safety
    ax.plot(ts, ys, linestyle=ls, color='orange', label='safety radius')
    ys = np.ones_like(ts, dtype=float) * radius_collision
    ax.plot(ts, ys, linestyle=ls, color='red', label='collision radius')

    ax.set(xlabel='time [s]', ylabel='distance [m]')
    ax.set(ylim=(0, None))
    ax.legend()


def plot_speed_timeseries(ax, ds):
    """Plot speed timeseries
    Args:
        ds (dataset): time x agent x space
    """
    # Compute speed from velocity
    speed_ts = xrutil.norm(ds.velocity, 'space')
    ts = np.array(ds.time, dtype=float) / 1e9
    ys, yerr = speed_ts.mean('agent'), speed_ts.std('agent')

    # Plot speed timeseries
    ax.plot(ts, ys, label='Mean speed')
    ax.fill_between(ts, ys - yerr, ys + yerr, alpha=0.4)
    ax.set(xlabel='time [s]', ylabel='speed [m/s]')


def plot_collision_threshold(ax, threshold, xmin, xmax, **kwargs):
    xs = [xmin, xmax]
    ys = [threshold, threshold]
    ax.plot(xs, ys, color='tab:red', linestyle='--', alpha=0.75, **kwargs)
    ax.legend()


def plot_mean_relative_velocity_by_distance_to_swarm_center(ax, ds):
    """Plot average velocity by distance to swarm center
    Args:
        ds (ds): each value: run x time x agent x state
    """
    num_agents = len(ds.agent)
    # Calculate relative distance to swarm center
    dist_ts = xrutil.norm(ds.position.mean('agent') - ds.position, 'space')
    # Subtract mean velocity to compute relative speed
    speed_ts = xrutil.norm(ds.velocity - ds.velocity.mean('agent'), 'space')
    topo_speed_ts = xr.zeros_like(speed_ts)
    for t in ds.time:
        # Calculate relative distances to swarm center
        # Create a timeseries that sorts the speed by the distance to the center of the swarm
        indices = dist_ts.sel(time=t).argsort().data + 1
        topo_speed_ts.loc[dict(time=t)] = speed_ts.sel(time=t, agent=indices).data
    mean_topo_speed = topo_speed_ts.mean('time')
    xs, ys = np.arange(num_agents) + 1, mean_topo_speed
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set(xlabel='topological distance from swarm center', ylabel='average speed [m/s]')
    ax.set(xlim=(0, num_agents + 1))


def plot_core_hull_vs_mean_relative_speed(ax, ds):
    """Plot average speed of core vs. peripheral agents
    Args:
        exp (dict): run x time x agent x state
    """
    width, capsize, color = 0.5, 5, 'C0'

    speed = xrutil.norm(ds.velocity - ds.velocity.mean('agent'), 'space')
    speed_core = speed.where(~ds.hull).mean(['agent', 'time'])
    x, y, yerr = 0, speed_core.mean().data, speed_core.std().data
    ax.bar(x, y, yerr=yerr, width=width, capsize=capsize, color=color)

    speed_hull = speed.where(ds.hull).mean(['agent', 'time'])
    x, y, yerr = 1, speed_hull.mean().data, speed_core.std().data
    ax.bar(x, y, yerr=yerr, width=width, capsize=capsize, color=color)

    # Plot
    ax.set(xticks=[0, 1], xticklabels=['core', 'hull'])
    ax.set(ylabel='mean speed [m/s]')


def voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D
    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points: bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha: float, optional
        Specifies the line alpha for polygon boundaries
    point_size: float, optional
        Specifies the size of points
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    See Also
    --------
    Voronoi
    Notes
    -----
    Requires Matplotlib.
    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)
    line_style = kw.get('line_style', 'solid')

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max() * 100

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle=line_style))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle=line_style))

    return ax.figure
