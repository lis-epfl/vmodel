import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from vmodel import geometry as vgeom
from vmodel import plot
from vmodel.util import color as vcolor


def plot_all(ax, data, args, focal=0):

    # Plot setup
    ax.clear()
    ax.grid(True, zorder=-1, linestyle='dotted')
    ax.set(xlabel=r'$x$ position [$m$]', ylabel=r'$y$ position [$m$]')
    # ax.set_aspect('equal', 'datalim')
    ax.set_aspect('equal')
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)

    # Plot agents and visibility
    if args.perception_radius > 0:
        plot_nselect_metric(ax, data.pos[-1], args.perception_radius, focal=focal)
    if args.filter_occluded:
        plot_nselect_visual(ax, data.pos[-1], radius=args.radius, focal=focal)
    if args.max_agents > 0:
        plot_nselect_topo(ax, data.pos[-1], data.vis[-1], focal=focal)
    if args.filter_voronoi:
        plot_nselect_voronoi(ax, data.pos[-1], focal=focal)
    plot_agents(ax, data.pos, data.vel, data.vis, radius=args.radius, tail_length=20)

    # Plot waypoint
    # plot.plot_circle(ax, args.pos_waypoint, color='tab:orange',
    #                  radius=args.radius_waypoint, zorder=-10)
    # Plot arena
    # plot.plot_circle(ax, (0, 0), radius=args.radius_arena, color='coral',
    #                  fill=False, zorder=0, ls='--', alpha=0.5)
    # plot.plot_lattice(ax, data.pos[-1], data.dist[-1], zorder=-1)


def plot_nselect_metric(ax, positions, perception_radius, focal=0):
    x, y = positions[focal]
    perc_circle = plt.Circle((x, y), radius=perception_radius, fill=False, ls='-',
                             lw=0.25, ec=vcolor.grey, zorder=100)
    ax.add_patch(perc_circle)
    perc_radius = plt.Circle((x, y), radius=perception_radius, color='white',
                             zorder=-1)
    ax.add_patch(perc_radius)
    ax.set(facecolor=vcolor.background)
    k = 1.1
    radius = k * perception_radius
    xlim = (x - radius, x + radius)
    ylim = (y - radius, y + radius)
    ax.set(xlim=xlim, ylim=ylim)


def plot_nselect_visual(ax, positions, radius=0.25, focal=0):

    pos_self = positions[focal]
    for a in range(len(positions)):

        # Don't draw shadows for focal agent
        if a == focal:
            continue

        rel = positions[a] - pos_self
        p1, p2 = vgeom.tangent_points_to_circle(rel, radius)
        p1, p2 = np.array(p1), np.array(p2)
        u1, u2 = p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2)
        scale = 100
        ps1, ps2 = u1 * scale, u2 * scale
        poly = np.array([p1, ps1, ps2, p2]) + pos_self
        polygon = plt.Polygon(poly, color=vcolor.shadow, zorder=-1)
        ax.add_patch(polygon)


def plot_nselect_topo(ax, positions, visibility, focal=0):
    pos_self = positions[focal]
    x, y = pos_self
    vis = visibility[focal]
    for i in range(len(positions)):
        isvisible = vis[i]
        isfocal = (i == focal)

        # Don't draw connecting line to focal or invisible agent
        if isfocal or not isvisible:
            continue

        xt, yt = positions[i]
        ax.plot([x, xt], [y, yt], color=vcolor.grey, lw=1)


def plot_nselect_voronoi(ax, positions, color_regions=False, focal=0):
    vor = Voronoi(positions)
    neighbors = np.array(vgeom.voronoi_neighbors(positions)[0]) - 1
    plot.voronoi_plot_2d(vor, ax=ax, show_vertices=False, point_size=0, line_alpha=0.7,
                         line_colors=vcolor.grey, line_width=0.25, line_style='-')

    if not color_regions:
        return

    # Color all non neighbor regions light grey
    for index, r in enumerate(vor.point_region):
        region = vor.regions[r]
        if index - 1 in neighbors or index == 0:
            continue
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=vcolor.lightgrey)


def plot_agents(ax, positions, velocities=None, visibility=None, radius=0.25,
                focal_agent=0, show_identity=False, tail_length=0):
    """Plot agents
    Args:
        positions (ndarray): Position array (T x N x D)
        velocities (ndarray): Velocity array (T x x N x D)
        visibility (ndarray): Visibility matrix (T x N x N)
        radius (float): Agent radius
        show_identity (bool): If true, show the agent identity
    """
    post, velt, vist = positions[-1], velocities[-1], visibility[-1]

    pos_self = post[focal_agent]
    x, y = pos_self

    postail = np.array(positions[-tail_length:])

    for a in range(len(post)):

        color = vcolor.visible
        x, y = post[a]

        isfocal = a == focal_agent
        isvisible = vist[focal_agent][a]

        if isfocal:
            color = vcolor.focal

        # Plot visibility
        if visibility is not None:
            if not isvisible and not isfocal:
                color = vcolor.invisible

        # Plot positions
        plot_circle(ax, (x, y), radius=radius, color=color, zorder=10)

        if show_identity:
            ax.text(x, y, s=f'{a + 1}', ha='center', va='center')

        alpha = 0.3

        # Plot tails
        if tail_length > 0:
            xs, ys = postail[:, a].T
            ax.plot(xs, ys, color=color, lw=1, alpha=alpha)

        # Plot velocities
        if velocities is not None:
            width = radius / 32
            head_width = radius / 2
            head_length = radius / 2
            vel, pos = velt[a], post[a]
            # Draw velocity vectors outside of agent radius!
            speed = np.linalg.norm(vel)
            dir = vel / (1e-9 + speed)
            x, y = pos + dir * radius
            scaled = np.maximum(0, speed - radius) / speed  # scale speed by rad
            dx, dy = dir * scaled
            ax.arrow(x, y, dx, dy, width=width, head_width=head_width,
                     length_includes_head=False, head_length=head_length,
                     zorder=10, edgecolor=color, facecolor=color, alpha=alpha)


def plot_circle(ax, pos, **kwargs):
    x, y = pos
    ax.plot(x, y)  # only for lims calculation
    ax.add_patch(plt.Circle((x, y), **kwargs))


def plot_clutter(ax, clutters):
    if len(clutters) == 0:
        return
    xs, ys = np.array(clutters).reshape(-1, 2).T
    ax.scatter(xs, ys, marker='x')


def plot_lattice(ax, positions, distances, distance_setpoint=None, epsilon=None, **kwargs):
    """Plot quasi-lattice
    Args:
        positions: (ndarray): Positions (N x D)
        distances: (ndarray): Distance matrix (N x N)
        distance_setpoint (float): Desired reference distance, take mean if not given
        epsilon (float): Max deviation from distance setpoint, use 1/3 mean if not given
    """
    # Use mean nearest neighbor distance as reference distance
    dmat = np.array(distances)
    dmat[dmat == 0.0] = float('inf')
    if distance_setpoint is None:
        distance_setpoint = dmat.min(axis=0).mean()
    if epsilon is None:
        epsilon = distance_setpoint / 3
    a, b = distance_setpoint - epsilon, distance_setpoint + epsilon
    for i in range(len(positions)):
        indices = np.arange(len(positions))
        dist = dmat[i]
        js = indices[(dist > a) & (dist < b)]
        for j in js:
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            xs, ys = [x1, x2], [y1, y2]
            ax.plot(xs, ys, color='silver', **kwargs)


def plot_distances(ax, distances, time, radius=0.25):
    """Plot distances
    Args:
        distances (ndarray): Distance matrices (K x N x N)
        time (ndarray): Time array
        radius (float): Agent radius [m]
    """
    dmats = np.array(distances)
    dmats[dmats == 0.0] = float('inf')
    min_dist = dmats.min(axis=-1)
    xs, xerr = min_dist.mean(axis=-1), min_dist.std(axis=-1)
    ax.plot(time, xs, color='tab:blue', label=f'mean & std ({xs[-1]:.2f} m)')
    ax.fill_between(time, xs - xerr, xs + xerr, alpha=0.25)
    xs = min_dist.min(axis=-1)
    ax.plot(time, xs, color='tab:red', label=f'minimum ({xs[-1]:.2f} m)')
    xs = np.ones(len(time)) * radius * 2
    ax.plot(time, xs, color='tab:red', ls='--', alpha=0.5, label='collision distance')


def plot_speed(ax, velocities, time):
    """Plot speed
    Args:
        velocities: velocity matrix (K x N x D)
        time: time array
    """
    vels = np.array(velocities)
    speeds = np.linalg.norm(vels, axis=-1)  # K x N
    xs, xerr = speeds.mean(axis=-1), speeds.std(axis=-1)  # K
    ax.plot(time, xs, color='tab:blue', label=f'mean & std ({xs[-1]:.2f} m/s)')
    ax.fill_between(time, xs - xerr, xs + xerr, alpha=0.25)


def plot_metric(ax, metrics, time, name='metric', **kwargs):
    """
    Args:
        metrics: Array of length K containing metrics
        time: Time array
    """
    ax.plot(time, metrics, label=f'{name} ({metrics[-1]:.2f})', **kwargs)
