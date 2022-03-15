import argparse

import numpy as np

from vmodel.geometry import subtended_angle_circle, voronoi_neighbors
from vmodel.math import filter_by_angle, limit_norm, wrap_angle
from vmodel.random import random_uniform_within_circle
from vmodel.visibility import visibility_set


# Uncomment the @profile decorator and run to perform line-by-line profiling:
# kernprof -l -v vmodel ...
# @profile
def update_single(i, positions, func, args):
    """Update a single agent
    Args:
        i: index of focal agent (zero-indexed)
        positions: absolute positions of agents
        func: flocking algorithm function to apply
        args: parsed command-line arguments
    Returns:
        command: velocity command for focal agent
        cache: holds precomputed values
    """

    cache = argparse.Namespace()

    # Shorthands for arguments
    ndim = args.num_dims

    # Get relative positions of others
    pos_self = positions[i]
    pos_others = np.delete(positions, i, axis=0)
    pos = pos_others - pos_self
    # pos_waypoint = args.pos_waypoint - pos_self

    # Keep track of original agent indices
    idx = np.arange(len(pos), dtype=int)

    # Radii of agents
    rad = np.full(len(pos), args.radius)

    # Compute relative distances
    dist = np.linalg.norm(pos, axis=1)
    cache.dist = dist.copy()
    # cache.dist = np.insert(dist, i, 0.0)

    # Optionally add false positive detections
    if args.num_clutter > 0.0:
        # Add more clutter if not enough available
        num_clutter = np.random.poisson(args.num_clutter)

        if num_clutter > 0:
            low, high = args.radius_safety, args.perception_radius
            pos_clutter = random_uniform_within_circle(low, high,
                                                       size=(num_clutter, ndim))
            pos_clutter = np.array(pos_clutter).reshape(-1, ndim)

            # while len(clutter_list) < num_clutter:
            #     low, high = args.radius_arena, args.radius_arena + args.perception_radius
            #     clutter = sample_fn(low=low, high=high)
            #     clutter_list.append(clutter)

            # # Ranomly choose clutters from list
            # choice = np.random.choice(np.arange(len(clutter_list)), num_clutter)
            # pos_clutter = np.array(clutter_list)[choice].reshape(-1, ndim)

            # Add clutter to rest of detections
            dist_clutter = np.linalg.norm(pos_clutter, axis=1)
            pos = np.concatenate([pos, pos_clutter])
            dist = np.concatenate([dist, dist_clutter])
            idx = np.arange(len(pos), dtype=int)
            rad = np.concatenate([rad, np.full(len(pos_clutter), args.radius)])

    # If using visual migration, check if the waypoint is visible (as with agents)
    # waypoint_visible = True
    # if args.visual_migration:
    #     idx_waypoint = idx[-1] + 1
    #     idx = np.append(idx, idx_waypoint)
    #     rad = np.append(rad, args.radius_waypoint)
    #     dist = np.append(dist, np.linalg.norm(pos_waypoint))
    #     pos = np.append(pos, [pos_waypoint], axis=0)

    # Filter out agents by metric distance
    if args.perception_radius > 0:
        mask = dist < args.perception_radius
        pos, dist, idx, rad = pos[mask], dist[mask], idx[mask], rad[mask]

    # Filter out agents by angle proportion of field of view
    if args.perception_angle > 0:
        angles_rad = subtended_angle_circle(dist, rad)
        mask = angles_rad > np.deg2rad(args.perception_angle)
        pos, dist, idx, rad = pos[mask], dist[mask], idx[mask], rad[mask]

    # Filter out occluded agents
    if args.filter_occluded:
        mask = visibility_set(pos, rad, dist)
        pos, dist, idx = pos[mask], dist[mask], idx[mask]

    # Filter out waypoint (if visible)
    # if args.visual_migration:
    #     waypoint_visible = idx_waypoint in idx
    #     if waypoint_visible:
    #         mask = idx != idx_waypoint
    #         pos, dist, idx = pos[mask], dist[mask], idx[mask]

    # Filter out agents by topological distance
    if args.max_agents > 0:
        indices = dist.argsort()[:args.max_agents]
        pos, dist, idx = pos[indices], dist[indices], idx[indices]

    if args.topo_angle > 0:
        angle = np.deg2rad(args.topo_angle)
        mask = filter_by_angle(pos, angle)
        pos, dist, idx = pos[mask], dist[mask], idx[mask]

    if args.filter_voronoi:
        posme = np.insert(pos, 0, np.zeros(ndim), axis=0)
        try:
            indices = np.array(voronoi_neighbors(posme)[0]) - 1
        except Exception:
            pass  # in case there are not enough points, do nothing
        else:
            pos, dist, idx = pos[indices], dist[indices], idx[indices]

    # Save visibility data (after applying perception filtering)
    visibility = np.zeros(len(pos_others), dtype=bool)
    visibility[idx[idx < len(pos_others)]] = True
    # cache.vis = np.insert(visibility, i, False)
    cache.vis = visibility.copy()

    # Optionally add noise to range and bearing
    # We add range/bearing noise *after* the visual filtering since it prevents false
    # occlusions where (visually speaking) there weren't any
    if args.range_std > 0 or args.bearing_std > 0:
        xs, ys = pos.T.copy()
        distance = dist.copy()
        bearing = np.arctan2(ys, xs)  # [-pi, +pi]
        noise_distance, noise_bearing = np.random.normal(size=(ndim, len(xs)))
        noise_distance *= args.range_std
        noise_bearing *= np.deg2rad(args.bearing_std)
        distance += (noise_distance * distance)  # distance noise is distance-dependent!
        bearing += noise_bearing
        bearing = wrap_angle(bearing)  # wrap to [-pi, +pi]
        xs, ys = distance * np.cos(bearing), distance * np.sin(bearing)
        pos = np.array([xs, ys]).T
        dist = np.linalg.norm(pos, axis=1)

    # Optionally discard agents with false negative prob
    if args.false_negative_prob:
        keep_prob, size = 1 - args.false_negative_prob, len(pos)
        mask = np.random.binomial(n=1, p=keep_prob, size=size).astype(bool)
        pos, dist = pos[mask], dist[mask]

    # Compute command from interactions (if any relative positions)
    command_interaction = np.zeros(ndim)

    if len(pos) > 0:
        command_interaction = func(pos, dist)

    # Compute migration command
    command_migration = np.zeros(ndim)

    # Migrate to migration point
    if len(args.migration_point) > 0:
        pos_waypoint = (args.migration_point - pos_self)
        command_migration = limit_norm(pos_waypoint, args.migration_gain)

    # Follow general migration direction
    if len(args.migration_dir) > 0:
        command_migration = limit_norm(args.migration_dir, args.migration_gain)

    # Compute final command
    command = command_interaction + command_migration

    # Limit speed
    command = limit_norm(command, args.max_speed)

    return command, cache
