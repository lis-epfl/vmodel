import argparse
import os


def parse_vmodel_args() -> argparse.Namespace:

    def formatter_class(prog):
        return argparse.ArgumentDefaultsHelpFormatter(prog,
                                                      max_help_position=52,
                                                      width=90)

    parser = argparse.ArgumentParser(description='vmodel',
                                     formatter_class=formatter_class)

    # Script arguments
    formats = ['netcdf', 'pickle']
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='verbose output')
    parser.add_argument('-f', '--file', type=str, default='',
                        help='read parameters from YAML file')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='show live plots')
    parser.add_argument('--plot-metrics', action='store_true', default=False,
                        help='show live plots of metrics')
    parser.add_argument('--plot-every', type=int, default=10, metavar='K',
                        help='plot every k timesteps')
    parser.add_argument('--plot-blocking', action='store_true', default=False,
                        help='wait for key press to resume plotting')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        metavar='J', help='number of parallel jobs')
    parser.add_argument('-n', '--dry-run', action='store_true', default=False,
                        help='dry run, do not save data')
    parser.add_argument('-P', '--progress', action='store_true', default=False,
                        help='show progress bar')
    parser.add_argument('--save-every', type=int, default=10, metavar='K',
                        help='save data only every k timesteps')
    parser.add_argument('--parallel-agents', action='store_true', default=False,
                        help='process every agent in parallel')
    parser.add_argument('--no-parallel-runs', action='store_true',
                        default=False, help='do not process runs in parallel')
    parser.add_argument('--no-save-precomputed', action='store_true',
                        default=False,
                        help='save precomputed variables (saves memory)')
    parser.add_argument('--format', choices=formats, default='netcdf',
                        help='format for saved dataset')
    parser.add_argument('--no-compress', action='store_true', default=False,
                        help='do not compress datasets')

    # Experimental arguments
    algorithms = ['reynolds', 'olfati']
    spawns = ['poisson', 'uniform', 'grid']
    experiment = parser.add_argument_group('experiment arguments')
    experiment.add_argument('--num-agents', type=int, default=10, metavar='N',
                            help='number of agents')
    experiment.add_argument('--num-runs', type=int, default=1, metavar='N',
                            help='number of runs')
    experiment.add_argument('--num-dims', type=int, default=2, metavar='N',
                            help='number of dimensions')
    experiment.add_argument('--delta-time', type=float, default=0.1,
                            metavar='SEC',
                            help='time delta between timesteps [s]')
    experiment.add_argument('--num-timesteps', type=int, default=None,
                            metavar='K',
                            help='number of timesteps for experiment')
    experiment.add_argument('--algorithm', choices=algorithms,
                            default='reynolds', help='flocking algorithm')
    experiment.add_argument('--spawn', choices=spawns, default='uniform',
                            help='spawn method')
    experiment.add_argument('--spawn-distance', type=float, default=None,
                            metavar='F', help='spawn distance')
    experiment.add_argument('--seed', type=int, default=None,
                            help='set seed for repeatability')

    # Perception arguments
    perception = parser.add_argument_group('perception arguments')
    perception.add_argument('--radius', type=float, default=0.25, metavar='F',
                            help='radius of an agent [m]')
    perception.add_argument('--perception-radius', type=float, default=0.0,
                            metavar='F', help='perception radius of agents [m]')
    perception.add_argument('--perception-angle', type=float, default=0.0,
                            metavar='DEG',
                            help='angle below which objects are invisible [deg]')
    perception.add_argument('--filter-occluded', action='store_true',
                            default=False,
                            help='if true, filter out occluded agents')
    perception.add_argument('--filter-voronoi', action='store_true',
                            default=False,
                            help='if true, filter out non-voronoi neighbors')
    perception.add_argument('--max-agents', type=int, default=0, metavar='N',
                            help='maximum number of closest agents to consider')
    perception.add_argument('--false-negative-prob', type=float, default=0.0,
                            metavar='P',
                            help='false negative probability [prob]')
    perception.add_argument('--num-clutter', type=float, default=0.0,
                            metavar='K',
                            help='avg number of clutter returns per timestep')
    perception.add_argument('--range-std', type=float, default=0.0,
                            metavar='STD',
                            help='distance-scaled range std dev [m]')
    perception.add_argument('--bearing-std', type=float, default=0.0,
                            metavar='STD',
                            help='bearing std dev [deg]')
    perception.add_argument('--visual-migration', action='store_true',
                            default=False,
                            help='if true, subject waypoint to occlusion')
    perception.add_argument('--topo-angle', type=float, default=0.0,
                            metavar='DEG',
                            help='minimum angle between closest agents [deg]')

    # Control arguments
    control = parser.add_argument_group('control arguments')
    control.add_argument('--ref-distance', type=float, default=1,
                         metavar='F', help='desired inter-agent distance [m]')
    control.add_argument('--migration-gain', type=float, default=0.5,
                         metavar='K', help='migration gain [m/s]')
    control.add_argument('--migration-point', nargs=2, type=float, default=[],
                         metavar=('X', 'Y'), help='migration point (x, y)')
    control.add_argument('--migration-dir', nargs=2, type=float, default=[],
                         metavar=('X', 'Y'), help='migration direction (x, y)')
    control.add_argument('--max-speed', type=float, default=1.0, metavar='F',
                         help='Maximum speed [m/s]')

    # Need to parse known args when launching from ROS node
    # Reason: __name and __log are not specified above
    args, _ = parser.parse_known_args()

    return args
