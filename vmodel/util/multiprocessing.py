import signal
from multiprocessing import Pool


def get_silent_pool(processes=None):
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool(processes)
    signal.signal(signal.SIGINT, sigint_handler)
    return pool
