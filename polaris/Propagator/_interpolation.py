"""
For interpolation of trajectories
"""

import numpy as np
from scipy.interpolate import splev, splrep

def prepare_interpolation(propout):
    """Prepare a list of B-spline representation of [x,y,z,vx,vy,vz] against time,
    generated through `scipy.interpolate.splrep`
    
    Args:
    
    Returns:
        (list): list of B-spline tuple
    """
    # interpolate against time
    spline_list = [
        splrep(propout.times, propout.xs),
        splrep(propout.times, propout.ys),
        splrep(propout.times, propout.zs),
        splrep(propout.times, propout.vxs),
        splrep(propout.times, propout.vys),
        splrep(propout.times, propout.vzs),
    ]
    return spline_list


def evaluate_interpolation(spline_list, time):
    """Evaluate list of spline through `scipy.interoplate.splev`
    Note this function does not check that `time` is within the bounds of the spline!

    Args:
        spline_list (list): list of B-spline tuples 
        time (real): time to evaluate the state

    Returns:
        (np.array): state vector evaluated at input time
    """
    return np.array([splev(time, spl) for spl in spline_list])


def evaluate_time_grid(propout, eval_times):
    """Evaluate propagation output at input time steps
    
    Args:
        propout (list-like): 1-D time-steps to evaluate propagation output, length N

    Returns:
        (np.array): solutions, shape (N,6)
    """
    # prepare splines
    spline_list = prepare_interpolation(propout)
    # evaluate states
    states = []
    for t in eval_times:
        states.append(evaluate_interpolation(spline_list, t))
    return np.array(states)