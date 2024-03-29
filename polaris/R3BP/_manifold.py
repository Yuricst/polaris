#!/usr/bin/env python3
"""
Functions associated with generating manifold
"""


import numpy as np
from numba import jit
import numpy.linalg as la
from tqdm.auto import tqdm

from ..Propagator import propagate_cr3bp


class Manifold:
    """Class of manifold

    Attributes
        mu
        monodromy
        stable
        eps
        branches (lst): list of Branch object
    """

    def __init__(self, mu, monodromy, stable, eps):
        """Initialize manifold

        Args:
            mu (float): CR3BP mass parameter
            monodromy (np.array): monodromy matrix
            stable (bool): boolean
            eps (float): linear perturbation magnitude
        """
        self.mu = mu
        self.monodromy = monodromy
        self.stable = stable
        self.eps = eps
        self.branches = []

    def add_branch(self, branch):
        self.branches.append(branch)
        return


class Branch:
    """Class of single manifold branch"""

    def __init__(self, mu, eps0, state0, stm, timeAlongHalo, extractind):
        self.eps0 = eps0
        self.state0 = state0
        self.stm = stm
        self.timeAlongHalo = timeAlongHalo
        self.extractind = extractind

    def set_eigenvectors(self, y0, y, tf_manif):
        self.y0 = y0
        self.y = y
        self.tf_manif = tf_manif

    def set_branch_propagation(self, state0_ptb, propout):
        self.state0_ptb = state0_ptb
        self.propout = propout


# ---------------------------------------------------------------------------------------- #
# function for extracting eigenvectors
def _get_eigvecs_yu_ys(monodromy):
    """Function obtains unstable and stable eigenvectors from monodromy matrix

    Args:
        monodromy (np.array): 6x6 array of monodromy matrix

    Returns:
        (tuple): tuple of unstable and stable eigenvectors
    """

    eigvals, eigvecs = la.eig(monodromy)
    realeigval = []
    realeigvec = []

    for i in range(len(eigvals)):
        if np.imag(eigvals[i]) == 0 and np.abs(la.norm(eigvals[i]) - 1) > 1e-4:
            realeigval.append(eigvals[i])
            realeigvec.append(eigvecs[:, i])

    try:
        if realeigval[0] > realeigval[1]:
            lamu = realeigval[0]
            lams = realeigval[1]
            yu0 = np.real(realeigvec[0])  # unstable
            ys0 = np.real(realeigvec[1])  # stable
        else:
            lamu = realeigval[1]
            lams = realeigval[0]
            yu0 = np.real(realeigvec[1])  # unstable
            ys0 = np.real(realeigvec[0])  # stable

        # double-check norm (ensure these are unit vectors)
        if la.norm(yu0) != 1.0:
            yu0 = yu0 / la.norm(yu0)
        if la.norm(ys0) != 1.0:
            ys0 = ys0 / la.norm(ys0)

        return yu0, ys0
    except:
        return [], []


# ---------------------------------------------------------------------------------------- #
# function to scale epsilon, the linear perturbation term for constructing manifolds
def _scale_epsilon(cr3bp_param, stateP, period, yu0, ys0, monodromy, stable):
    """Function evaluates linear perturbation approriate to the LPO

    Choices include: perturbation_km_lst = np.array( [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0] )

    Args:
        cr3bp_param (CR3BP): CR3BP parameter
        stateP (np.array): state of periodic LPO
        period (float): period of LPO
        yu0 (np.array): unstable eigenvector at stateP
        ys0 (np.array): stable eigenvector at stateP
        monodromy (np.array): 6 by 6 array of monodromy matrix
        stable (bool): stable (True) or unstable (False) manifolds (default is False)

    Returns:
        (float): linear perturbation epsilon
    """
    # extract CR3BP arameters
    mu = cr3bp_param.mu
    lstar = cr3bp_param.lstar

    # tolerance
    relative_tol_manifold_lst = 0.1
    absolute_tol_manifold_km_lst = 100.0
    absolute_tol_manifold_lst = absolute_tol_manifold_km_lst / lstar

    # assign tolerance
    relTol = relative_tol_manifold_lst
    absTol = absolute_tol_manifold_lst

    # list of perturbation sizes to try
    perturbation_km_lst = np.array(
        [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    )
    perturbation_lst = perturbation_km_lst / lstar

    # compute eigenvectors error after 1 rev using STM
    if stable == True:  # stable case
        ef = np.dot(la.inv(monodromy), ys0)
    else:  # unstbale case
        ef = np.dot(monodromy, yu0)
    # compute position error
    efPosition = la.norm(ef[0:3])

    # initialise epsilon with smallest epsilon
    eps = perturbation_lst[0]

    for epsilon in perturbation_lst:
        # predicted error based on STM
        predictedError = epsilon * efPosition

        # compute error by propagating one period
        if stable == True:  # stable case
            x0_manifold = stateP + epsilon * ys0
            prop_manif_out = propagate_cr3bp(cr3bp_param, x0_manifold, -period)
        else:  # unstable case
            x0_manifold = stateP + epsilon * yu0
            prop_manif_out = propagate_cr3bp(cr3bp_param, x0_manifold, period)

        # actual position error
        propPositionError = la.norm(prop_manif_out.statef[0:3] - stateP[0:3])

        # break for-loop if allowed tolerance is broken
        if (
            np.abs((propPositionError - predictedError) / propPositionError) > relTol
            and np.abs(propPositionError - predictedError) > absTol
        ):
            break

        # else store solution largest epsilon allowed
        eps = epsilon

    return eps


# ---------------------------------------------------------------------------------------- #
# function to generate branch of manifold
def _get_branch(
    cr3bp_param,
    state0,
    eigvec,
    eps,
    tf_manif,
    events=None,
    manif_steps=2000,
    ivp_method="LSODA",
    force_solve_ivp=False,
    verbose=False,
):
    """Function wraps propagate_cr3bp to generate branch of manifold, called internally in get_manifolds()

    Args:
        mu (float): CR3BP mu
        state0 (np.array): numpy array of initial state for the branch, along the halo (i.e. unperturbed)
        eigvec (np.array): stable or unstable eigenvector
        eps0 (float): magnitude of linear perturbation to be applied
        tf_manif (float): final propagation time of manifold; should be negative if stable, positive if unstable branch
        detect_rp_m2 (bool): whether to detect the periapses of the branch (default is True)
        stopAtPeriapsis (bool): whether to stop at first occurence of the periapsis (default is False)
        verbose (bool): verbosity flag

    Returns:
        (tuple): tuple of np.array of perturbed state at the root of the manifold, and output dictionary from propagate_cr3bp
    """
    # generate perturbed states and propagate
    ptb_state0 = state0 + eps * eigvec
    propout = propagate_cr3bp(
        cr3bp_param,
        ptb_state0,
        tf_manif,
        steps=manif_steps,
        ivp_method=ivp_method,
        events=events,
        ivp_rtol=1e-12,
        ivp_atol=1e-12,
        force_solve_ivp=force_solve_ivp,
    )
    return ptb_state0, propout


# ---------------------------------------------------------------------------------------- #
# function to get manifolds
def get_manifold(
    cr3bp_param,
    stateP,
    period,
    tf_manif,
    num_branches=50,
    eps=None,
    stable=True,
    events=None,
    manif_steps=2000,
    ivp_method="LSODA",
    full_output=False,
    force_solve_ivp=False,
    verbose=False,
):
    """Function generates and returns manifolds

    Args:
        cr3bp_param (R3BP.Parameters): CR3BP system parameters
        stateP (np.array): state of periodic motion
        period (float): period of periodic motion
        tf_manif (float): propagation time of manifold, always passed as positive value
        lstar (float): canonical unit length scale of the CR3BP system, used for scaling eps automatically
        num_branches (int): number of manifold branches
        eps (float): linear perturbation, if set to None the function determines appropriate eps based on stability of periodic motion
        stable (bool): nature of manifold branch to be obtained, True for stable manifold, False for unstable manifold
        events (fun): event function handle used to terminate manifold propagation; must be of the form fun(t, state, mu)
        manif_steps (int): number of steps used for propagating manifold
        ivp_method (str): integrator used for propagating manifold
        full_output (bool): whether to return full information of the generated manifold
        force_solve_ivp (bool): whether to force solve_ivp() for generating manifolds
        verbose (bool): verbosity flag

    Returns:
        (tuple): plus-direction and minus-direction manifold branches
    """
    assert tf_manif >= 0.0

    # extract CR3BP arameters
    mu = cr3bp_param.mu
    lstar = cr3bp_param.lstar

    # propagate result for plotting along with stm (need stm!)
    propout = propagate_cr3bp(
        cr3bp_param,
        stateP,
        period,
        stm_option=True,
        steps=4000,
        ivp_method=ivp_method,
        ivp_rtol=1e-12,
        ivp_atol=1e-12,
        force_solve_ivp=force_solve_ivp,
    )
    monodromy = np.reshape(propout.stms[:, -1], (6, 6))
    # extract the stable and unstable eigenvectors
    yu0, ys0 = _get_eigvecs_yu_ys(monodromy)
    if verbose:
        print(f"monodromy = {monodromy}")
        print(f"yu0 = {yu0}")
        print(f"ys0 = {ys0}")

    # linear perturbation
    if eps == None:
        eps = _scale_epsilon(cr3bp_param, stateP, period, yu0, ys0, monodromy, stable)
    if verbose:
        print(f"eps = {eps}")

    # get interval between the manifolds as number of points to skip
    interval_manif = propout.times.size / (num_branches)

    # initialize list to store manifolds
    manifolds_pls = Manifold(mu, monodromy, stable, eps)
    manifolds_min = Manifold(mu, monodromy, stable, eps)

    for i in tqdm(range(num_branches), desc="Manifold", leave=False):
        # compute index to extract state
        extractind = int(np.round(i * interval_manif))
        # state at current index
        state0 = propout.state_at_index(extractind)
        # np.array([propout.xs[extractind],  propout.ys[extractind],  propout.zs[extractind],
        #                   propout.vxs[extractind], propout.vys[extractind], propout.vzs[extractind]])
        # STM at current index
        stm = np.reshape(propout.stms[:, extractind], (6, 6))

        # initialize branch dictionary with index to extract halo state, state, STM, and the time from t=0 at which the branch stems, eps
        branch_pls = Branch(
            mu,
            eps0=eps,
            state0=state0,
            stm=stm,
            timeAlongHalo=propout.times[extractind],
            extractind=extractind,
        )
        # { "extractind": extractind, "state0": state0, "STM": stm, "timeAlongHalo": propout.times[extractind], "eps0":  eps }
        branch_min = Branch(
            mu,
            eps0=-eps,
            state0=state0,
            stm=stm,
            timeAlongHalo=propout.times[extractind],
            extractind=extractind,
        )
        # { "extractind": extractind, "state0": state0, "STM": stm, "timeAlongHalo": propout.times[extractind], "eps0": -eps }

        # translate eigenvectors to current position and construct branch
        if stable == True:
            # store propagation time
            # branch_pls["tf"] = -tf_manif
            # branch_min["tf"] = -tf_manif
            # translate stable eigenvector
            ys = np.dot(stm, ys0) / la.norm(np.dot(stm, ys0))
            # call function to propagate branch
            ptb_state0_pls, propout_pls = _get_branch(
                cr3bp_param=cr3bp_param,
                state0=state0,
                eigvec=ys,
                eps=eps,
                tf_manif=-tf_manif,
                events=events,
                manif_steps=manif_steps,
                ivp_method=ivp_method,
                force_solve_ivp=force_solve_ivp,
            )
            ptb_state0_min, propout_min = _get_branch(
                cr3bp_param=cr3bp_param,
                state0=state0,
                eigvec=ys,
                eps=-eps,
                tf_manif=-tf_manif,
                events=events,
                manif_steps=manif_steps,
                ivp_method=ivp_method,
                force_solve_ivp=force_solve_ivp,
            )
            # store eigenvectors
            branch_pls.set_eigenvectors(y0=ys0, y=ys, tf_manif=-tf_manif)
            branch_min.set_eigenvectors(y0=ys0, y=ys, tf_manif=-tf_manif)
            # branch_pls["ys0"] = ys0
            # branch_pls.ys  = ys
            # branch_min["ys0"] = ys0
            # branch_min.ys  = ys
        else:
            # store propagation time
            # branch_pls["tf"] = tf_manif
            # branch_min["tf"] = tf_manif
            # translate unstable eigenvector
            yu = np.dot(stm, yu0) / la.norm(np.dot(stm, yu0))
            # call function to propagate branch
            ptb_state0_pls, propout_pls = _get_branch(
                cr3bp_param=cr3bp_param,
                state0=state0,
                eigvec=yu,
                eps=eps,
                tf_manif=tf_manif,
                events=events,
                manif_steps=manif_steps,
                ivp_method=ivp_method,
                force_solve_ivp=force_solve_ivp,
            )
            ptb_state0_min, propout_min = _get_branch(
                cr3bp_param=cr3bp_param,
                state0=state0,
                eigvec=yu,
                eps=-eps,
                tf_manif=tf_manif,
                events=events,
                manif_steps=manif_steps,
                ivp_method=ivp_method,
                force_solve_ivp=force_solve_ivp,
            )
            # store eigenvectors
            branch_pls.set_eigenvectors(y0=yu0, y=yu, tf_manif=tf_manif)
            branch_min.set_eigenvectors(y0=yu0, y=yu, tf_manif=tf_manif)
            # # store eigenvectors
            # branch_pls["yu0"] = yu0
            # branch_pls["yu"]  = yu
            # branch_min["yu0"] = yu0
            # branch_min["yu"]  = yu

        # append these to branch dictionary
        branch_pls.set_branch_propagation(ptb_state0_pls, propout_pls)
        branch_min.set_branch_propagation(ptb_state0_min, propout_min)
        # branch_pls["ptb_state0"] = ptb_state0_pls
        # branch_min["ptb_state0"] = ptb_state0_min

        # # store to propagation output to branch
        # branch_pls["propout"] = propout_pls
        # branch_min["propout"] = propout_min

        # append branches to lst
        manifolds_pls.add_branch(branch_pls)
        manifolds_min.add_branch(branch_min)

    # categorize manifolds based on x-axis location of perturbed state and actual state
    if (
        manifolds_pls.branches[0].state0[0] < manifolds_min.branches[0].state0_ptb[0]
    ):  # branch_2 is perturbed in the exterior direction
        manifolds_min_out = manifolds_pls
        manifolds_pls_out = manifolds_min
    else:
        manifolds_min_out = manifolds_min
        manifolds_pls_out = manifolds_pls
    # return manifold
    return manifolds_pls_out, manifolds_min_out
