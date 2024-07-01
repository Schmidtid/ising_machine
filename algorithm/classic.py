"""A simple wrapper to call classical solvers"""
import numpy as np
from dwave.samplers.sa.simulated_annealing import simulated_annealing
from dwave.samplers.tabu import TabuSearch


def sa(matrix: np.ndarray,
       n_iter: int = 20,
       n_sweeps: int = 10000,
       n_sweeps_per_beta: int = 1000,
       initial_state: np.ndarray = None,
       seed: int = None,
       **params):
    """A simple wrapper to call simulated annealing from dwave.samplers

    Parameters
    ----------
        matrix (np.ndarray): Ising matrix
        n_iter (int, optional): Number of algorithm runs. Defaults to 20.
        n_sweeps (int, optional): Number of attempts to flip variables for the entire algorithm. \
            Defaults to 10_000.
        n_sweeps_per_beta (int, optional): Number of sweeps for each temperature in schedule. \
            Defaults to 1000.
        initial_state (np.ndarray, optional): Initial solution vector. Defaults to None.
        seed (int, optional): 32-bit unsigned integer seed
        schedule (str, optioanal): interpolation type for reverse temperatures
        criteria (str, optional): Acceptance criteria: `Metropolis` or `Gibbs`
        random_order (bool, optioanal): When `True`, each spin update selects \
            a variable uniformly at random. This method is ergodic, \
            obeys detailed balance and preserves symmetries of the model. Defaults to False.

    Returns
    ----------
        tuple[np.ndarray]: binary solution vectors and corresponding energies
    """
    seed = np.random.randint(0, 2**31) if seed is None else np.clip(seed, 0, 2**31)
    randomize_order = False if params.get('random_order') is None else params.get('random_order')
    criteria = ('Metropolis'
                if params.get('criteria') not in ['Metropolis', 'Gibbs'] else
                params.get('criteria'))
    quadratic = matrix[:-1, :-1]
    linear = matrix[:-1, -1]
    row,col = np.where(np.triu(quadratic, k=1).T)
    data = quadratic.T[(row,col)]
    num_betas = n_sweeps//n_sweeps_per_beta
    beta_schedule = (np.linspace(*get_beta(quadratic, linear), num_betas)
                     if params.get('schedule') == 'linear' else
                     np.geomspace(*get_beta(quadratic, linear), num_betas))
    if initial_state is None:
        initial_states = np.empty((0, len(linear)), dtype=np.int8)
        rnd = np.random.default_rng(seed)
        values = np.array([-1,1], dtype=np.int8)
        initial_states = rnd.choice(values, size=(n_iter, len(linear)))
    else:
        initial_states = np.tile(initial_state.reshape(1, -1), (n_iter, 1))
    interrupt_function = params.get('interrupt')
    interrupt_function = None if not callable(interrupt_function) else interrupt_function
    samples, energies = simulated_annealing(
            n_iter, linear, row, col, data,
            n_sweeps_per_beta, beta_schedule,
            seed, initial_states,
            randomize_order, criteria,
            interrupt_function)
    return samples, energies

def get_beta(J, h):
    """Determine the starting and ending reverse temperature for \
    Simulated Annealing. Implementation `neal._default_ising_beta_range`

    Parameters
    ----------
        J (np.ndarray): Couplings of Ising matrix
        h (np.ndarray): External field

    Returns:
    ----------
        tuple[float]: the starting and ending reverse temperature
    """
    a = np.diag(np.abs(h)) + np.abs(J)
    min_abs = np.minimum(np.min(np.where(a==0, a.max(), a), axis=0),
                        np.min(np.where(a==0, a.max(), a), axis=1))
    sum_abs = a.sum(0) + a.sum(1) - np.abs(h)
    max_field = sum_abs.max()
    if max_field == 0:
        hot_beta = 1
    else:
        hot_beta = np.log(2) / (2*max_field)
    cold_beta = np.log(100) / (2*min_abs.min())
    return [hot_beta, cold_beta]

def ts(matrix: np.ndarray,
       n_iter: int = 10,
       tenure: int = None,
       seed: int = None,
       num_restarts: int = 1,
       **params):
    """A simple wrapper to call tabu search from dwave.samplers
    
    Parameters
    ----------
        matrix (np.ndarray): Ising matrix
        n_iter (int, optional): Number of algorithm runs. Defaults to 20.
        tenure (int, optional): Tabu tenure, which is the length of the tabu list, \
            or number of recently explored solutions kept in memory. \
            Default is a quarter of the number of problem variables up to \
            a maximum value of 20.
        seed (int, optional): 32-bit unsigned integer seed to use for the PRNG. \
            If the ``timeout`` parameter is not None, results from the same seed may not be \
            identical between runs due to finite clock resolution.
        num_restarts (int, optional): Number of search restarts per run.
        timeout (int, optional): Maximum running time per read in milliseconds.
        lb_z (int, optional): Sets a minimum number of variable updates on all algorithm. \
            The bound defaults to 500000.
        z_1: :code:`max(len(matrix)*z_1, lb_z)` bounds the number of variable updates \
            considered in the first simple tabu search (STS).
        z_2 (int, optional): Controls the number of variable updates \
            on restarted simple tabu search stages

    Returns:
    ----------
        tuple[np.ndarray]: binary solution vectors and corresponding energies
    """

    symm = (matrix + matrix.T)/2
    seed = np.random.randint(0, 2**31) if seed is None else np.clip(seed, 0, 2**31)
    tenure = tenure or len(matrix) - 1
    rng = np.random.default_rng(seed)
    samples = np.empty((n_iter, len(matrix)), dtype=np.int8)
    best_en = np.zeros(n_iter)
    timeout = -1 if params.get('timeout') is None else abs(params.get('timeout'))
    z_1, z_2, lb_z = params.get('z_1'), params.get('z_2'), params.get('lb_z')
    for run in range(n_iter):
        initial_state = np.random.choice([0, 1], len(matrix)).astype(np.int8)
        seed_per_read = rng.integers(0, 2**32, dtype=np.uint32)
        r = TabuSearch(symm,
                       initial_state,
                       tenure,
                       timeout,
                       num_restarts,
                       seed_per_read, None, 
                       z_1, z_2, lb_z)
        samples[run] = np.asarray(r.bestSolution())
        best_en[run] = r.bestEnergy()
    return samples, best_en