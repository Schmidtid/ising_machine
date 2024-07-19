"""IM Workflow"""
import time
import numpy as np
import torch as tr
from tqdm import tqdm
from scipy.optimize import brute
import scipy.stats as sp
import matplotlib.pyplot as plt
from ising_machine.im.utils import qubo_ising, ising_qubo
from ising_machine.algorithm.classic import sa, ts

plt.rcParams['font.size'] = 16
plt.style.use('bmh')


class Solver:
    """IM model to solving CO problem"""
    def __init__(self,
                 matrix: np.ndarray,
                 h: np.ndarray = None,
                 values='binary',
                 scale: float = None) -> None:
        """
        Parameters
        ----------
            matrix (np.ndarray): QUBO or Ising spin-spin iteraction matrix \
                in upper triangle or symmetric form 
            h (np.ndarray): magnetic fields in Ising form
            values (str, optional): decision variables passed to the solver. Defaults to 'binary'.
            scale (float, optional): Scaling by maximum values. Defaults to None.
        """
        if not np.count_nonzero(matrix)==np.count_nonzero(np.triu(matrix)):
            matrix -= np.triu(matrix,k=1).T - np.triu(matrix,k=1)
        self.type = -1
        if values == 'spin':
            matrix = ising_qubo(matrix, h)
            self.type = 0
        self.matrix = qubo_ising(matrix, a_ising=True, scale=scale)
        self.x = np.zeros_like(matrix.shape[0])
        self.energy = 0
        print(self)
        self.cim_param = {
            'alpha_0':0, 'alpha_1':1,
            'beta_0':None, 'beta_1':None,
            'a_grow':0, 'a_end':None,
            'b_start_grow':None, 'b_end_grow':None,
            'b_start_decr':None, 'b_end_decr':None,
            'b_decr_0':0, 'b_decr_1':None,
            'delta_alpha':None, 'delta_beta':None
            }
        self.runs = np.empty(0)
        self.traj = np.empty(0)

    def __repr__(self) -> str:
        non_zero = np.count_nonzero(self.matrix[:-1, :-1])
        return (f"QUBO with {len(self.matrix) - 1} nodes," +
              f'{non_zero} edges,' +
              f'{(non_zero / ((self.matrix.shape[0] - 1) * (self.matrix.shape[0] - 2) / 2)):.4f}' +
              ' density')

    def _default_params(self,**kward):
        self.cim_param = {
            'alpha_0':0, 'alpha_1':1,
            'beta_0':0, 'beta_1': 0.4 / np.log(self.matrix.size),
            'a_grow':0, 'a_end':0.083,
            'b_start_grow':0.166, 'b_end_grow':0.416,
            'b_start_decr':0.416, 'b_end_decr':0.583,
            'b_decr_0':0, 'b_decr_1':0,
            'delta_alpha':6e-3, 'delta_beta':6e-4
            }
        for key,value in kward.items():
            try:
                self.cim_param[key] = value
            except LookupError:
                print(f'Parameter {key} not found')
                continue

    def tabu_search(self,
                   n_iter=10,
                   seed = 1,
                   tenure: int =None,
                   timeout = 100,
                   num_restarts: int = 10,
                   lower_bound_z= None,
                   coefficient_z_first = None,
                   coefficient_z_restart = None
                   ):
        """
        Parameters
        ----------
            n_iter (int, optional): Amount of runs of the entire algorithm.
            seed (int, optional): 32-bit unsigned integer seed to use for the PRNG. \
                If the ``timeout`` parameter is not None, results from the same seed may not be \
                identical between runs due to finite clock resolution.
            tenure (int, optional): Tabu tenure, which is the length of the tabu list, \
                or number of recently explored solutions kept in memory. \
                Default is a quarter of the number of problem variables up to \
                a maximum value of 20.
            timeout (int, optional): Maximum running time per read in milliseconds.
            num_restarts (int, optional): Number of search restarts per run.
            lower_bound_z (int, optional): Sets a minimum number of variable updates \
                on whole algorithm. The bound defaults to 500000.
            coefficient_z_first: :code:`max(n*coefficient_z_first, lower_bound_z)` \
                bounds the number of variable updates considered in the first tabu search. \
                Variable updates ainterp from the STS greedy-descent subroutine, \
                invoked upon discovery of new global optima, are excluded from the count. \
                The coefficient defaults to 10000 for small problems (<= 500 variables) \
                and 25000 for larger problems.
            coefficient_z_restart (int, optional): Controls the number of variable updates \
                on restarted simple tabu search stages. Defaults to :code:`coefficient_z_first/4`

        Returns
        ----------
            (np.ndarray): binary solution vector
        """
        matrix = ising_qubo(self.matrix[:-1, :-1], self.matrix[:-1, -1])
        start = time.time()
        response = ts(matrix,
                      n_iter,
                      tenure,
                      seed,
                      num_restarts,
                      timeout=timeout,
                      z_1=coefficient_z_first,
                      z_2=coefficient_z_restart,
                      lb_z=lower_bound_z)
        end = time.time()
        self.x = (response[0][response[1].argmin()] * 2) - 1
        self.energy = response[1].min()
        self.runs = response[1]
        print(f'Total time: {end-start:.6} s')
        print(f'QUBO energy {self.energy}')
        if self.type == 0:
            return self.x
        return (self.x+1)/2

    def simulated_annealing(self,
                            n_iter=20,
                            n_sweeps=10_000,
                            n_sweeps_per_beta=1000,
                            initial_state = None,
                            seed: int = None,
                            schedule_type: str = None,
                            criteria:str = None,
                            random_order: bool = False):
        """Implementation of simulated annealing algorithm

        Parameters
        ----------
            n_iter (int, optional): Amount of algorithm runs.
            n_sweeps (int,optional): The number of sweeps used per run annealing pass.
            n_sweeps_per_beta (int,optional): Number of sweeps for each temperature in schedule.
            initial_state (np.ndarray, optional): Initial solution vector. Defaults to None.
            seed (int, optional): 32-bit unsigned integer seed
            schedule_type (str, optional): interpolation type for reverse temperatures. \
                if `None` used geometric interpolation, also can be `linear`
            criteria (str, optional): Acceptance criteria: `Metropolis` or `Gibbs`
            random_order (bool, optioanal): When `True`, each spin update selects \
                a variable uniformly at random. This method is ergodic, \
                obeys detailed balance and preserves symmetries of the model. Defaults to False.

        Returns
        ----------
            (np.ndarray): binary solution vector
        """
        start = time.time()
        response = sa(self.matrix,
                      n_iter,
                      n_sweeps,
                      n_sweeps_per_beta,
                      initial_state,
                      seed = seed,
                      schedule = schedule_type,
                      criteria = criteria,
                      random_order = random_order)
        end = time.time()
        self.x = response[0][response[1].argmin()]
        self.energy = response[1].min()
        self.runs = response[1]
        print(f'Total time: {end-start:.6} s')
        print(f"Ising energy: {self.energy}")
        if self.type == 0:
            return self.x
        return (self.x+1)/2

    @staticmethod
    def interp(min_p, max_p, start, stop, k):
        """Interpolation map schedulling for parameters in linear space

        Parameters
        ----------
            min_p (float): The starting value of the sequence.
            max_p (float): The end value of the sequence
            start (float): The step number to start growing
            stop (float): The last step before the end of growth
            k (int): Current iteration step

        Returns
        ----------
            float: value of calculated parameters over the interval [`min_p`, `max_p`].
        """
        if k <= start:
            return min_p
        if k <= stop:
            return ((max_p - min_p) / (stop - start) * k
                    + (max_p * start - min_p * stop) / (start - stop))
        return max_p

    @staticmethod
    def noise_psd(n, psd = lambda f: f**2):
        """Noise generator

        Parameters
        ----------
            N (int): Sample length
            psd (func, optional): Function to generate power spectrum. Defaults to np.square

        Returns
        ----------
            np.ndarray: normolized white noise
        """
        x_white = np.fft.rfft(0.02*np.random.rand(n))
        S = psd(np.fft.rfftfreq(n))
        S = S / np.sqrt(np.mean(S**2))
        # Shape white noise with power spectrum
        x_shaped = x_white * S
        return np.fft.irfft(x_shaped)

    def get_obj(self,
                x: np.ndarray = None):
        """
        Parameters
        ----------
            x (np.ndarray, optional): solution binary vector

        Returns
        ----------
            float: QUBO energy value
        """
        J = self.matrix + self.matrix.T
        x = self.x if x is None else x
        if np.isin(x, np.array([0,1])).all():
            x = (x*2)/1
        energy = x@((J[:-1, :-1])/2)@x + J[:-1, -1]@x
        x = (x+1)/2
        print(f'Ising {energy=}')
        energy = x.dot(x.dot(ising_qubo(self.matrix[:-1, :-1], self.matrix[:-1, -1])))
        print(f'Qubo {energy=}')
        return energy

    def mzm_ising(self, n_iter=1000, min_eigv=None,opt=False, n_runs=1, device='cpu'):
        """Simulation of analog Ising machine based on Mach-Zender modulator

        Parameters
        ----------
            n_iter (int, optional): Amount of iteration. Defaults to 1000.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            opt (bool, optional): If `True` return energy history.
            n_runs (int, optional): Amount of runs of the entire simulation. Defaults to 1.
            device (str, optional): Defaults to 'cpu'.

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector"""
        J = self.matrix + self.matrix.T
        scale = np.abs(np.linalg.eigvals(J).min()/min_eigv) if min_eigv else 1
        J_norm = tr.from_numpy( J / scale).float().to(device)
        J = tr.from_numpy(J).float().to(device)
        a_0 = self.cim_param['alpha_0'] or 0
        a_1 = self.cim_param['alpha_1'] or 1
        b_0 = self.cim_param['beta_0'] or 0.0 / np.log(J.shape[0])
        b_1 = self.cim_param['beta_1'] or 0.2 / np.log(J.shape[0])
        a_thr0 = self.cim_param['a_grow'] or 0
        a_thr1 = round((self.cim_param['a_end'] or 0.083)*n_iter)
        b_thr0 = round((self.cim_param['b_start_grow'] or 0.166)*n_iter)
        b_thr1 = round((self.cim_param['b_end_grow'] or 0.416)*n_iter)
        b_thr0f = round((self.cim_param['b_start_decr'] or 0.416)*n_iter)
        b_thr1f = round((self.cim_param['b_end_decr'] or 0.583)*n_iter)
        b_0f = self.cim_param['b_decr_0'] or 0
        b_1f = self.cim_param['b_decr_1'] or -0.0 / np.log(J.shape[0])
        energy = tr.zeros(n_iter).to(device)
        traj = tr.zeros((n_iter, J.shape[0])).to(device)
        best_en = tr.zeros(n_runs).to(device)
        best_energy = 0
        start = time.time()
        for run in range(n_runs):
            noise = 0.005*tr.randn(J.shape[0], n_iter).float().to(device)
            x = 0.02*tr.randn(J.shape[0]).float().to(device)
            traj[0] = x
            sig = tr.sign(x)
            sig = sig[-1] * sig[:-1]
            energy[0] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
            for i in range(1, n_iter):
                x = tr.pow(tr.cos(x * self.interp(a_0, a_1, a_thr0, a_thr1, i)
                                  + noise[:, i]
                                  + (-(self.interp(b_0, b_1, b_thr0, b_thr1, i) +
                                       self.interp(b_0f, b_1f, b_thr0f, b_thr1f, i))
                                       * tr.matmul(J_norm, x) - tr.pi/4)), 2) - 0.5
                traj[i] = x
                sig = tr.sign(x)
                sig = sig[-1] * sig[:-1]
                energy[i] = sig@((J[:-1, :-1])/2)@sig + tr.matmul(J[:-1, -1], sig)
            best_en[run] = energy.min()
            if best_energy > energy.min().item():
                self.x = (tr.sign(traj[energy.argmin()])[:-1].cpu().numpy()
                          * tr.sign(traj[energy.argmin()][-1]).cpu().numpy())
                best_energy = energy.min().item()
        end = time.time()
        print(f'Total time: {end-start:.6}')
        if (energy[-3:] != energy.min()).all():
            print(f'Last energy {energy[-1]}, Min energy {energy.min()}')
        print(f'Min energy {best_en.argmin()} run, value: {best_en.min()}')
        self.traj = traj
        self.energy = energy
        self.runs = best_en.cpu().numpy()
        if opt:
            return energy.cpu().numpy()
        if self.type == 0:
            return self.x
        return (self.x+1)/2

    def clip_fb(self, n_iter=1000, min_eigv=None,
               opt=False, n_runs=1, device='cpu',h_max=100):
        """Simulation of analog Ising machine based on optoelectronic modulator with limeted \
            feedback function

        Parameters
        ----------
            n_iter (int, optional): Amount of iteration. Defaults to 1000.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            opt (bool, optional): If `True` return energy history.
            n_runs (int, optional): Amount of runs of the entire simulation. Defaults to 1.
            device (str, optional): Defaults to 'cpu'.
            h_max (float, optional): Maximum absolute value of feedback function. Defaults to 100.

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector
        """
        J = self.matrix + self.matrix.T
        scale = np.abs(np.linalg.eigvals(J).min()/min_eigv) if min_eigv else 1.0
        J_norm = tr.from_numpy( J / scale).float().to(device)
        J = tr.from_numpy(J).float().to(device)
        if any(i is None for i in self.cim_param.values()):
            self._default_params()
        a_0, a_1 = self.cim_param['alpha_0'], self.cim_param['alpha_1']
        b_0, b_1 = self.cim_param['beta_0'], self.cim_param['beta_1']
        a_gr0, a_gr1 = self.cim_param['a_grow'], int(self.cim_param['a_end']*n_iter)
        b_gr0 = int(self.cim_param['b_start_grow']*n_iter)
        b_gr1 = int(self.cim_param['b_end_grow']*n_iter)
        b_f0, b_f1 = self.cim_param['b_decr_0'], self.cim_param['b_decr_1']
        b_thr0 = round(self.cim_param['b_start_decr']*n_iter)
        b_thr1 = round(self.cim_param['b_end_decr']*n_iter)
        energy = tr.zeros(n_iter).to(device)
        traj = tr.zeros((n_iter, J.shape[0])).to(device)
        best_en = tr.zeros(n_runs).to(device)
        best_energy = 0
        start = time.time()
        for run in range(n_runs):
            noise = tr.normal(mean=tr.zeros((J.shape[0], n_iter)), std=0.01).to(device)
            x = 0.02*tr.randn(J.shape[0]).float().to(device)
            traj[0] = x
            sig = tr.sign(x)
            sig = sig[-1] * sig[:-1]
            energy[0] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
            for i in range(1, n_iter):
                x = tr.pow(tr.cos(x * self.interp(a_0, a_1, a_gr0, a_gr1, i)
                                  + noise[:, i]
                                  + tr.clip(-(self.interp(b_0, b_1, b_gr0, b_gr1, i) +
                                       self.interp(b_f0, b_f1, b_thr0, b_thr1, i))
                                       * tr.matmul(J_norm, x),-h_max, h_max) - tr.pi/4,), 2) - 0.5
                traj[i] = x
                sig = tr.sign(x)
                sig = sig[-1] * sig[:-1]
                energy[i] = sig@((J[:-1, :-1])/2)@sig + tr.matmul(J[:-1, -1], sig)
            best_en[run] = energy.min()
            if best_energy > energy.min().item():
                self.x = (tr.sign(traj[energy.argmin()])[:-1].cpu().numpy()
                          * tr.sign(traj[energy.argmin()][-1]).cpu().numpy())
                best_energy = energy.min().item()
        end = time.time()
        print(f'Total time: {end-start:.6}')
        if (energy[-3:] != energy.min()).all():
            print(f'Last energy {energy[-1]}, Min energy {energy.min()}')
        print(f'Min energy {best_en.argmin()} run, value: {best_en.min()}')
        self.traj = traj.cpu()
        self.energy = energy.cpu().numpy()
        self.runs = best_en.cpu().numpy()
        print(f'alpha={round(a_1, 6)},'
              + f' beta={round(b_1 - b_f1, 6)},'
              + f' scale: {scale:.4}, Min: {best_energy}')
        if opt:
            return energy
        if self.type == 0:
            return self.x
        return (self.x+1)/2

    def noise_cim(self,
                  n_iter: int = 1000,
                  n_runs: int = 1,
                  min_eigv=None,
                  opt=False,
                  device='cpu',
                  blue=True) -> np.ndarray:
        """Simulation of Noise Ising machine based on optoelectronic modulator
        
        Parameters
        ----------
            n_iter (int, optional): Amount of iteration. Defaults to 1000.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            opt (bool, optional): If `True` return energy history.
            n_runs (int, optional): Amount of runs of the entire simulation. Defaults to 1.
            device (str, optional): Defaults to 'cpu'.
            blue (bool, optional): If `True` use additional noise like `ifft(sqrt(PSD)*N(0,1))*k/n` \
                else laplace distribution.

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector
        """
        J = self.matrix + self.matrix.T
        scale = np.abs(np.linalg.eigvals(J).min())/min_eigv if min_eigv else 1.0
        J_norm = tr.from_numpy( J / scale).float().to(device)
        J = tr.from_numpy(J).float().to(device)
        if any(i is None for i in self.cim_param.values()):
            self._default_params(beta_0 = 0.0001 / np.log(J.shape[0]),
                                 beta_1 = 0.2 / np.log(J.shape[0]),
                                 b_decr_1 = -0.0001 / np.log(J.shape[0]))
        a_0, a_1 = self.cim_param['alpha_0'], self.cim_param['alpha_1']
        b_0, b_1 = self.cim_param['beta_0'], self.cim_param['beta_1']
        a_gr0, a_gr1 = self.cim_param['a_grow'], int(self.cim_param['a_end']*n_iter)
        b_gr0 = int(self.cim_param['b_start_grow']*n_iter)
        b_gr1 = int(self.cim_param['b_end_grow']*n_iter)
        b_f0, b_f1 = self.cim_param['b_decr_0'], self.cim_param['b_decr_1']
        b_thr0 = round(self.cim_param['b_start_decr']*n_iter)
        b_thr1 = round(self.cim_param['b_end_decr']*n_iter)
        energy = tr.zeros(n_iter).to(device)
        traj = tr.empty((n_iter, J.shape[0]), dtype=tr.float32).to(device)
        best_energy = 0
        best_en = tr.zeros(n_runs).to(device)
        start = time.time()
        for run in range(n_runs):
            noise = 0.005*tr.randn(J.shape[0], n_iter).float().to(device)
            noises = (tr.from_numpy(
                np.array(
                    [np.clip(self.noise_psd(n_iter, np.sqrt)*(i/J.shape[0]), -0.01, 0.01)
                     for i in range(J.shape[0])])).float().to(device)
                     if blue
                     else tr.from_numpy(
                         np.asarray(
                             [np.clip(sp.laplace.rvs(0,i*1e-4, size=n_iter), -0.015, 0.015)
                              for i in range(J.shape[0])])).float().to(device))
            x = (1e-6*tr.randint(-J.shape[0], J.shape[0], size=(J.shape[0],))).float().to(device)
            traj[0] = x
            for i in range(1, n_iter):
                x = tr.pow(tr.cos(x * self.interp(a_0, a_1, a_gr0, a_gr1, i)
                                  + noise[:, i]
                                  + noises[:, i]
                                  + (-(self.interp(b_0, b_1, b_gr0, b_gr1, i) +
                                       self.interp(b_f0, b_f1, b_thr0, b_thr1, i))
                                       * tr.matmul(J_norm, x) - tr.pi/4)), 2) - 0.5
                traj[i] = x
                sig = tr.sign(x)
                sig = sig[-1] * sig[:-1]
                energy[i] = sig@((J[:-1, :-1])/2)@sig + tr.matmul(J[:-1, -1], sig)
            best_en[run] = energy.min()
            if best_energy > energy.min().item():
                self.x = (tr.sign(traj[energy.argmin()])[:-1].cpu().numpy()
                          * tr.sign(traj[energy.argmin()][-1]).cpu().numpy())
                best_energy = energy.min().item()
        end = time.time()
        print(f'Up:{tr.sum(x > 0).cpu().numpy()}')
        print(f'Total time: {end-start:.6}')
        if (energy[-3:] == energy.min()).all():
            print(f'Min energy {energy[-1]}')
        else:
            print(f'Last energy {energy[-1]}, Min energy {energy.min()}')
        print(f'alpha={round(a_1, 6)},'
              + f' beta={round(b_1 - b_f1, 6)},'
              + f' scale: {scale:.4}, Min: {best_energy}')
        self.traj = traj
        if opt:
            return energy.cpu().numpy()
        self.energy = energy.cpu()
        self.runs = best_en.cpu().numpy()
        if self.type == 0:
            return self.x
        return (self.x+1)/2

    def analog_ising(self, n_iter=1000, min_eigv=None,opt=False, n_runs=1, device='cpu', clip=True):
        """Simulation of clip- and sigmoid- based Ising machine 

        Parameters
        ----------
            n_iter (int, optional): Amount of iteration. Defaults to 1000.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            opt (bool, optional): If `True` return energy history.
            n_runs (int, optional): Amount of runs of the entire simulation. Defaults to 1.
            device (str, optional): Defaults to 'cpu'.
            clip (bool, optional): if `True` then runs sigmoid-based else clip-based IM

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector
        """
        J = self.matrix + self.matrix.T
        scale = np.abs(np.linalg.eigvals(J).min()/min_eigv) if min_eigv else 1
        J_norm = tr.from_numpy( J / scale).float().to(device)
        J = tr.from_numpy(J).float().to(device)
        a_0 = self.cim_param['alpha_0'] or 0
        a_1 = self.cim_param['alpha_1'] or 1
        b_0 = self.cim_param['beta_0'] or 0.0 / np.log(J.shape[0])
        b_1 = self.cim_param['beta_1'] or 0.2 / np.log(J.shape[0])
        a_thr0 = self.cim_param['a_grow'] or 0
        a_thr1 = round((self.cim_param['a_end'] or 0.083)*n_iter)
        b_thr0 = round((self.cim_param['b_start_grow'] or 0.166)*n_iter)
        b_thr1 = round((self.cim_param['b_end_grow'] or 0.416)*n_iter)
        b_thr0f = round((self.cim_param['b_start_decr'] or 0.416)*n_iter)
        b_thr1f = round((self.cim_param['b_end_decr'] or 0.583)*n_iter)
        b_0f = self.cim_param['b_decr_0'] or 0
        b_1f = self.cim_param['b_decr_1'] or -0.0 / np.log(J.shape[0])
        energy = tr.zeros(n_iter).to(device)
        traj = tr.zeros((n_iter, J.shape[0])).to(device)
        best_en = tr.zeros(n_runs).to(device)
        best_energy = 0
        start = time.time()
        for run in range(n_runs):
            noise = 0.005*tr.randn(J.shape[0], n_iter).float().to(device)
            x = 0.02*tr.randn(J.shape[0]).float().to(device)
            traj[0] = x
            sig = tr.sign(x)
            sig = sig[-1] * sig[:-1]
            energy[0] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
            for i in range(1, n_iter):
                x = (tr.tanh(x * self.interp(a_0, a_1, a_thr0, a_thr1, i)
                            + noise[:, i]
                            + (-(self.interp(b_0, b_1, b_thr0, b_thr1, i)
                                + self.interp(b_0f, b_1f, b_thr0f, b_thr1f, i))
                                * tr.matmul(J_norm, x)))
                                if clip
                                else tr.clip((x * (self.interp(a_0, a_1, a_thr0, a_thr1, i))
                                              + noise[:, i]
                                              + (-(self.interp(b_0, b_1, b_thr0, b_thr1, i)
                                              + self.interp(b_0f, b_1f, b_thr0f, b_thr1f, i))
                                              * tr.matmul(J_norm, x))),-0.4,0.4))
                traj[i] = x
                sig = tr.sign(x)
                sig = sig[-1] * sig[:-1]
                energy[i] = sig@((J[:-1, :-1])/2)@sig + tr.matmul(J[:-1, -1], sig)
            best_en[run] = energy[-1]
            if best_energy > energy[-1].item():
                self.x = (tr.sign(traj[energy.argmin()])[:-1].cpu().numpy()
                          * tr.sign(traj[energy.argmin()][-1]).cpu().numpy())
                best_energy = energy[-1].item()
        end = time.time()
        print(f'Total time: {end-start:.6}')
        print(f'Min energy {best_en.argmin()} run, value: {best_en.min()}')
        self.traj = traj
        self.energy = energy
        self.runs = best_en.cpu().numpy()
        if opt:
            return energy
        if self.type == 0:
            return self.x
        return (self.x+1)/2

    def plot_traj(self):
        # plt.rcParams['font.family'] = 'Comic Sans MS, Times New Roman'
        """Plot energy and spin amplitude history"""
        try:
            ax = plt.subplots(2,1,figsize=(12,10))[1]
            ax[0].plot(self.energy)
            ax[0].set_title('Energy evolution')
            for i in range(len(self.traj.T)):
                ax[1].plot(self.traj[:,i])
            ax[1].set_title('Aplitude evolution')
            ax[1].set_xlabel('Iteration step')
            ax[1].set_ylabel('Spin amplitude')
            ax[0].set_ylabel('Hamiltonian value')
        except Exception:
            ax = plt.subplots(1,1,figsize=(12,10))[1]
            ax.plot(self.energy)
            ax.set_title('Energy evolution')
            ax.set_xlabel('Iteration step')
            ax.set_ylabel('Hamiltonian value')
        plt.show()

    def polinom(self, d_t=0.05, n_iter=1000, min_eigv=None,opt=False, n_runs=1, device='cpu'):
        """Optical like CIM

        Parameters
        ----------
            d_t (float, optional): time step. Defaults to 0.05.
            n_iter (int, optional): Amount of iteration. Defaults to 1000.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            opt (bool, optional): If `True` return energy history.
            n_runs (int, optional): Amount of runs of the entire simulation. Defaults to 1.
            device (str, optional): Defaults to 'cpu'.

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector
        """
        J = self.matrix + self.matrix.T
        scale = np.abs(np.linalg.eigvals(J).min()/min_eigv) if min_eigv else 1
        J_norm = tr.from_numpy( J / scale).float().to(device)
        J = tr.from_numpy(J).float().to(device)
        a_0 = self.cim_param['alpha_0'] or 0
        a_1 = self.cim_param['alpha_1'] or 1
        b_0 = self.cim_param['beta_0'] or 0.0 / np.log(J.shape[0])
        b_1 = self.cim_param['beta_1'] or 0.2 / np.log(J.shape[0])
        a_thr0 = self.cim_param['a_grow'] or 0
        a_thr1 = round((self.cim_param['a_end'] or 0.083)*n_iter)
        b_thr0 = round((self.cim_param['b_start_grow'] or 0.166)*n_iter)
        b_thr1 = round((self.cim_param['b_end_grow'] or 0.416)*n_iter)
        b_thr0f = round((self.cim_param['b_start_decr'] or 0.416)*n_iter)
        b_thr1f = round((self.cim_param['b_end_decr'] or 0.583)*n_iter)
        b_0f = self.cim_param['b_decr_0'] or 0
        b_1f = self.cim_param['b_decr_1'] or -0.0 / np.log(J.shape[0])
        energy = tr.zeros(n_iter).to(device)
        traj = tr.zeros((n_iter, J.shape[0])).to(device)
        best_en = tr.zeros(n_runs).to(device)
        best_energy = 0
        start = time.time()
        for run in range(n_runs):
            noise = 0.005*tr.randn(J.shape[0], n_iter).float().to(device)
            x = 0.02*tr.randn(J.shape[0]).float().to(device)
            traj[0] = x
            sig = tr.sign(x)
            sig = sig[-1] * sig[:-1]
            energy[0] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
            for t in range(1, n_iter):
                alpha = self.interp(a_0, a_1, a_thr0, a_thr1, t)
                beta = -(self.interp(b_0, b_1, b_thr0, b_thr1, t)
                         + self.interp(b_0f, b_1f, b_thr0f, b_thr1f, t))
                x = d_t*(x*alpha - x**3 + noise[:, t] + beta*tr.matmul(J_norm, x))
                traj[t] = x
                sig = tr.sign(x)
                sig = sig[-1] * sig[:-1]
                energy[t] = sig@((J[:-1, :-1])/2)@sig + tr.matmul(J[:-1, -1], sig)
            best_en[run] = energy[-1]
            if best_energy > energy[-1].item():
                self.x = (tr.sign(traj[energy.argmin()])[:-1].cpu().numpy()
                          * tr.sign(traj[energy.argmin()][-1]).cpu().numpy())
                best_energy = energy[-1].item()
        end = time.time()
        print(f'Total time: {end-start:.6}')
        print(f'Min energy {best_en.argmin()} run, value: {best_en.min()}')
        self.traj = traj
        self.energy = energy
        self.runs = best_en.cpu().numpy()
        if opt:
            return energy
        if self.type == 0:
            return self.x
        return (self.x+1)/2


class IMOpt(Solver):
    """Bruto-force like search of Optoelectronic IM params"""
    def __init__(self, matrix: np.ndarray,
                 h: np.ndarray = None, values='binary', scale: float = None) -> None:
        super().__init__(matrix, h, values, scale)
        self.all_traj = np.empty(0)

    def cim_poor(self, alpha, beta, n_iter=1000, min_eigv=None, opt=False):
        """Simulation Ideal Ising machine based on Mach-Zender modulator with single parameters

        Parameters
        ----------
            alpha (float): strenght of feedback-function
            beta (float): strenght of spin-spin matrix iteraction
            n_iter (int, optional): Amount of iteration. Defaults to 1000.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            opt (bool, optional): If `True` return energy history.

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector
        """
        J = self.matrix + self.matrix.T
        scale = np.linalg.eigvals(J).min()/min_eigv if min_eigv else 1.0
        J_norm = tr.from_numpy( J / scale).float()
        J = tr.from_numpy(J).float()
        energy = tr.zeros(n_iter)
        noise = tr.normal(mean=tr.zeros((J.shape[0], n_iter)), std=0.01)
        # x = (1e-6*tr.randint(-J.shape[0], J.shape[0], size=(J.shape[0],))).float()
        x = 0.02*tr.randn(J.shape[0]).float()
        sig = tr.sign(x).ravel()
        sig = sig[-1] * sig[:-1]
        energy[0] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
        traj = tr.zeros((n_iter, J.shape[0]))
        for t in range(1, n_iter):
            x = tr.pow(tr.cos(x * alpha
                              + noise[:, t]
                              + (beta * tr.matmul(J_norm, x) - tr.pi/4)), 2) - 0.5
            sig = tr.sign(x)
            sig = sig[-1] * sig[:-1]
            traj[t] = x
            energy[t] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
        print(f'Up:{tr.sum(x > 0).numpy()}')
        if (energy[-3:] == energy.min()).all():
            print(f'alpha={round(alpha,6)},beta={round(beta,6)}')
            print(f'Min energy {energy[-1]}')
        else:
            print(f'Not work {energy[-1]}')
            print(f'alpha={round(alpha,6)},beta={round(beta,6)}')
        self.x = (tr.sign(traj[energy.argmin()])[:-1].numpy()
                  * tr.sign(traj[energy.argmin()][-1]).numpy())
        self.traj = traj
        self.energy = energy
        if opt:
            return energy.numpy()
        if self.type == 0:
            return self.x
        return (sig.numpy()+1)/2

    def fast_search(self,alpha = 0.5,
                    beta = 0,
                    device='cpu',
                    min_eigv=None,
                    max_iter = 30000,
                    opt=False):
        """Heuristic search resonable `cim_poor` parameters

        Parameters
        ----------
            alpha (float, optional): Start point to increase strenght of feedback-function. \
                Defaults to 0.5.
            beta (int, optional): Start point to strenght of spin-spin matrix iteraction. \
                Defaults to 0.
            device (str, optional): Defaults to 'cpu'.
            min_eigv (float, optional): Minimum allowed eigen value for matrix scaling.
            max_iter (int, optional): Amount of iteration. Defaults to 30000.
            opt (bool, optional): If `True` return energy history.

        Returns
        ----------
            np.ndarray: enrgy history | solution binary vector
        """
        J = self.matrix + self.matrix.T
        scale = np.linalg.eigvals(J).min()/min_eigv if min_eigv else 1
        J_norm = tr.from_numpy( J / scale).float().to(device)
        J = tr.from_numpy(J).float().to(device)
        b = beta
        N = J_norm.shape[0]
        k = T_time = max_iter
        traj = tr.zeros((T_time, J.shape[0]))
        energy_plot_data = tr.zeros(T_time).to(device)
        x_ = (1e-6*tr.randint(-N, N, size=(N,))).float().ravel().to(device)
        x_ = 0.02*tr.randn(J.shape[0]).float().to(device)
        noise = tr.normal(mean=tr.zeros((N, T_time)), std=0.01).to(device)
        history_a, history_b = np.zeros(T_time), np.zeros(T_time)
        delta_a = self.cim_param['delta_alpha'] or 6e-3
        delta_b = self.cim_param['delta_beta'] or 6e-4
        for t in tqdm(range(T_time),leave=False):
            history_a[t] = alpha
            if t % 50 == 49:
                alpha += delta_a
            history_b[t] = beta
            if t % 50 == 24:
                beta += delta_b
            x_ = tr.cos((x_*alpha + noise[::1,t] + 
                         + beta*(J_norm@x_) - tr.pi/4))**2 - 0.5
            traj[t] = x_
            sig = tr.sign(x_)
            sig = sig[-1] * sig[:-1]
            energy_plot_data[t] = tr.matmul(((J[:-1, :-1])/2),sig)@sig + J[:-1, -1].dot(sig)
            if (t > (t/2)) & (t>1000) & ((energy_plot_data[t-400:t]==energy_plot_data[t]).all()):
                k = T_time - t
                # alpha = history_a[tr.argmin(energy_plot_data).item()]
                # beta =  history_b[tr.argmin(energy_plot_data[200:]).item()]
                break
        print(f'alpha={alpha}')
        beta = b
        s = tr.sum(sig > 0).cpu().numpy()
        print(f'Up:{s} alpha={alpha},beta={beta},\nEnergy={energy_plot_data[-1]}')
        x_ = (1e-6*tr.randint(-N, N, size=(N,))).float().ravel().to(device)
        if k == 0 or k == T_time:
            k = T_time//3
        for t in tqdm(range(T_time-k, T_time),leave=False):
            history_b[t] = beta
            if t % 25 == 12:
                beta += delta_b/0.075
            x_ = tr.cos((x_*round(alpha,6) + noise[::1,t]
                         + round(beta,6)*(J_norm@x_) - tr.pi/4))**2 - 0.5
            sig = tr.sign(x_).ravel()
            sig = sig[-1] * sig[:-1]
            energy_plot_data[t] = tr.matmul(((J[:-1, :-1])/2),sig)@sig + J[:-1, -1].dot(sig)
            traj[t] = x_
            if ((t > k/2) & 
                ((energy_plot_data[t-300:t] == energy_plot_data[t]).all())) or t == T_time-1:
                beta = history_b[tr.argmin(energy_plot_data).item()]
                break
        self.all_traj = traj
        print(f'beta={beta}')
        s = tr.sum(sig > 0).cpu().numpy()
        print(f'Up:{s} alpha={alpha},beta={beta},\nEnergy={energy_plot_data[-1]}')
        n_iter = 1000
        return self.cim_poor(alpha, beta, n_iter, min_eigv, opt)

    def bruto_v(self, min_eigv=None):
        """Simple brutoforce search optimal parameters"""
        self._default_params()
        alpha = self.cim_param['alpha_1']
        beta = self.cim_param['beta_1']

        def im(x,*args):
            """Simulation Ideal Ising machine based on Mach-Zender modulator with single parameters

            Parameters
            ----------
                J (np.ndarray): Ising matrix
                alpha (float): strenght of feedback-function
                beta (float): strenght of spin-spin matrix iteraction
                n_iter (int, optional): Amount of iteration. Defaults to 1000.

            Returns
            ----------
                float: enrgy history
            """
            alpha, beta = x
            J, n_iter, scale = args
            J = tr.from_numpy(J).float()
            J_norm = J / scale
            energy = tr.zeros(n_iter)
            noise = tr.normal(mean=tr.zeros((J.shape[0], n_iter)), std=0.01)
            # x = (1e-6*tr.randint(-J.shape[0], J.shape[0], size=(J.shape[0],))).float()
            x = 0.02*tr.randn(J.shape[0]).float()
            sig = tr.sign(x).ravel()
            sig = sig[-1] * sig[:-1]
            energy[0] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
            best_en = 0
            for t in range(1, n_iter):
                x = tr.pow(tr.cos(x * alpha + noise[:, t]
                                  + (beta * tr.matmul(J_norm, x) - tr.pi/4)), 2) - 0.5
                sig = tr.sign(x)
                sig = sig[-1] * sig[:-1]
                energy[t] = sig.dot(sig@J[:-1, :-1])/2 + J[:-1, -1].dot(sig)
                if best_en > energy[t]:
                    best_en = energy[t]
            energy = energy.numpy()
            if (energy[-10:] == energy.min()).sum()>2:
                print(f'alpha={round(alpha,6)}, beta={round(beta,6)}')
                print(f'Min energy {energy[-1]}')
                return best_en
            # else:
            #     print(f'Not work {energy[-1]}')
            #     print(f'alpha={round(alpha,4)}, beta={round(beta,4)}')
            return abs(energy[-1])*100
        J = self.matrix + self.matrix.T
        scale = np.linalg.eigvals(J).min()/min_eigv if min_eigv else 1.0
        a,b = brute(im, ((alpha*0.98, alpha*1.02),(beta*0.1, beta*2)),
                    args=(J, 900, scale), Ns=20)
        return a,b
